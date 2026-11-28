"""Ray Serve voice cloning with Qwen3-TTS Base behind OpenAI's speech API, plus a test page at `/`."""

import asyncio
import base64
import contextlib
import functools
import html
import io
import json
import logging
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy.signal
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from ray import serve
from starlette.exceptions import HTTPException

from caller_log import CallerLog

ROOT = Path(__file__).resolve().parent
VERSION_FILE = ROOT.parent / "VERSION"  # written by the image build, shared by all the apps
VERSION = VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else "dev"
# name -> (type, default, min, max); generation_config.json overrides defaults, None is derived per request
SAMPLING = {
    "temperature": (float, 0.9, 0.0, 1.5),
    "top_k": (int, 50, 0, 1024),
    "top_p": (float, 1.0, 0.05, 1.0),
    "repetition_penalty": (float, 1.05, 1.0, 3.0),
    "max_tokens": (int, None, 1, 4096),
}
# Speech takes ~4 codec tokens per text token; text needing more than max_tokens is a 413. Without max_tokens,
# generation stops at the budget, so a runaway is bounded by the text length.
SPEECH_TOKENS_PER_TEXT_TOKEN = 4
SPEECH_TOKEN_BUDGET_PER_TEXT_TOKEN = 8
MIN_SPEECH_TOKEN_BUDGET = 75
# speed (a time-stretch after synthesis): default, min, max
SPEED_DEFAULT, SPEED_MIN, SPEED_MAX = 1.0, 0.5, 2.0
# target loudness in LUFS (BS.1770): default, min, max; capped so peaks never clip
VOLUME_DEFAULT, VOLUME_MIN, VOLUME_MAX = -20.0, -40.0, -9.0
PEAK_CEILING = 10 ** (-1.0 / 20)  # leave 1 dB of headroom below full scale when gaining up
# codec frames per streamed piece (12.5 Hz, about a second)
STREAM_FRAMES = 12
# frames the vocoder re-runs before each piece; covers its 5-8 frame receptive field
STREAM_VOCODER_MARGIN = 8

# response_format -> libsndfile container, subtype, media type; no `aac` (no FFmpeg); `pcm` is raw 16-bit LE
OPENAI_FORMATS = {
    "mp3": ("MP3", "MPEG_LAYER_III", "audio/mpeg"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "pcm": (None, None, "audio/pcm"),
}
# no flac: libsndfile writes its header on close
STREAM_FORMATS = ("mp3", "opus", "wav", "pcm")
# CBR (~80 kb/s): a stream has no Xing header, so readers misjudge a VBR one's length
STREAM_MP3_COMPRESSION = 0.5

logger = logging.getLogger("ray.serve")
api = FastAPI()
api.add_middleware(CallerLog)


@api.exception_handler(HTTPException)
async def http_error(request, exc):
    return openai_error(exc.status_code, "Not found" if exc.status_code == 404 else str(exc.detail))


class TextTooLong(ValueError):
    """The text needs more codec tokens than max_tokens allows at the typical rate."""

    def __init__(self, tokens, limit):
        super().__init__(f"Text is {tokens} tokens; at most {limit} fit in max_tokens")
        self.tokens, self.limit = tokens, limit


async def request_json(request):
    body = await request.body()
    if not 0 < len(body) <= 100_000:
        raise ValueError("Request must be between 1 and 100000 bytes")
    return json.loads(body)


def openai_error(status, message, kind="invalid_request_error"):
    """An error in OpenAI's shape, for /v1 routes."""
    return reply(status, {"error": {"message": message, "type": kind, "param": None, "code": None}})


def encode(audio, sample_rate, response_format):
    """16-bit audio in one of OPENAI_FORMATS (all through libsndfile); returns the bytes and their media type."""
    import soundfile as sf

    if response_format == "pcm":
        return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes(), "audio/pcm"
    container, subtype, media_type = OPENAI_FORMATS[response_format]
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format=container, subtype=subtype)
    return output.getvalue(), media_type


def wav_header(sample_rate):
    """A 16-bit mono WAV header for a stream of unknown length (sizes at their maximum)."""
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, 2 * sample_rate, 2, 16)
    return b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " + struct.pack("<I", 16) + fmt + b"data\xff\xff\xff\xff"


class StreamEncoder:
    """Incremental encoder: `feed(pcm)` and `close()` return the newly encoded bytes."""

    def __init__(self, response_format, sample_rate):
        self.file, self.head = None, b""
        if response_format == "wav":
            self.head = wav_header(sample_rate)
        elif response_format != "pcm":
            import soundfile as sf

            container, subtype, _ = OPENAI_FORMATS[response_format]
            mp3 = container == "MP3"
            extra = {"bitrate_mode": "CONSTANT", "compression_level": STREAM_MP3_COMPRESSION} if mp3 else {}
            self.buffer, self.sent = io.BytesIO(), 0
            self.file = sf.SoundFile(self.buffer, "w", sample_rate, 1, subtype, format=container, **extra)

    def feed(self, pcm):
        if self.file is None:
            data, self.head = self.head + pcm, b""
            return data
        self.file.write(np.frombuffer(pcm, dtype="<i2"))
        return self.drain()

    def close(self):
        if self.file is None:
            return b""
        self.file.close()
        return self.drain()

    def drain(self):
        with self.buffer.getbuffer() as view:  # released before libsndfile writes again, so the buffer can grow
            data = bytes(view[self.sent :])
        self.sent += len(data)
        return data


def reply(status, body, media_type="application/json", headers=None):
    headers = {"Cache-Control": "no-store", **{name: str(value) for name, value in (headers or {}).items()}}
    if media_type == "application/json":
        return JSONResponse(body, status, headers=headers)
    return Response(body, status, headers=headers, media_type=media_type)


def settings():
    """Replica settings, read on the head (which has the app's env_vars) and passed through bind()."""
    return {
        "model": os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        "model_dir": os.environ.get("TTS_MODEL_DIR"),  # local copy of the model; the Hub ID is used when unset
        "voices_dir": os.environ.get("TTS_VOICES_DIR", str(ROOT / "voices")),
        "cuda_graphs": os.environ.get("TTS_CUDA_GRAPHS", "1") != "0",
        "kv_cache_tokens": int(os.environ.get("TTS_KV_CACHE_TOKENS", "3072")),
        # Frames of reference + speech the decoder transformer sees per streamed piece; qwen-tts decodes 300 at once.
        "stream_context_frames": int(os.environ.get("TTS_STREAM_CONTEXT_FRAMES", "300")),
    }


def voices(directory):
    return sorted(p.stem for p in Path(directory).glob("*.wav") if p.with_suffix(".txt").is_file())


def time_stretch(samples, speed, sample_rate, frame_seconds=0.04, tolerance_seconds=0.015):
    """Scale duration by 1/speed while preserving pitch, via WSOLA (a phase vocoder sounded hoarse).

    50%-overlapped Hann frames, each shifted up to one 70 Hz pitch period to best continue the previous one.
    """
    if speed == SPEED_DEFAULT or samples.size == 0:
        return samples
    n = 2 * round(frame_seconds * sample_rate / 2)
    hop = n // 2
    tolerance = round(tolerance_seconds * sample_rate)
    window = np.hanning(n + 1)[:n]  # periodic Hann, which sums to one at 50% overlap
    length = round(samples.size / speed)
    padded = np.pad(samples.astype(np.float64), (tolerance, n + tolerance + hop))
    audio = np.zeros(length + n)
    offset = 0  # where the last placed frame starts in the input
    for k in range(1, length // hop + 1):
        audio[(k - 1) * hop : (k - 1) * hop + n] += window * padded[tolerance + offset : tolerance + offset + n]
        continuation = padded[tolerance + offset + hop : tolerance + offset + hop + n]
        low = max(round(k * hop * speed) - tolerance, -tolerance)
        region = padded[tolerance + low : tolerance + low + n + 2 * tolerance]
        offset = low + int(np.argmax(scipy.signal.correlate(region, continuation, mode="valid", method="fft")))
    k = length // hop
    audio[k * hop : k * hop + n] += window * padded[tolerance + offset : tolerance + offset + n]
    return audio[:length].astype(np.float32)


def integrated_loudness(samples, sample_rate):
    """Integrated loudness in LUFS per ITU-R BS.1770 (mono): K-weighting, 400 ms blocks, two-stage gating.

    Returns -inf when the signal is shorter than one block or entirely below the absolute gate.
    """
    # K-weighting: a high-shelf pre-filter cascaded with an RLB high-pass, both designed for this sample rate.
    f0, gain, q = 1681.9744509555319, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sample_rate)
    vh, a0 = 10 ** (gain / 20), 1.0 + k / q + k * k
    vb = vh**0.4996667741545416
    shelf_b = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    shelf_a = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    f0, q = 38.13547087613982, 0.5003270373253953
    k = np.tan(np.pi * f0 / sample_rate)
    a0 = 1.0 + k / q + k * k
    hp_a = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    weighted = scipy.signal.lfilter(
        [1.0, -2.0, 1.0], hp_a, scipy.signal.lfilter(shelf_b, shelf_a, samples.astype(np.float64))
    )
    block, step = round(0.4 * sample_rate), round(0.1 * sample_rate)
    if weighted.size < block:
        return -np.inf
    power = np.array([np.mean(weighted[s : s + block] ** 2) for s in range(0, weighted.size - block + 1, step)])
    loud = -0.691 + 10 * np.log10(np.where(power > 0, power, 1e-12))
    power = power[loud >= -70.0]  # absolute gate
    if power.size:
        power = power[-0.691 + 10 * np.log10(power) >= -0.691 + 10 * np.log10(np.mean(power)) - 10.0]  # relative gate
    return -0.691 + 10 * np.log10(np.mean(power)) if power.size else -np.inf


def normalize_loudness(samples, sample_rate, target):
    """Scale to `target` LUFS with a single gain, capped so the peak stays a dB below full scale.

    Returns the scaled audio, its achieved loudness, and whether the peak cap kept it short of the target.
    """
    measured = integrated_loudness(samples, sample_rate)
    if not np.isfinite(measured):
        return samples, measured, False
    gain_db = target - measured
    peak = float(np.abs(samples).max())
    limited = peak > 0 and gain_db > 20 * np.log10(PEAK_CEILING / peak)
    if limited:
        gain_db = 20 * np.log10(PEAK_CEILING / peak)
    return (samples * 10 ** (gain_db / 20)).astype(np.float32), measured + gain_db, limited


class ClientGone(Exception):
    """The client disconnected, so generation stopped and the rest of the synthesis would be discarded."""


class StreamFinished(Exception):
    """A streamed synthesis has sent all its audio, so generate_voice_clone's full-clip decode is skipped."""


class Job:
    """One synthesis: its stop event, set on client disconnect, and what the talker hooks report back.

    `stream`, when set, is the Streamer that receives each codec frame as the talker completes it.
    """

    def __init__(self):
        self.stop = threading.Event()
        self.token_limited = False
        self.stream = None


def decode_tail(decoder, codes, new, margin):
    """Waveform of the last `new` frames of `codes` (frames x codebooks), as the speech tokenizer decodes them.

    Qwen3TTSTokenizerV2Decoder.forward, with the costly causal vocoder on the last `new + margin` frames only.
    Recheck when upgrading qwen-tts.
    """
    hidden = decoder.quantizer.decode(codes.T[None].clamp(min=0))
    hidden = decoder.pre_conv(hidden).transpose(1, 2)
    hidden = decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state.permute(0, 2, 1)
    hidden = hidden[..., -(new + margin) :]
    for blocks in decoder.upsample:
        for block in blocks:
            hidden = block(hidden)
    for block in decoder.decoder:
        hidden = block(hidden)
    return hidden.clamp(min=-1, max=1)[0, 0, -new * int(decoder.total_upsample) :]


class Streamer:
    """Decodes a streamed synthesis every STREAM_FRAMES codec frames and hands each piece of PCM to `emit`.

    Each piece is decoded after the reference and earlier speech (up to `context` frames). The clip's loudness is
    unknown until the end, so gain comes from the reference recording; peaks past the ceiling clip.
    """

    def __init__(self, decoder, ref_code, sample_rate, context, gain_db, emit):
        self.decoder, self.ref, self.sample_rate, self.context, self.emit = (
            decoder,
            ref_code,
            sample_rate,
            context,
            emit,
        )
        self.gain = 10 ** (gain_db / 20)
        self.frames, self.sent, self.samples, self.clipped = [], 0, 0, 0
        self.first: float | None = None  # perf_counter of the first piece

    def frame(self, codes):
        self.frames.append(codes)
        if len(self.frames) - self.sent >= STREAM_FRAMES:
            self.flush()

    def flush(self):
        import torch

        new = len(self.frames) - self.sent
        if not new:
            return
        window = torch.cat([self.ref.to(self.frames[0].device), torch.stack(self.frames)])[-max(self.context, new) :]
        with torch.inference_mode():
            audio = decode_tail(self.decoder, window, new, min(STREAM_VOCODER_MARGIN, window.shape[0] - new))
        audio = audio.float().cpu().numpy() * self.gain
        if not np.isfinite(audio).all():
            raise ValueError("Model generated non-finite audio")
        self.sent += new
        self.samples += audio.size
        self.clipped += int(np.count_nonzero(np.abs(audio) > PEAK_CEILING))
        pcm = np.round(np.clip(audio, -PEAK_CEILING, PEAK_CEILING) * 32767).astype("<i2")
        self.first = self.first or time.perf_counter()
        self.emit({"type": "audio", "pcm": pcm.tobytes()})


def hook_talker(model, current, prepare=None):
    """Add the end-of-speech and disconnect hooks to the talker's first-codebook generate.

    generate_voice_clone drops unknown kwargs, so the hooks can't pass through it. `current()` returns the running Job;
    `prepare` adjusts the talker's generate kwargs (cuda_graphs.install). Recheck when upgrading qwen-tts.
    """
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

    talker, eos = model.talker, model.config.talker_config.codec_eos_token_id

    class TakeLikelyEos(LogitsProcessor):
        """Stop whenever end-of-speech is the most likely token instead of leaving it to sampling.

        It peaks at 35-55% for a step or two, and a missed draw runs to max_tokens.
        """

        def __call__(self, input_ids, scores):
            only_eos = torch.full_like(scores, -float("inf"))
            only_eos[:, eos] = scores[:, eos]
            return torch.where((scores.argmax(dim=-1) == eos)[:, None], only_eos, scores)

    class StopWhenClientGone(StoppingCriteria):
        """Abandon a synthesis at the next codec frame once its client disconnects."""

        def __init__(self, stop):
            self.stop = stop

        def __call__(self, input_ids, scores, **kwargs):
            return torch.full((input_ids.shape[0],), self.stop.is_set(), dtype=torch.bool, device=input_ids.device)

    forward = talker.forward

    @functools.wraps(forward)  # generate validates its kwargs against forward's signature
    def captured(*args, **kwargs):
        """Hand each completed codec frame to a streamed job. A decode step's codec_ids are the frame whose first code
        was sampled in the step before, completed by the code predictor; prefill has none."""
        outputs = forward(*args, **kwargs)
        job, codes = current(), outputs.hidden_states[1]
        if job is not None and job.stream is not None and codes is not None:
            job.stream.frame(codes[0])
        return outputs

    talker.forward = captured
    generate = talker.generate

    def hooked(*args, logits_processor=None, stopping_criteria=None, **kwargs):
        job = current()
        if prepare is not None:
            kwargs = prepare(kwargs)
        result = generate(
            *args,
            logits_processor=LogitsProcessorList([*(logits_processor or []), TakeLikelyEos()]),
            stopping_criteria=StoppingCriteriaList([*(stopping_criteria or []), StopWhenClientGone(job.stop)]),
            **kwargs,
        )
        if job.stop.is_set():
            raise ClientGone  # skip decoding codes nobody will receive
        job.token_limited = not bool((result.sequences == eos).any())
        if job.stream is not None:
            job.stream.flush()
            raise StreamFinished  # every frame has been decoded and sent
        return result

    talker.generate = hooked


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "resources": {"tts": 1}},
    num_replicas=int(os.environ.get("TTS_NUM_REPLICAS", "2")),
    # one synthesis per replica; past max_queued_requests, 503
    max_ongoing_requests=1,
    max_queued_requests=int(os.environ.get("TTS_MAX_QUEUED_REQUESTS", "4")),
)
@serve.ingress(api)
class TTS:
    def __init__(self, settings):
        import torch
        from qwen_tts import Qwen3TTSModel

        self.model_id, self.voices_dir = settings["model"], Path(settings["voices_dir"])
        path = settings["model_dir"] or self.model_id
        logger.info("Loading %s from %s; voices from %s", self.model_id, path, self.voices_dir)
        self.tts = Qwen3TTSModel.from_pretrained(path, device_map="cuda:0", dtype=torch.bfloat16)
        model = self.tts.model
        speech_tokenizer = model.speech_tokenizer
        if (
            model.tts_model_type != "base"
            or speech_tokenizer is None
            or getattr(speech_tokenizer.model, "encoder", None) is None
        ):
            raise ValueError("Live cloning requires a Base model with its reference encoder")
        defaults = model.generate_config or {}
        # One request at a time per replica (max_ongoing_requests=1), so one slot holds the running job
        self.job = None
        self.cuda_graphs = settings["cuda_graphs"]
        self.stream_context = settings["stream_context_frames"]
        prepare = None
        if self.cuda_graphs:
            from cuda_graphs import install, warm_up

            prepare = install(model, settings["kv_cache_tokens"], defaults)
        hook_talker(model, lambda: self.job, prepare)
        if self.cuda_graphs:
            self.job = Job()
            seconds = warm_up(self.tts)
            logger.info(
                "CUDA graphs captured in %.1f s; talker KV cache %d tokens, %.2f GB CUDA allocated",
                seconds,
                settings["kv_cache_tokens"],
                torch.cuda.memory_allocated() / 1e9,
            )
        # voice prompts, encoded once; keyed by the files' mtimes so a replaced voice is re-encoded
        self.prompts: dict[str, tuple[tuple[int, int], Any, Any]] = {}
        started = time.perf_counter()
        for voice in voices(self.voices_dir):
            try:
                self.prompt(voice)
            except (
                Exception
            ) as exc:  # e.g. an empty transcript or unreadable WAV: that voice fails per request, as before
                logger.warning("Voice %s not cached: %s", voice, exc)
        logger.info("Encoded %d voice prompts in %.1f s", len(self.prompts), time.perf_counter() - started)
        self.languages = ["auto", *sorted(set(self.tts.get_supported_languages()) - {"auto"})]
        self.sampling = {
            name: (t, d if d is None or name not in defaults else t(defaults[name]), lo, hi)
            for name, (t, d, lo, hi) in SAMPLING.items()
        }
        self.device = f"{torch.cuda.get_device_name(model.device)} / CUDA {torch.version.cuda}"

    def prompt(self, voice):
        """The cached voice-clone prompt for `voice`, encoding it on first use or after its files change."""
        wav, txt = self.voices_dir / f"{voice}.wav", self.voices_dir / f"{voice}.txt"
        stamp = (wav.stat().st_mtime_ns, txt.stat().st_mtime_ns)
        cached = self.prompts.get(voice)
        if cached is None or cached[0] != stamp:
            transcript = txt.read_text(encoding="utf-8")
            if not transcript.strip():
                raise ValueError("Reference transcript is empty")
            cached = stamp, self.tts.create_voice_clone_prompt(ref_audio=str(wav), ref_text=transcript), None
            self.prompts[voice] = cached
        return cached[1]

    def reference_loudness(self, voice):
        """Integrated loudness of a voice's reference WAV, cached with its prompt; streamed requests gain from it."""
        import soundfile as sf

        self.prompt(voice)
        stamp, prompt, loudness = self.prompts[voice]
        if loudness is None:
            audio, rate = sf.read(self.voices_dir / f"{voice}.wav", dtype="float32", always_2d=True)
            loudness = integrated_loudness(audio.mean(axis=1), rate)
            self.prompts[voice] = stamp, prompt, loudness
        return loudness

    @api.get("/")
    def page(self):
        page = (ROOT / "index.html").read_text().replace("{{VERSION}}", html.escape(VERSION))
        return reply(200, page.encode(), "text/html; charset=utf-8")

    @api.get("/v1/health")
    def health(self):
        return reply(
            200,
            {
                "status": "ok",
                "version": VERSION,
                "model": self.model_id,
                "device": self.device,
                "cuda_graphs": self.cuda_graphs,
                "languages": self.languages,
                "sampling": {
                    name: {"default": d, "min": lo, "max": hi, "integer": t is int}
                    for name, (t, d, lo, hi) in self.sampling.items()
                },
                "speed": {"default": SPEED_DEFAULT, "min": SPEED_MIN, "max": SPEED_MAX},
                "volume": {"default": VOLUME_DEFAULT, "min": VOLUME_MIN, "max": VOLUME_MAX},
            },
        )

    @api.get("/v1/audio/voices")
    def list_voices(self):
        """Voices in the OpenAI-style TTS server shape, for clients such as Open WebUI."""
        return reply(200, {"voices": [{"id": name, "name": name} for name in voices(self.voices_dir)]})

    @api.get("/v1/models")
    def models(self):
        return reply(200, {"object": "list", "data": [{"id": self.model_id, "object": "model", "owned_by": "local"}]})

    @api.post("/v1/audio/speech")
    async def speech(self, request: Request):
        """OpenAI's speech API, plus `seed`, `language`, `volume` and sampling; `model` and `instructions` are ignored.

        Without `stream_format`: the whole clip with `X-*` headers. `audio` streams bytes, `sse` OpenAI's events.
        """
        try:
            req = await request_json(request)
            if not isinstance(req, dict):
                raise ValueError("Request must be a JSON object")
            response_format = req.get("response_format", "mp3")
            stream_format = req.get("stream_format")
            if response_format not in OPENAI_FORMATS:
                raise ValueError(f"response_format must be one of: {', '.join(OPENAI_FORMATS)}")
            if stream_format not in (None, "audio", "sse"):
                raise ValueError("stream_format must be audio or sse")
            if stream_format and response_format not in STREAM_FORMATS:
                raise ValueError(f"A streamed response_format must be one of: {', '.join(STREAM_FORMATS)}")
            options = self.options(req, stream=stream_format is not None)
        except KeyError as exc:
            return openai_error(400, f"Missing field {exc}")
        except TextTooLong as exc:
            return openai_error(413, str(exc))
        except (ValueError, TypeError, AttributeError) as exc:
            return openai_error(400, str(exc))
        if stream_format:
            return await self.stream_speech(options, response_format, stream_format)
        options.pop("text_tokens")
        try:
            return reply(200, *await self.clip(options, response_format))
        except Exception as exc:
            return openai_error(500, str(exc), "server_error")

    async def clip(self, options, response_format):
        """generate() in a worker thread; returns reply()'s body, media type and headers.

        Shielded so a cancellation reaches the except below while generation keeps running until it sees the event.
        """
        job = Job()
        task = asyncio.ensure_future(
            asyncio.to_thread(self.generate, **options, response_format=response_format, job=job)
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Serve cancels on disconnect: wait for generation to stop so the replica is free for the next request.
            job.stop.set()
            logger.info("POST /v1/audio/speech abandoned: client disconnected")
            with contextlib.suppress(Exception):
                await task
            raise
        except Exception:
            logger.exception("Synthesis failed")
            raise

    def options(self, req, stream=False):
        """Validate a speech request; returns generate()'s arguments plus `text_tokens`.

        Raises ValueError (TextTooLong past max_tokens), or TypeError, KeyError or AttributeError for a bad field.
        """
        text, voice = req["input"], req["voice"]
        seed = int(req.get("seed", 0))
        speed = float(req.get("speed", SPEED_DEFAULT))
        volume = float(req.get("volume", VOLUME_DEFAULT))
        language = req.get("language", "auto")
        sampling = {name: d if req.get(name) is None else t(req[name]) for name, (t, d, _, _) in self.sampling.items()}
        if not isinstance(text, str) or not text.strip() or len(text) > 20_000:
            raise ValueError("Enter between 1 and 20000 characters")
        if voice not in voices(self.voices_dir):
            raise ValueError("Select a reference voice with an exact transcript")
        if not 0 <= seed < 2**32:
            raise ValueError("Seed must be between 0 and 4294967295")
        if stream and speed != SPEED_DEFAULT:
            raise ValueError("speed is not supported when streaming")
        if not SPEED_MIN <= speed <= SPEED_MAX:
            raise ValueError(f"speed must be between {SPEED_MIN} and {SPEED_MAX}")
        if not VOLUME_MIN <= volume <= VOLUME_MAX:
            raise ValueError(f"volume must be between {VOLUME_MIN} and {VOLUME_MAX} LUFS")
        if language not in self.languages:
            raise ValueError(f"Language must be one of: {', '.join(self.languages)}")
        for name, (_, _, low, high) in self.sampling.items():
            value = sampling[name]
            if value is not None and not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
        text_tokens = len(self.tts.processor.tokenizer.encode(text))
        max_tokens = sampling["max_tokens"] or self.sampling["max_tokens"][3]
        if text_tokens * SPEECH_TOKENS_PER_TEXT_TOKEN > max_tokens:
            raise TextTooLong(text_tokens, max_tokens // SPEECH_TOKENS_PER_TEXT_TOKEN)
        if sampling["max_tokens"] is None:
            budget = max(MIN_SPEECH_TOKEN_BUDGET, SPEECH_TOKEN_BUDGET_PER_TEXT_TOKEN * text_tokens)
            sampling["max_tokens"] = min(max_tokens, budget)
        return {
            "text": text,
            "voice": voice,
            "seed": seed,
            "speed": speed,
            "volume": volume,
            "language": language,
            "sampling": sampling,
            "text_tokens": text_tokens,
        }

    async def stream_speech(self, options, response_format, stream_format):
        """Reply as speech is generated: encoded bytes (`audio`) or SSE events (`sse`).

        It starts at the first piece, so earlier failures are an error status; later ones an SSE `error` or a cut.
        """
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        job = Job()
        text_tokens = options.pop("text_tokens")
        options.pop("speed")

        def emit(event):
            loop.call_soon_threadsafe(events.put_nowait, event)

        task = asyncio.ensure_future(asyncio.to_thread(self.generate_stream, **options, job=job, emit=emit))

        async def finish(abandoned):
            if abandoned:
                # client left: stop so the replica is free for the next request
                job.stop.set()
                logger.info("POST /v1/audio/speech stream abandoned: client disconnected")
            with contextlib.suppress(Exception):
                await asyncio.shield(task)

        try:
            first = await events.get()
        except asyncio.CancelledError:
            await finish(abandoned=True)
            raise
        if first["type"] == "error":
            await finish(abandoned=False)
            return openai_error(500, first["error"], "server_error")
        sample_rate = self.tts.model.speech_tokenizer.get_output_sample_rate()
        encoder = StreamEncoder(response_format, sample_rate)
        sse = stream_format == "sse"

        def delta(data):
            return f"data: {json.dumps({'type': 'speech.audio.delta', 'audio': base64.b64encode(data).decode()})}\n\n"

        async def body():
            event, finished = first, False
            try:
                while True:
                    if event["type"] == "audio":
                        if data := encoder.feed(event["pcm"]):
                            yield delta(data) if sse else data
                    elif event["type"] == "done":
                        finished = True
                        if data := encoder.close():
                            yield delta(data) if sse else data
                        if sse:
                            stats = {k: v for k, v in event.items() if k not in ("type", "frames")}
                            usage = {"input_tokens": text_tokens, "output_tokens": event["frames"]}
                            usage["total_tokens"] = text_tokens + event["frames"]
                            yield f"data: {json.dumps({'type': 'speech.audio.done', 'usage': usage, **stats})}\n\n"
                        break
                    else:
                        finished = True
                        if not sse:
                            raise RuntimeError(event["error"])  # ends the response unfinished, so the client sees it
                        error = {"message": event["error"], "type": "server_error", "param": None, "code": None}
                        yield f"data: {json.dumps({'type': 'error', 'error': error})}\n\n"
                        break
                    event = await events.get()
            finally:
                # not `task.done()`: the thread emits `done` just before it returns
                await finish(abandoned=not finished)
                with contextlib.suppress(Exception):
                    encoder.close()

        media_type = "text/event-stream" if sse else OPENAI_FORMATS[response_format][2]
        headers = {"Cache-Control": "no-store", "X-Sample-Rate": str(sample_rate)}
        return StreamingResponse(body(), media_type=media_type, headers=headers)

    def generate_stream(self, text, voice, seed, volume, language, sampling, job, emit):
        """Synthesize a streamed request in a worker thread, emitting its events through a Streamer."""
        import torch

        try:
            loudness = self.reference_loudness(voice)
            gain_db = volume - loudness if np.isfinite(loudness) else 0.0
            speech_tokenizer = self.tts.model.speech_tokenizer
            job.stream = streamer = Streamer(
                speech_tokenizer.model.decoder,
                self.prompt(voice)[0].ref_code,
                speech_tokenizer.get_output_sample_rate(),
                self.stream_context,
                gain_db,
                emit,
            )
            try:
                self.run(text, voice, seed, language, sampling, job)
                raise ValueError("Generation ended without streaming")  # the talker hook always raises StreamFinished
            except StreamFinished:
                pass
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - self.started
            if not streamer.samples:
                raise ValueError("Model generated no audio")
            assert streamer.first is not None  # set with the first audio sample
            duration = streamer.samples / streamer.sample_rate
            emit(
                {
                    "type": "done",
                    "synthesis_seconds": round(elapsed, 2),
                    "first_audio_seconds": round(streamer.first - self.started, 2),
                    "audio_seconds": round(duration, 2),
                    "rtf": round(elapsed / duration, 3),
                    "token_limited": job.token_limited,
                    "peak_cuda_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "gain_db": round(gain_db, 1),
                    "clipped_samples": streamer.clipped,
                    "frames": streamer.sent,
                }
            )
        except ClientGone:
            pass
        except Exception as exc:
            logger.exception("Streamed synthesis failed")
            emit({"type": "error", "error": str(exc)})

    def run(self, text, voice, seed, language, sampling, job):
        """generate_voice_clone for `job`, seeded, with peak memory reset; `self.started` marks its start.

        Unset options fall back to generation_config.json; greedy decoding leaves temperature at that default unused.
        """
        import torch

        self.job = job
        prompt = self.prompt(voice)
        greedy = sampling["temperature"] == 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.started = time.perf_counter()
        return self.tts.generate_voice_clone(
            text=text,
            language="Auto" if language == "auto" else language,
            voice_clone_prompt=prompt,
            do_sample=not greedy,
            temperature=None if greedy else sampling["temperature"],
            top_k=sampling["top_k"],
            top_p=sampling["top_p"],
            repetition_penalty=sampling["repetition_penalty"],
            max_new_tokens=sampling["max_tokens"],
        )

    def generate(self, text, voice, seed, speed, volume, language, sampling, job, response_format):
        """Synthesize in a worker thread; returns reply()'s body, media type and headers."""
        import torch

        wavs, sample_rate = self.run(text, voice, seed, language, sampling, job)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        audio = np.asarray(wavs[0], dtype=np.float32) if wavs else np.zeros(0, dtype=np.float32)
        if not np.isfinite(audio).all():
            raise ValueError("Model generated non-finite audio")
        if not audio.size:
            raise ValueError("Model generated no audio")
        audio = time_stretch(audio, speed, sample_rate)
        duration = len(audio) / sample_rate
        if duration <= 0:
            raise ValueError("Model generated empty audio")
        audio, loudness, loudness_limited = normalize_loudness(audio, sample_rate, volume)
        body, media_type = encode(audio, sample_rate, response_format)
        return (
            body,
            media_type,
            {
                "X-Sample-Rate": sample_rate,
                "X-Synthesis-Seconds": f"{elapsed:.2f}",
                "X-Audio-Seconds": f"{duration:.2f}",
                "X-RTF": f"{elapsed / duration:.3f}",
                "X-Token-Limited": str(job.token_limited).lower(),
                "X-Peak-CUDA-GB": f"{torch.cuda.max_memory_allocated() / 1e9:.2f}",
                "X-Loudness-LUFS": f"{loudness:.1f}" if np.isfinite(loudness) else "none",
                "X-Loudness-Limited": str(loudness_limited).lower(),
            },
        )


# mypy ignores class decorators (python/mypy#3135)
app = cast(serve.Deployment, TTS).bind(settings())
