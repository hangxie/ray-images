"""Ray Serve voice-cloning service using Qwen3-TTS Base (qwen-tts / PyTorch on CUDA).

Port of site-up's ai/mlx-voice-svc with the same HTTP surface. Serve builds the application on the head with the
image's system Python, whose transformers is vLLM's 5.x; replicas run in the TTS virtualenv (transformers 4.57.3 for
qwen-tts) through runtime_env py_executable. Module-level imports are therefore limited to what both environments
have; torch, transformers and qwen_tts load inside the replica.
"""

import asyncio
import contextlib
import io
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import scipy.signal
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from ray import serve
from starlette.exceptions import HTTPException

ROOT = Path(__file__).resolve().parent
# Sampling options exposed for Base cloning (ICL): name -> (type, fallback default, min, max). Defaults come from the
# model's generation_config.json when it sets them. A None default means the option is derived per request when omitted.
# Unlike mlx-audio, qwen-tts passes repetition_penalty through unchanged in ICL mode, so there is no 1.5 floor.
SAMPLING = {
    "temperature": (float, 0.9, 0.0, 1.5),
    "top_k": (int, 50, 0, 1024),
    "top_p": (float, 1.0, 0.05, 1.0),
    "repetition_penalty": (float, 1.05, 1.0, 3.0),
    "max_tokens": (int, None, 1, 4096),
}
# Normal speech takes 3.6-4.4 codec tokens per text token. Text needing more than max_tokens at the typical rate is
# rejected rather than cut off. Without an explicit max_tokens, generation stops at twice that rate so a runaway is
# bounded by the text length; text that expands when spoken, such as acronyms, can need an explicit max_tokens.
SPEECH_TOKENS_PER_TEXT_TOKEN = 4
SPEECH_TOKEN_BUDGET_PER_TEXT_TOKEN = 8
MIN_SPEECH_TOKEN_BUDGET = 75
# Playback speed factor applied after synthesis: default, min, max. Done here as a pitch-preserving time-stretch
# rather than a model option, matching the MLX service.
SPEED_DEFAULT, SPEED_MIN, SPEED_MAX = 1.0, 0.5, 2.0
# Output loudness in LUFS (ITU-R BS.1770 integrated): default, min, max. Each request is normalized to this target
# so every voice comes out at one volume; a louder target is capped so the waveform never clips.
VOLUME_DEFAULT, VOLUME_MIN, VOLUME_MAX = -20.0, -40.0, -9.0
PEAK_CEILING = 10 ** (-1.0 / 20)  # leave 1 dB of headroom below full scale when gaining up

logger = logging.getLogger("ray.serve")
api = FastAPI()


@api.exception_handler(HTTPException)
async def http_error(request, exc):
    return JSONResponse({"error": "Not found" if exc.status_code == 404 else str(exc.detail)}, exc.status_code,
                        headers={"Cache-Control": "no-store"})


def reply(status, body, media_type="application/json", headers=None):
    headers = {"Cache-Control": "no-store", **{name: str(value) for name, value in (headers or {}).items()}}
    if media_type == "application/json":
        return JSONResponse(body, status, headers=headers)
    return Response(body, status, headers=headers, media_type=media_type)


def settings():
    """Replica settings, read where Serve builds the app (the head), which has the application's env_vars.

    They reach the replica as TTS.bind() arguments: a deployment's runtime_env replaces the application's, so the
    replica process does not see the application's env_vars, and module-level globals there would fall back to defaults.
    """
    return {
        "model": os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        "model_dir": os.environ.get("TTS_MODEL_DIR"),  # local copy of the model; the Hub ID is used when unset
        "voices_dir": os.environ.get("TTS_VOICES_DIR", str(ROOT / "voices")),
        "cuda_graphs": os.environ.get("TTS_CUDA_GRAPHS", "1") != "0",
        "kv_cache_tokens": int(os.environ.get("TTS_KV_CACHE_TOKENS", "3072")),
    }


def replica_runtime_env():
    """The TTS virtualenv's interpreter, plus the build environment's HF_* variables (e.g. HF_HUB_OFFLINE)."""
    env_vars = {name: value for name, value in os.environ.items() if name.startswith("HF_")}
    runtime_env = {"py_executable": os.environ.get("TTS_PYTHON", "/home/ray/tts-venv/bin/python")}
    return {**runtime_env, "env_vars": env_vars} if env_vars else runtime_env


def voices(directory):
    return sorted(p.stem for p in Path(directory).glob("*.wav") if p.with_suffix(".txt").is_file())


def time_stretch(samples, speed, sample_rate, frame_seconds=0.04, tolerance_seconds=0.015):
    """Scale duration by 1/speed while preserving pitch, via WSOLA.

    Output is built from Hann-windowed input frames overlapped at 50%. Each frame is read near its
    nominal position, shifted by up to the tolerance (about one pitch period at 70 Hz) to the offset
    whose waveform best matches the natural continuation of the previous frame, so glottal pulses
    line up across the overlap. Unlike a phase vocoder, no phase is resynthesized, which is what made
    slowed-down speech sound hoarse.
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
    weighted = scipy.signal.lfilter([1.0, -2.0, 1.0], hp_a, scipy.signal.lfilter(shelf_b, shelf_a, samples.astype(np.float64)))
    block, step = round(0.4 * sample_rate), round(0.1 * sample_rate)
    if weighted.size < block:
        return -np.inf
    power = np.array([np.mean(weighted[s:s + block] ** 2) for s in range(0, weighted.size - block + 1, step)])
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


class Job:
    """One synthesis: its stop event, set on client disconnect, and what the talker hooks report back."""

    def __init__(self):
        self.stop = threading.Event()
        self.token_limited = False


def hook_talker(model, current, prepare=None):
    """Add the end-of-speech and disconnect hooks to the talker's first-codebook generate.

    Qwen3TTSForConditionalGeneration.generate builds the talker's kwargs itself and drops unknown ones, so the hooks
    cannot pass through generate_voice_clone. The code predictor's own generate, which fills the other codebooks and
    has no end-of-speech, gets no hooks. `current()` returns the running Job; `prepare`, if given, adjusts the talker's
    generate kwargs (cuda_graphs.install). Recheck this when upgrading qwen-tts.
    """
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

    talker, eos = model.talker, model.config.talker_config.codec_eos_token_id

    class TakeLikelyEos(LogitsProcessor):
        """Stop whenever end-of-speech is the most likely token instead of leaving it to sampling.

        End-of-speech is only likely for a step or two after the last syllable, at roughly 35-55%. A missed draw
        never recovers and runs to max_tokens, so the stop is taken greedily while all other tokens stay sampled.
        Custom processors run after repetition penalty and token suppression and before the sampling warpers, so
        the argmax here is that of the processed scores.
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
        return result

    talker.generate = hooked


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "resources": {"tts": 1}, "runtime_env": replica_runtime_env()},
    num_replicas=int(os.environ.get("TTS_NUM_REPLICAS", "2")),
    # One synthesis per replica: Serve queues the rest and routes each to an idle replica. A backlog beyond
    # max_queued_requests fails fast with 503 instead of growing without bound.
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
        if model.tts_model_type != "base" or speech_tokenizer is None or getattr(speech_tokenizer.model, "encoder", None) is None:
            raise ValueError("Live cloning requires a Base model with its reference encoder")
        defaults = model.generate_config or {}
        # One request at a time per replica (max_ongoing_requests=1), so one slot holds the running job
        self.job = None
        self.cuda_graphs = settings["cuda_graphs"]
        prepare = None
        if self.cuda_graphs:
            from cuda_graphs import install, warm_up

            prepare = install(model, settings["kv_cache_tokens"], defaults)
        hook_talker(model, lambda: self.job, prepare)
        if self.cuda_graphs:
            self.job = Job()
            seconds = warm_up(self.tts)
            logger.info("CUDA graphs captured in %.1f s; talker KV cache %d tokens, %.2f GB CUDA allocated", seconds,
                        settings["kv_cache_tokens"], torch.cuda.memory_allocated() / 1e9)
        # Reference prompts (speech-tokenizer codes + speaker embedding) per voice, encoded once here instead of on
        # every request. Keyed by the pair's mtimes, so a replaced WAV or transcript is re-encoded on next use.
        self.prompts = {}
        started = time.perf_counter()
        for voice in voices(self.voices_dir):
            try:
                self.prompt(voice)
            except Exception as exc:  # e.g. an empty transcript or unreadable WAV: that voice fails per request, as before
                logger.warning("Voice %s not cached: %s", voice, exc)
        logger.info("Encoded %d voice prompts in %.1f s", len(self.prompts), time.perf_counter() - started)
        self.languages = ["auto", *sorted(set(self.tts.get_supported_languages()) - {"auto"})]
        self.sampling = {name: (t, d if d is None or name not in defaults else t(defaults[name]), lo, hi)
                         for name, (t, d, lo, hi) in SAMPLING.items()}
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
            cached = stamp, self.tts.create_voice_clone_prompt(ref_audio=str(wav), ref_text=transcript)
            self.prompts[voice] = cached
        return cached[1]

    @api.get("/")
    def page(self):
        return reply(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")

    @api.get("/health")
    def health(self):
        return reply(200, {
            "status": "ok", "model": self.model_id, "device": self.device, "cuda_graphs": self.cuda_graphs,
            "languages": self.languages,
            "sampling": {name: {"default": d, "min": lo, "max": hi, "integer": t is int} for name, (t, d, lo, hi) in self.sampling.items()},
            "speed": {"default": SPEED_DEFAULT, "min": SPEED_MIN, "max": SPEED_MAX},
            "volume": {"default": VOLUME_DEFAULT, "min": VOLUME_MIN, "max": VOLUME_MAX},
        })

    @api.get("/voices")
    def list_voices(self):
        return reply(200, voices(self.voices_dir))

    @api.post("/api/tts")
    async def synthesize(self, request: Request):
        try:
            body = await request.body()
            if not 0 < len(body) <= 100_000:
                raise ValueError("Request must be between 1 and 100000 bytes")
            req = json.loads(body)
            text, voice = req["text"], req["voice"]
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
            if not SPEED_MIN <= speed <= SPEED_MAX:
                raise ValueError(f"speed must be between {SPEED_MIN} and {SPEED_MAX}")
            if not VOLUME_MIN <= volume <= VOLUME_MAX:
                raise ValueError(f"volume must be between {VOLUME_MIN} and {VOLUME_MAX} LUFS")
            if language not in self.languages:
                raise ValueError(f"Language must be one of: {', '.join(self.languages)}")
            for name, (_, _, low, high) in self.sampling.items():
                if sampling[name] is not None and not low <= sampling[name] <= high:
                    raise ValueError(f"{name} must be between {low} and {high}")
            text_tokens = len(self.tts.processor.tokenizer.encode(text))
            max_tokens = sampling["max_tokens"] or self.sampling["max_tokens"][3]
            if text_tokens * SPEECH_TOKENS_PER_TEXT_TOKEN > max_tokens:
                limit = max_tokens // SPEECH_TOKENS_PER_TEXT_TOKEN
                return reply(413, {"error": f"Text is {text_tokens} tokens; at most {limit} fit in max_tokens", "tokens": text_tokens, "limit": limit})
            if sampling["max_tokens"] is None:
                budget = max(MIN_SPEECH_TOKEN_BUDGET, SPEECH_TOKEN_BUDGET_PER_TEXT_TOKEN * text_tokens)
                sampling["max_tokens"] = min(max_tokens, budget)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return reply(400, {"error": str(exc)})
        job = Job()
        # Shielded so a cancellation reaches the except below while generation keeps running until it sees the event.
        task = asyncio.ensure_future(asyncio.to_thread(self.generate, text, voice, seed, speed, volume, language, sampling, job))
        try:
            return reply(200, *await asyncio.shield(task))
        except asyncio.CancelledError:
            # Serve cancels the request when the HTTP client disconnects. Wait for generation to stop at the next codec
            # frame so this replica is idle before Serve routes it the next queued request.
            job.stop.set()
            logger.info("POST /api/tts abandoned: client disconnected")
            with contextlib.suppress(Exception):
                await task
            raise
        except Exception as exc:
            logger.exception("Synthesis failed")
            return reply(500, {"error": str(exc)})

    def generate(self, text, voice, seed, speed, volume, language, sampling, job):
        """Synthesize in a worker thread; returns reply()'s body, media type and headers."""
        import soundfile as sf
        import torch

        self.job = job
        prompt = self.prompt(voice)
        greedy = sampling["temperature"] == 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Unset options fall back to generation_config.json; greedy decoding leaves temperature at that default unused.
        wavs, sample_rate = self.tts.generate_voice_clone(
            text=text, language="Auto" if language == "auto" else language,
            voice_clone_prompt=prompt,
            do_sample=not greedy, temperature=None if greedy else sampling["temperature"],
            top_k=sampling["top_k"], top_p=sampling["top_p"], repetition_penalty=sampling["repetition_penalty"],
            max_new_tokens=sampling["max_tokens"],
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
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
        output = io.BytesIO()
        sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
        return output.getvalue(), "audio/wav", {
            "X-Synthesis-Seconds": f"{elapsed:.2f}", "X-Audio-Seconds": f"{duration:.2f}",
            "X-RTF": f"{elapsed / duration:.3f}", "X-Token-Limited": str(job.token_limited).lower(),
            "X-Peak-CUDA-GB": f"{torch.cuda.max_memory_allocated() / 1e9:.2f}",
            "X-Loudness-LUFS": f"{loudness:.1f}" if np.isfinite(loudness) else "none",
            "X-Loudness-Limited": str(loudness_limited).lower(),
        }


app = TTS.bind(settings())
