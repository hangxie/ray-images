"""Ray Serve speech-to-text service: Whisper (transformers / PyTorch on CUDA) behind OpenAI's transcription API.

Port of site-up's ai/mlx-stt-svc with the same HTTP surface under the application's route prefix (/v1), plus a
recording page at `GET /` (stt.html) and a WebSocket at `/audio/stream` that transcribes a live recording as it goes.
It runs in the TTS virtualenv beside serve_tts, which already has transformers (4.57.3, with Whisper), soundfile and
scipy, so the image needs no packages for it. As in serve_tts, module-level imports are limited to what both
environments have.
"""

import asyncio
import collections
import email.parser
import email.policy
import html
import io
import json
import logging
import os
import time
from math import gcd
from pathlib import Path
from typing import cast

import numpy as np
import scipy.signal
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response
from ray import serve
from starlette.exceptions import HTTPException

ROOT = Path(__file__).resolve().parent
VERSION = (ROOT / "VERSION").read_text().strip() if (ROOT / "VERSION").exists() else "dev"  # written by the image build
SAMPLE_RATE = 16_000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # OpenAI's own limit for /v1/audio/transcriptions
MAX_AUDIO_SECONDS = 300
WINDOW_FRAMES = 3000  # Whisper's 30 s window in feature frames; longer audio is decoded long-form with timestamps
PROMPT_TOKENS = 160  # previous text kept as Whisper's prompt; Whisper takes at most 223 (half its 448 positions)

logger = logging.getLogger("ray.serve")
api = FastAPI()


@api.exception_handler(HTTPException)
async def http_error(request, exc):
    message = "Not found" if exc.status_code == 404 else str(exc.detail)
    return JSONResponse({"error": {"message": message}}, exc.status_code, headers={"Cache-Control": "no-store"})


def reply(status, body, media_type="application/json", headers=None):
    headers = {"Cache-Control": "no-store", **(headers or {})}
    if media_type == "application/json":
        return JSONResponse(body, status, headers=headers)
    return Response(body, status, headers=headers, media_type=media_type)


def error(status, message):
    return reply(status, {"error": {"message": message}})


def settings():
    """Replica settings, read on the head and passed through bind(), as in serve_tts."""
    return {
        "model": os.environ.get("STT_MODEL", "openai/whisper-large-v3-turbo"),
        "model_dir": os.environ.get("STT_MODEL_DIR"),  # local copy of the model; the Hub ID is used when unset
    }


def replica_runtime_env():
    """The TTS virtualenv's interpreter, plus the build environment's HF_* variables (e.g. HF_HUB_OFFLINE)."""
    env_vars = {name: value for name, value in os.environ.items() if name.startswith("HF_")}
    runtime_env = {"py_executable": os.environ.get("STT_PYTHON", "/home/ray/tts-venv/bin/python")}
    return {**runtime_env, "env_vars": env_vars} if env_vars else runtime_env


def multipart_fields(content_type, body):
    """Form fields of a multipart/form-data body: name -> bytes. Stdlib only, so no python-multipart is needed."""
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + body
    )
    if not message.is_multipart():
        raise ValueError("Request must be multipart/form-data")
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
        for part in message.iter_parts()
    }


def load_audio(data):
    """Mono float32 at 16 kHz from any format libsndfile reads (WAV, FLAC, OGG, MP3)."""
    import soundfile as sf

    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except sf.LibsndfileError as exc:
        raise ValueError(f"Unsupported or corrupt audio: {exc}") from exc
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        common = gcd(rate, SAMPLE_RATE)
        audio = scipy.signal.resample_poly(audio, SAMPLE_RATE // common, rate // common).astype(np.float32)
    return audio


class Segmenter:
    """Cuts a live 16 kHz stream into pieces for Whisper at short pauses, so text keeps up with long speech.

    Past `min_seconds` the next `pause_seconds` of quiet ends a piece; at `max_seconds` it ends after the quietest
    20 ms frame past `min_seconds`. Speech and quiet are judged against the noise floor, the lowest 20 ms level of the
    last `floor_seconds` (speech always has gaps between syllables), so they don't depend on the microphone's gain;
    only levels under -54 dBFS never count as speech. A piece is trimmed to its speech plus `margin_seconds`, and one
    with less than `min_speech_seconds` of speech is dropped (Whisper makes up text from silence). A piece starts a
    paragraph after `paragraph_seconds` without speech. Times are seconds from the start of the stream.
    """

    FRAME = SAMPLE_RATE // 50
    MIN_SPEECH_LEVEL = 0.002  # -54 dBFS
    MIN_FLOOR = 0.0005

    def __init__(
        self,
        min_seconds=5.0,
        max_seconds=15.0,
        pause_seconds=0.25,
        paragraph_seconds=2.0,
        min_speech_seconds=0.3,
        margin_seconds=0.3,
        floor_seconds=5.0,
    ):
        def frames(seconds):
            return max(1, int(np.ceil(seconds * 50 - 1e-6)))

        self.min_frames, self.max_frames = frames(min_seconds), frames(max_seconds)
        self.pause_frames, self.margin_frames = frames(pause_seconds), frames(margin_seconds)
        self.min_speech_frames, self.paragraph_seconds = frames(min_speech_seconds), paragraph_seconds
        self.audio = np.zeros(0, dtype=np.float32)  # the current piece, starting on a frame boundary
        self.levels: list[float] = []  # RMS of each whole frame of the current piece
        self.recent: collections.deque[float] = collections.deque(maxlen=frames(floor_seconds))
        self.start = 0  # stream sample where the current piece starts
        self.floor = self.MIN_FLOOR
        self.last_speech: float | None = None  # stream time where the previous piece's speech ended
        self.peak = 0.0  # the loudest frame so far, for the stream's log line

    def speech(self, level):
        return level > max(self.MIN_SPEECH_LEVEL, self.floor * 3)

    def quiet(self, level):
        return level < max(self.MIN_SPEECH_LEVEL * 0.75, self.floor * 2)

    def feed(self, samples):
        """Pieces completed by `samples`: dicts with start, end, audio, and paragraph."""
        self.audio = np.concatenate([self.audio, np.asarray(samples, dtype=np.float32)])
        pieces = []
        while len(self.audio) >= (len(self.levels) + 1) * self.FRAME:
            i = len(self.levels)
            level = float(np.sqrt(np.mean(np.square(self.audio[i * self.FRAME : (i + 1) * self.FRAME]))))
            self.levels.append(level)
            self.recent.append(level)
            self.floor, self.peak = max(min(self.recent), self.MIN_FLOOR), max(self.peak, level)
            n = len(self.levels)
            if n >= self.min_frames and all(self.quiet(v) for v in self.levels[-self.pause_frames :]):
                pieces += self.cut(n)
            elif n >= self.max_frames:
                pieces += self.cut(self.min_frames + int(np.argmin(self.levels[self.min_frames :])) + 1)
        return pieces

    def flush(self):
        """The rest of the stream as a last piece, if it has speech."""
        return self.cut(len(self.levels), whole=True)

    def cut(self, frames, whole=False):
        end = len(self.audio) if whole else frames * self.FRAME
        audio, levels, start = self.audio[:end], self.levels[:frames], self.start
        self.audio, self.levels, self.start = self.audio[end:], self.levels[frames:], start + end
        voiced = [i for i, level in enumerate(levels) if self.speech(level)]
        if len(voiced) < self.min_speech_frames:
            return []
        first = max(0, voiced[0] - self.margin_frames) * self.FRAME
        last = min(len(audio), (voiced[-1] + 1 + self.margin_frames) * self.FRAME)
        speech_start = (start + voiced[0] * self.FRAME) / SAMPLE_RATE
        paragraph = self.last_speech is None or speech_start - self.last_speech >= self.paragraph_seconds
        self.last_speech = (start + (voiced[-1] + 1) * self.FRAME) / SAMPLE_RATE
        piece = {
            "start": (start + first) / SAMPLE_RATE,
            "end": (start + last) / SAMPLE_RATE,
            "audio": audio[first:last],
        }
        return [{**piece, "paragraph": paragraph}]


def control(message):
    """The type of a text message on the stream; anything that isn't a JSON object has none."""
    try:
        body = json.loads(message)
    except ValueError:
        return None
    return body.get("type") if isinstance(body, dict) else None


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "runtime_env": replica_runtime_env()},
    num_replicas=int(os.environ.get("STT_NUM_REPLICAS", "1")),
    # A live stream holds its request for as long as it lasts, so several run at once and self.gpu takes
    # transcriptions one at a time. Utterances take well under a second, so requests wait instead of failing; past
    # the queue they get 503.
    max_ongoing_requests=int(os.environ.get("STT_MAX_ONGOING_REQUESTS", "16")),
    max_queued_requests=int(os.environ.get("STT_MAX_QUEUED_REQUESTS", "8")),
)
@serve.ingress(api)
class STT:
    def __init__(self, settings):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE

        self.model_id = settings["model"]
        path = settings["model_dir"] or self.model_id
        logger.info("Loading %s from %s", self.model_id, path)
        self.processor = WhisperProcessor.from_pretrained(path)
        self.model = WhisperForConditionalGeneration.from_pretrained(path, dtype=torch.float16).to("cuda:0").eval()
        self.languages, self.language_codes = LANGUAGES, TO_LANGUAGE_CODE
        self.id_to_language = {i: token.strip("<|>") for token, i in self.model.generation_config.lang_to_id.items()}
        self.device = f"{torch.cuda.get_device_name(0)} / CUDA {torch.version.cuda}"
        self.gpu = asyncio.Lock()
        # Warm up so the first real request does not pay for CUDA kernel selection.
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), None, "Warm up.")
        logger.info("STT ready; %.2f GB CUDA allocated", torch.cuda.memory_allocated() / 1e9)

    @api.get("/")
    def page(self):
        page = (ROOT / "stt.html").read_text().replace("{{VERSION}}", html.escape(VERSION))
        return reply(200, page.encode(), "text/html; charset=utf-8")

    @api.get("/health")
    def health(self):
        return reply(200, {"status": "ok", "version": VERSION, "model": self.model_id, "device": self.device})

    @api.get("/models")
    def models(self):
        return reply(200, {"object": "list", "data": [{"id": self.model_id, "object": "model", "owned_by": "local"}]})

    @api.post("/audio/transcriptions")
    async def transcriptions(self, request: Request):
        try:
            body = await request.body()
            if not 0 < len(body) <= MAX_UPLOAD_BYTES:
                raise ValueError(f"Request must be between 1 and {MAX_UPLOAD_BYTES} bytes")
            fields = multipart_fields(request.headers.get("content-type", ""), body)
            if not fields.get("file"):
                raise ValueError("Missing audio file field")
            response_format = (fields.get("response_format") or b"json").decode()
            if response_format not in ("json", "verbose_json", "text"):
                raise ValueError("response_format must be json, verbose_json, or text")
            language = self.language((fields.get("language") or b"").decode())
            audio = load_audio(fields["file"])
            duration = audio.size / SAMPLE_RATE
            if not 0 < duration <= MAX_AUDIO_SECONDS:
                raise ValueError(f"Audio must be between 0 and {MAX_AUDIO_SECONDS} seconds")
        except (ValueError, TypeError, UnicodeDecodeError, RuntimeError) as exc:
            return error(400, str(exc))
        try:
            started = time.perf_counter()
            text, language = await self.run(audio, language)
            elapsed = time.perf_counter() - started
        except Exception as exc:
            logger.exception("Transcription failed")
            return error(500, str(exc))
        headers = {
            "X-Transcribe-Seconds": f"{elapsed:.2f}",
            "X-Audio-Seconds": f"{duration:.2f}",
            "X-RTF": f"{elapsed / duration:.3f}",
        }
        if response_format == "text":
            return reply(200, text.encode(), "text/plain; charset=utf-8", headers)
        return reply(
            200,
            {"text": text, "language": self.languages.get(language, language), "duration": round(duration, 2)},
            headers=headers,
        )

    @api.websocket("/audio/stream")
    async def stream(self, websocket: WebSocket):
        """Live transcription of one long recording.

        The client sends 16 kHz mono 16-bit little-endian PCM as binary messages, then `{"type": "stop"}`. The server
        cuts it with a Segmenter and sends, in order, a `segment` event (start, end, text, language, paragraph,
        seconds) for each piece with text, an `error` event for a piece that failed, and `done` after the last one.
        Each piece gets the text before it as its prompt, and keeps the language of the first piece with text.
        """
        await websocket.accept()
        try:
            language = self.language(websocket.query_params.get("language", ""))
        except ValueError as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(1008)
            return
        segmenter, pieces = Segmenter(), asyncio.Queue[dict | None]()
        counts = {"pieces": 0, "segments": 0}

        async def transcriber():
            nonlocal language
            context = ""
            while (piece := await pieces.get()) is not None:
                counts["pieces"] += 1
                try:
                    started = time.perf_counter()
                    text, found = await self.run(piece["audio"], language, context)
                    elapsed = time.perf_counter() - started
                except Exception as exc:
                    logger.exception("Transcription failed")
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue
                if not text:
                    continue
                language, context = found, f"{context} {text}".strip()[-2000:]
                event = {"type": "segment", "start": round(piece["start"], 2), "end": round(piece["end"], 2)}
                event |= {"text": text, "language": self.languages.get(found, found), "paragraph": piece["paragraph"]}
                await websocket.send_json({**event, "seconds": round(elapsed, 2)})
                counts["segments"] += 1

        worker = asyncio.create_task(transcriber())
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    data = message["bytes"]
                    samples = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
                    for piece in segmenter.feed(samples):
                        pieces.put_nowait(piece)
                elif control(message.get("text") or "") == "stop":
                    for piece in segmenter.flush():
                        pieces.put_nowait(piece)
                    pieces.put_nowait(None)
                    await worker
                    await websocket.send_json({"type": "done"})
                    await websocket.close()
                    break
        except Exception:
            logger.exception("Live transcription failed")
        finally:
            worker.cancel()
            logger.info(
                "Stream ended: %.1f s of audio, %d pieces, %d with text, peak %.1f dBFS, noise floor %.1f dBFS",
                (segmenter.start + len(segmenter.audio)) / SAMPLE_RATE,
                counts["pieces"],
                counts["segments"],
                20 * np.log10(segmenter.peak + 1e-9),
                20 * np.log10(segmenter.floor),
            )

    def language(self, value):
        """Whisper's language code for a code or English name, or None (detect) when empty."""
        language = value.strip().lower() or None
        if language is not None:
            language = self.language_codes.get(language, language)
            if language not in self.languages:
                raise ValueError(f"Unknown language: {language}")
        return language

    async def run(self, audio, language, prompt=""):
        async with self.gpu:
            work = asyncio.ensure_future(asyncio.to_thread(self.transcribe, audio, language, prompt))
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:  # e.g. the stream closed; the thread can't be stopped, so keep the GPU
                await asyncio.wait([work])
                raise

    def transcribe(self, audio, language, prompt=""):
        """Transcript and language code; the language is detected from the first 30 s when not given. `prompt` is
        what was said before this audio, which keeps words split across pieces and spellings consistent."""
        import torch

        extract = self.processor.feature_extractor  # type: ignore[attr-defined]  # set by ProcessorMixin
        inputs = extract(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        long_form = inputs.input_features.shape[-1] > WINDOW_FRAMES
        if not long_form:  # short audio is padded to the 30 s window the model was trained on
            inputs = extract(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
        features = inputs.input_features.to("cuda:0", torch.float16)
        prompt_ids = None
        if prompt.strip():  # <|startofprev|> and the end of the text
            ids = self.processor.get_prompt_ids(prompt, return_tensors="pt")  # type: ignore[attr-defined]
            prompt_ids = torch.cat([ids[:1], ids[1:][-PROMPT_TOKENS:]]).to("cuda:0")
        with torch.inference_mode():
            if language is None:
                detected = self.model.detect_language(input_features=features[..., :WINDOW_FRAMES])
                language = self.id_to_language[int(detected[0])]
            tokens = self.model.generate(
                features,
                attention_mask=inputs.attention_mask.to("cuda:0"),
                language=language,
                task="transcribe",
                return_timestamps=long_form,
                prompt_ids=prompt_ids,
            )
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip(), language


# mypy ignores class decorators (python/mypy#3135), so it sees STT as the plain class rather than a Deployment
app = cast(serve.Deployment, STT).bind(settings())
