"""Ray Serve voice chat: speech -> STT -> streamed LLM reply -> TTS per sentence, streamed back to the browser.

Port of site-up's media/voice-chat with the same HTTP surface: `GET /` (chat.html), `GET /health`, `GET /api/voices`,
and `POST /api/chat` answering NDJSON events. It holds no models: STT, the LLM, and TTS are URLs, so replicas need a CPU
and the system Python only. The pipeline is the original's: worker threads talk to the backends with http.client, and
an interrupted turn shuts their sockets. What changed is the server (a FastAPI ingress instead of http.server, with
Serve cancelling the response when the client disconnects), settings (read on the head and passed through bind()), and
the client-IP allowlist, which is gone: access control belongs to the deployment in front, as for TTS and STT.
"""

import asyncio
import base64
import contextlib
import http.client
import json
import logging
import os
import queue
import re
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from ray import serve
from starlette.exceptions import HTTPException

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = (
    "You are a friendly voice assistant in a spoken conversation. Reply in the language the user speaks. "
    "Keep answers brief, one to three short sentences, unless asked for more detail. "
    "Your words are read aloud, so use plain spoken sentences only: "
    "no markdown, lists, headings, code, emoji, or URLs. "
    "Write numbers, units, and symbols the way they are said."
)
HISTORY_MESSAGES = 20  # earlier turns are dropped so prompt processing stays short
MAX_REQUEST_BYTES = 10 * 1024 * 1024
BACKEND_TIMEOUT = 120
# mlx-tts-svc answers 409 and Ray Serve 503/429 while busy, such as during a txt2m4b run; wait this long for a slot.
TTS_BUSY_WAIT = 20
TTS_BUSY_STATUSES = (409, 429, 503)

# A chunk ends at sentence punctuation; ASCII stops need a following space so "3.14" stays whole. The first chunk may
# also end at a comma once it is long enough, so playback starts sooner, and so may any chunk that grows too long.
# The full-width marks are CJK punctuation, deliberately (RUF001).
HARD_STOPS = set("。！？；…\n")  # noqa: RUF001
SOFT_STOPS = set(".!?;")
PAUSES = set("，,、：:")  # noqa: RUF001
CLOSERS = set("\"'”’」』）)]")  # noqa: RUF001
TRAILING = CLOSERS | HARD_STOPS | SOFT_STOPS  # kept with the stop they follow, as in "……" or "?!"
FIRST_CHUNK_CHARS = 12
MAX_CHUNK_CHARS = 150
MIN_SPOKEN_CHARS = 2

MARKDOWN = [
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),  # [text](url) -> text
    (re.compile(r"https?://\S+"), ""),
    (re.compile(r"^\s{0,3}(#{1,6}|[-*+]|\d+[.)])\s+", re.M), ""),  # headings, bullets, numbered items
    (re.compile(r"[*_`~#|>]+"), ""),
    (re.compile("[\U0001f000-\U0001faff☀-➿️‍]"), ""),  # emoji and their joiners
    (re.compile(r"\s+"), " "),
]

logger = logging.getLogger("ray.serve")
api = FastAPI()


@api.exception_handler(HTTPException)
async def http_error(request, exc):
    return JSONResponse(
        {"error": "Not found" if exc.status_code == 404 else str(exc.detail)},
        exc.status_code,
        headers={"Cache-Control": "no-store"},
    )


def reply(status, body, media_type="application/json"):
    headers = {"Cache-Control": "no-store"}
    if media_type == "application/json":
        return JSONResponse(body, status, headers=headers)
    return Response(body, status, headers=headers, media_type=media_type)


def settings():
    """Replica settings, read where Serve builds the app (the head) and passed through bind(), as in serve_tts."""
    env = os.environ.get
    return {
        "stt_url": env("CHAT_STT_URL", "http://localhost:5005/v1").rstrip("/"),
        "llm_url": env("CHAT_LLM_URL", "http://localhost:8080/v1").rstrip("/"),
        "llm_model": env("CHAT_LLM_MODEL", "qwen3.5-9b-q8-uncensored"),
        "llm_max_tokens": int(env("CHAT_LLM_MAX_TOKENS", "512")),
        "tts_url": env("CHAT_TTS_URL", "http://localhost:5004").rstrip("/"),
        "voice": env("CHAT_VOICE", ""),
        "system_prompt": env("CHAT_SYSTEM_PROMPT") or SYSTEM_PROMPT,
        # Sent as `Authorization: Bearer` to that backend; empty sends none.
        "tokens": {name: env(f"CHAT_{name}_TOKEN", "") for name in ("STT", "LLM", "TTS")},
    }


def spoken(text):
    """Text with the markdown, links, and emoji the model still sometimes writes removed, for TTS."""
    for pattern, replacement in MARKDOWN:
        text = pattern.sub(replacement, text)
    return text.strip()


class Chunker:
    """Splits streamed LLM text into chunks small enough to synthesize while the rest is still generating."""

    def __init__(self):
        self.buffer, self.first = "", True

    def feed(self, text, final=False):
        self.buffer += text
        chunks, start = [], 0
        i = 0
        while i < len(self.buffer):
            c, length = self.buffer[i], i + 1 - start
            cut = (
                c in HARD_STOPS
                or (c in SOFT_STOPS and (i + 1 < len(self.buffer) and self.buffer[i + 1].isspace()))
                or (c in PAUSES and length >= (FIRST_CHUNK_CHARS if self.first else MAX_CHUNK_CHARS))
            )
            if cut:
                end = i + 1
                while end < len(self.buffer) and self.buffer[end] in TRAILING:
                    end += 1
                if end == len(self.buffer) and not final:
                    break  # the next token may still continue this run of punctuation
                if len(spoken(self.buffer[start:end])) >= MIN_SPOKEN_CHARS:
                    chunks.append(self.buffer[start:end])
                    start, self.first = end, False
                i = end
            else:
                i += 1
        self.buffer = self.buffer[start:]
        if final and spoken(self.buffer):
            chunks.append(self.buffer)
            self.buffer = ""
        return [chunk for chunk in map(spoken, chunks) if chunk]


class BackendError(Exception):
    def __init__(self, name, status, message):
        super().__init__(f"{name} returned {status}: {message}")
        self.status = status


class Turn:
    """One request's pipeline: events for the browser, and the backend connections to cut on interrupt.

    The backend workers are threads; `emit` hands their events to the response's event loop.
    """

    def __init__(self, loop, tokens):
        self.loop, self.tokens = loop, tokens
        self.events: asyncio.Queue[dict] = asyncio.Queue()
        self.cancelled = threading.Event()
        # Each connection's socket, kept here because http.client drops `connection.sock` in getresponse() when the
        # response closes the connection (HTTP/1.0, `Connection: close`) while the response still reads from it.
        self.connections: dict[http.client.HTTPConnection, socket.socket] = {}
        self.lock = threading.Lock()

    def emit(self, **event):
        self.loop.call_soon_threadsafe(self.events.put_nowait, event)

    def request(self, name, url, body, content_type, timeout=BACKEND_TIMEOUT):
        """POST to a backend and return its response once the status is 2xx; the connection is tracked until closed."""
        connection, parts = connect(url, timeout)
        headers = {"Content-Type": content_type, **auth(self.tokens[name])}
        connection.connect()
        with self.lock:
            if self.cancelled.is_set():
                connection.close()
                raise ConnectionAbortedError("turn cancelled")
            self.connections[connection] = connection.sock
        connection.request("POST", parts.path + (f"?{parts.query}" if parts.query else ""), body, headers)
        response = connection.getresponse()
        if not 200 <= response.status < 300:
            detail = response.read(2000).decode(errors="replace")
            try:
                error = json.loads(detail).get("error", detail)
                detail = error.get("message", detail) if isinstance(error, dict) else error
            except (ValueError, AttributeError):
                pass
            self.close(connection)
            raise BackendError(name, response.status, detail)
        return connection, response

    def close(self, connection):
        with self.lock:
            self.connections.pop(connection, None)
        connection.close()

    def cancel(self):
        """Abort every in-flight backend request; TTS drops a synthesis whose client disconnects."""
        with self.lock:
            self.cancelled.set()
            sockets = list(self.connections.values())
        for sock in sockets:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)


def auth(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def connect(url, timeout):
    parts = urlsplit(url)
    cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    return cls(parts.netloc, timeout=timeout), parts


def tts_get(config, path):
    """GET a JSON document from the TTS service."""
    connection, parts = connect(config["tts_url"], 10)
    try:
        connection.request("GET", parts.path + path, headers=auth(config["tokens"]["TTS"]))
        return json.loads(connection.getresponse().read())
    finally:
        connection.close()


def transcribe(config, turn, audio):
    boundary = uuid.uuid4().hex
    body = b"".join(
        [
            f'--{boundary}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\nverbose_json\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nwhisper\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n".encode(),
            audio,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    connection, response = turn.request(
        "STT", f"{config['stt_url']}/audio/transcriptions", body, f"multipart/form-data; boundary={boundary}"
    )
    try:
        return json.loads(response.read())
    finally:
        turn.close(connection)


def run_llm(config, turn, messages, chunks):
    """Stream the reply as text events and hand finished chunks to the TTS worker; None marks the end."""
    reply = ""
    body = json.dumps(
        {
            "model": config["llm_model"],
            "messages": messages,
            "stream": True,
            "max_tokens": config["llm_max_tokens"],
            # Thinking would be a long silent pause before the first word.
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    try:
        for attempt in (1, 2):  # one retry, but only before any text has arrived
            try:
                connection, response = turn.request(
                    "LLM", f"{config['llm_url']}/chat/completions", body, "application/json"
                )
                break
            except (OSError, BackendError) as exc:
                if attempt == 2 or turn.cancelled.is_set() or (isinstance(exc, BackendError) and exc.status < 500):
                    raise
                time.sleep(1)
        chunker = Chunker()
        try:
            while line := response.readline():
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                choices = json.loads(data).get("choices") or [{}]
                text = (choices[0].get("delta") or {}).get("content")
                if text:
                    if not reply:
                        turn.emit(type="llm_first_token")
                    reply += text
                    turn.emit(type="text", delta=text)
                    for chunk in chunker.feed(text):
                        chunks.put(chunk)
        finally:
            turn.close(connection)
        for chunk in chunker.feed("", final=True):
            chunks.put(chunk)
    except Exception as exc:
        if not turn.cancelled.is_set():
            turn.emit(type="error", stage="llm", message=str(exc))
    finally:
        turn.emit(type="llm_done", reply=reply)
        chunks.put(None)


def run_tts(config, turn, chunks, voice, language):
    """Synthesize chunks one at a time, in order, since a TTS replica serves one request at a time.

    Each chunk is requested as a stream and its audio forwarded piece by piece as it is generated, so playback starts
    about a second into the chunk instead of after all of it. A backend that answers with a whole WAV still works.
    """
    failed = False
    try:
        index = 0
        while (text := chunks.get()) is not None:
            if failed or turn.cancelled.is_set():
                continue  # drain until the LLM worker finishes
            body = json.dumps({"text": text, "voice": voice, "language": language, "stream": True}).encode()
            deadline = time.monotonic() + TTS_BUSY_WAIT
            try:
                while True:
                    try:
                        connection, response = turn.request(
                            "TTS", f"{config['tts_url']}/api/tts", body, "application/json"
                        )
                        break
                    except BackendError as exc:
                        if (
                            exc.status not in TTS_BUSY_STATUSES
                            or time.monotonic() > deadline
                            or turn.cancelled.is_set()
                        ):
                            raise
                        time.sleep(0.3)
                try:
                    if response.getheader("Content-Type", "").startswith("application/x-ndjson"):
                        stream_tts(turn, response, index, text)
                    else:
                        wav = response.read()
                        rtf = response.getheader("X-RTF")
                        seconds = response.getheader("X-Audio-Seconds")
                        turn.emit(
                            type="audio",
                            index=index,
                            text=text,
                            wav=base64.b64encode(wav).decode(),
                            seconds=float(seconds) if seconds else None,
                            rtf=float(rtf) if rtf else None,
                        )
                finally:
                    turn.close(connection)
                index += 1
            except Exception as exc:
                failed = True
                if not turn.cancelled.is_set():
                    turn.emit(type="error", stage="tts", message=str(exc))
    finally:
        turn.emit(type="tts_done")


def stream_tts(turn, response, index, text):
    """Forward each streamed piece of 16-bit PCM as an `audio` event; the chunk's text rides on its first piece."""
    part = 0
    for line in response:
        if turn.cancelled.is_set():
            return
        event = json.loads(line)
        if event["type"] == "audio":
            seconds = len(base64.b64decode(event["pcm"])) / 2 / event["sample_rate"]
            turn.emit(
                type="audio",
                index=index,
                part=part,
                text=text if part == 0 else None,
                pcm=event["pcm"],
                sample_rate=event["sample_rate"],
                seconds=round(seconds, 2),
            )
            part += 1
        elif event["type"] == "error":
            raise BackendError("TTS", 500, event["error"])
        elif event["type"] == "done":
            return
    raise BackendError("TTS", 502, "stream ended without finishing")


def reply_language(text, detected=None):
    """Language to answer in: from the script of the text when it is unambiguous, else the STT-detected language."""
    if re.search(r"[぀-ヿ]", text):
        return "japanese"
    if re.search(r"[가-힯]", text):
        return "korean"
    if re.search(r"[一-鿿]", text):
        return "chinese"
    return detected or None


def system_prompt(prompt, language):
    """The system prompt with this turn's reply language, which a small model otherwise drops once the history
    holds a reply in another language and copies that pattern instead."""
    if language:
        return f"{prompt}\nThe user's latest message is in {language.capitalize()}; reply in {language.capitalize()}."
    return f"{prompt}\nReply in the language of the user's latest message, whatever language earlier turns used."


def history(messages):
    """Validated user/assistant turns from the browser, most recent last."""
    if not isinstance(messages, list):
        raise ValueError("history must be a list")
    turns = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
        and m["content"]
    ]
    return turns[-HISTORY_MESSAGES:]


def line(event):
    return json.dumps(event, ensure_ascii=False) + "\n"


@serve.deployment(
    num_replicas=int(os.environ.get("CHAT_NUM_REPLICAS", "1")),
    # Each turn is one streamed request that mostly waits on STT, the LLM, and TTS, so a replica holds many at once.
    max_ongoing_requests=int(os.environ.get("CHAT_MAX_ONGOING_REQUESTS", "16")),
)
@serve.ingress(api)
class Chat:
    def __init__(self, settings):
        self.config = settings
        self.languages = None  # TTS's languages, read from its /health on first use

    def tts_languages(self):
        """Languages the TTS service accepts, read once from its /health."""
        if self.languages is None:
            try:
                self.languages = set(tts_get(self.config, "/health").get("languages") or ())
            except (OSError, ValueError):
                return set()
        return self.languages

    @api.get("/")
    def page(self):
        return reply(200, (ROOT / "chat.html").read_bytes(), "text/html; charset=utf-8")

    @api.get("/health")
    def health(self):
        c = self.config
        return reply(
            200,
            {
                "status": "ok",
                "stt": c["stt_url"],
                "llm": c["llm_url"],
                "llm_model": c["llm_model"],
                "tts": c["tts_url"],
                "voice": c["voice"],
            },
        )

    @api.get("/api/voices")
    async def voices(self):
        try:
            voices = await asyncio.to_thread(tts_get, self.config, "/voices")
        except (OSError, ValueError) as exc:
            return reply(502, {"error": f"TTS unavailable: {exc}"})
        default = self.config["voice"]
        return reply(
            200, {"voices": voices, "default": default if default in voices else (voices[0] if voices else "")}
        )

    @api.post("/api/chat")
    async def chat(self, request: Request):
        started = time.perf_counter()
        try:
            body = await request.body()
            if not 0 < len(body) <= MAX_REQUEST_BYTES:
                raise ValueError(f"Request must be between 1 and {MAX_REQUEST_BYTES} bytes")
            req = json.loads(body)
            audio = base64.b64decode(req["audio"], validate=True) if req.get("audio") else None
            text = req.get("text") if isinstance(req.get("text"), str) else ""
            if not audio and not text.strip():
                raise ValueError("Send audio or text")
            voice = req.get("voice") or self.config["voice"]
            if not isinstance(voice, str) or not voice:
                raise ValueError("Select a voice")
            past = history(req.get("history", []))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return reply(400, {"error": str(exc)})
        turn = Turn(asyncio.get_running_loop(), self.config["tokens"])
        return StreamingResponse(
            self.converse(turn, started, audio, text, voice, past),
            media_type="application/x-ndjson; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    async def converse(self, turn, started, audio, text, voice, past):
        """The turn's events, ending with `done`; Serve cancels this when the client disconnects."""
        timings, finished = {}, False
        try:
            detected = None
            if audio:
                try:
                    result = await asyncio.to_thread(transcribe, self.config, turn, audio)
                except Exception as exc:
                    finished = True
                    yield line({"type": "error", "stage": "stt", "message": str(exc)})
                    return
                timings["stt"] = round(time.perf_counter() - started, 2)
                text = (result.get("text") or "").strip()
                detected = result.get("language")
                yield line({"type": "transcript", "text": text, "language": detected, "seconds": timings["stt"]})
                if not text:
                    finished = True
                    yield line({"type": "done", "reply": "", "timings": timings})
                    return
            reply_in = reply_language(text, detected)
            language = reply_in if reply_in in await asyncio.to_thread(self.tts_languages) else "auto"
            messages = [
                {"role": "system", "content": system_prompt(self.config["system_prompt"], reply_in)},
                *past,
                {"role": "user", "content": text},
            ]
            chunks: queue.Queue[str | None] = queue.Queue()
            threading.Thread(target=run_llm, args=(self.config, turn, messages, chunks), daemon=True).start()
            threading.Thread(target=run_tts, args=(self.config, turn, chunks, voice, language), daemon=True).start()
            reply_text = ""
            while True:
                event = await turn.events.get()
                kind = event["type"]
                elapsed = round(time.perf_counter() - started, 2)
                if kind == "llm_first_token":
                    timings["llm_first_token"] = elapsed
                elif kind == "llm_done":
                    reply_text = event["reply"]
                    timings["llm_done"] = elapsed
                elif kind == "tts_done":
                    break
                else:
                    if kind == "audio" and "first_audio" not in timings:
                        timings["first_audio"] = event["first_audio"] = elapsed
                    yield line(event)
            finished = True
            yield line({"type": "done", "reply": reply_text, "timings": timings})
        finally:
            turn.cancel()
            if not finished:
                logger.info("POST /api/chat interrupted: client disconnected")


# mypy ignores class decorators (python/mypy#3135), so it sees Chat as the plain class rather than a Deployment
app = cast(serve.Deployment, Chat).bind(settings())
