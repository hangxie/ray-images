"""Ray Serve voice chat: speech -> STT -> streamed LLM reply -> TTS per sentence, streamed back to the browser.

Port of site-up's media/voice-chat with the same HTTP surface: `GET /` (chat.html), `GET /health`, `GET /api/voices`,
and `POST /api/chat` answering NDJSON events. It holds no models: STT, the LLM, and TTS are URLs (all OpenAI-style `/v1`
APIs), so replicas need a CPU and the system Python only. The pipeline is the original's: worker threads talk to the
backends with http.client, and an interrupted turn shuts their sockets. What changed is the server (a FastAPI ingress
instead of http.server, with Serve cancelling the response when the client disconnects), settings (read on the head and
passed through bind()), and the client-IP allowlist, which is gone: access control belongs to the deployment in front,
as for TTS and STT.
"""

import asyncio
import base64
import contextlib
import html
import http.client
import json
import logging
import os
import queue
import re
import select
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

from caller_log import CallerLog

ROOT = Path(__file__).resolve().parent
VERSION_FILE = ROOT.parent / "VERSION"  # written by the image build, shared by all the apps
VERSION = VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else "dev"
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
# Ray Serve answers 503 (or 429) while TTS is busy, such as during a txt2m4b run; wait this long for a slot.
TTS_BUSY_WAIT = 20
TTS_BUSY_STATUSES = (429, 503)
TTS_SAMPLE_RATE = 24000  # OpenAI's pcm rate, for a TTS that doesn't send X-Sample-Rate
POOL_IDLE_PER_ORIGIN = 8  # idle keep-alive connections kept per backend; more are closed
DRAIN_TIMEOUT = 5  # how long to wait for the rest of a TTS stream after its `done` before giving up on reuse

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
api.add_middleware(CallerLog)


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
        "tts_url": env("CHAT_TTS_URL", "http://localhost:5004/v1").rstrip("/"),
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


class Pool:
    """Idle keep-alive backend connections, shared by a replica's turns.

    Every STT call and TTS sentence used to open its own connection, and through an HTTPS Gateway the TLS handshake
    was most of each call's overhead. A connection goes back here only after its response was read to the end.
    """

    def __init__(self):
        self.idle: dict[tuple[str, str], list[http.client.HTTPConnection]] = {}
        self.lock = threading.Lock()

    def get(self, url, timeout):
        """A connection for `url`'s origin and whether it is a reused one; a new one is not connected yet."""
        parts = urlsplit(url)
        key = (parts.scheme, parts.netloc)
        while True:
            with self.lock:
                connection = self.idle.get(key, []).pop() if self.idle.get(key) else None
            if connection is None:
                cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
                return cls(parts.netloc, timeout=timeout), parts, False
            if alive(connection):
                connection.timeout = timeout
                cast(socket.socket, connection.sock).settimeout(timeout)
                return connection, parts, True
            connection.close()

    def put(self, connection, parts):
        with self.lock:
            idle = self.idle.setdefault((parts.scheme, parts.netloc), [])
            if len(idle) < POOL_IDLE_PER_ORIGIN:
                idle.append(connection)
                return
        connection.close()


def alive(connection):
    """Whether an idle connection can take a request: a readable socket means the server closed it (or sent junk)."""
    sock = connection.sock
    if sock is None:
        return False
    try:
        return not select.select([sock], [], [], 0)[0]
    except (OSError, ValueError):
        return False


# What a request on a pooled connection the server has already closed fails with; a timeout is not one of them, since
# the server may still be working on the request.
STALE = (ConnectionError, http.client.BadStatusLine)


def reusable(response):
    return response.isclosed() and not response.will_close


class Turn:
    """One request's pipeline: events for the browser, and the backend connections to cut on interrupt.

    The backend workers are threads; `emit` hands their events to the response's event loop.
    """

    def __init__(self, loop, tokens, pool):
        self.loop, self.tokens, self.pool = loop, tokens, pool
        self.events: asyncio.Queue[dict] = asyncio.Queue()
        self.cancelled = threading.Event()
        # Each connection's socket, kept here because http.client drops `connection.sock` in getresponse() when the
        # response closes the connection (HTTP/1.0, `Connection: close`) while the response still reads from it.
        self.connections: dict[http.client.HTTPConnection, tuple[socket.socket, object]] = {}
        self.lock = threading.Lock()

    def emit(self, **event):
        self.loop.call_soon_threadsafe(self.events.put_nowait, event)

    def request(self, name, url, body, content_type, timeout=BACKEND_TIMEOUT):
        """POST to a backend and return its response once the status is 2xx; the connection is tracked until closed."""
        headers = {"Content-Type": content_type, **auth(self.tokens[name])}
        while True:
            connection, parts, reused = self.pool.get(url, timeout)
            if not reused:
                connection.connect()
            with self.lock:
                if self.cancelled.is_set():
                    connection.close()
                    raise ConnectionAbortedError("turn cancelled")
                self.connections[connection] = (cast(socket.socket, connection.sock), parts)
            try:
                connection.request("POST", parts.path + (f"?{parts.query}" if parts.query else ""), body, headers)
                response = connection.getresponse()
                break
            except (OSError, http.client.HTTPException) as exc:
                self.close(connection)
                # A reused connection the server closed before the request arrived: try again on a new one.
                if not reused or self.cancelled.is_set() or not isinstance(exc, STALE):
                    raise
        if not 200 <= response.status < 300:
            detail = response.read(2000).decode(errors="replace")
            try:
                error = json.loads(detail).get("error", detail)
                detail = error.get("message", detail) if isinstance(error, dict) else error
            except (ValueError, AttributeError):
                pass
            self.close(connection, response)
            raise BackendError(name, response.status, detail)
        return connection, response

    def close(self, connection, response=None):
        """Done with a connection: back to the pool when `response` was read to the end, else closed."""
        with self.lock:
            _, parts = self.connections.pop(connection, (None, None))
            keep = response is not None and parts is not None and reusable(response) and not self.cancelled.is_set()
        if keep:
            self.pool.put(connection, parts)
        else:
            connection.close()

    def cancel(self):
        """Abort every in-flight backend request; TTS drops a synthesis whose client disconnects."""
        with self.lock:
            self.cancelled.set()
            sockets = [sock for sock, _ in self.connections.values()]
        for sock in sockets:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)


def auth(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def tts_get(pool, config, path):
    """GET a JSON document from the TTS service; a non-2xx answer (busy, down) raises BackendError."""
    while True:
        connection, parts, reused = pool.get(config["tts_url"], 10)
        try:
            connection.request("GET", parts.path + path, headers=auth(config["tokens"]["TTS"]))
            response = connection.getresponse()
            body = response.read()
        except (OSError, http.client.HTTPException) as exc:
            connection.close()
            if not reused or not isinstance(exc, STALE):
                raise
            continue
        if reusable(response):
            pool.put(connection, parts)
        else:
            connection.close()
        if not 200 <= response.status < 300:
            raise BackendError("TTS", response.status, body[:200].decode(errors="replace"))
        return json.loads(body)


async def read_body(request, limit):
    """The request body, refused past `limit` bytes before reading them when Content-Length says so, and while reading
    them otherwise (chunked uploads), so an oversized upload is never held in memory whole."""
    too_large = ValueError(f"Request must be between 1 and {limit} bytes")
    declared = request.headers.get("content-length")
    if declared is not None and (not declared.isdigit() or int(declared) > limit):
        raise too_large
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise too_large
    if not body:
        raise too_large
    return bytes(body)


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
        turn.close(connection, response)


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
        chunker, finished = Chunker(), False
        try:
            while line := response.readline():
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    finished = True
                    break
                event = json.loads(data)
                if event.get("error"):  # a failure after the 200, as llama.cpp and LiteLLM send it
                    error = event["error"]
                    raise BackendError("LLM", 200, error.get("message", error) if isinstance(error, dict) else error)
                choices = event.get("choices") or [{}]
                text = (choices[0].get("delta") or {}).get("content")
                if text:
                    if not reply:
                        turn.emit(type="llm_first_token")
                    reply += text
                    turn.emit(type="text", delta=text)
                    for chunk in chunker.feed(text):
                        chunks.put(chunk)
        finally:
            turn.close(connection, response)
        if not finished:
            raise BackendError("LLM", 200, "stream ended before [DONE]")
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

    Each chunk is requested as an SSE stream of 16-bit PCM and its audio forwarded piece by piece as it is generated,
    so playback starts about a second into the chunk instead of after all of it.
    """
    failed = False
    try:
        index = 0
        while (text := chunks.get()) is not None:
            if failed or turn.cancelled.is_set():
                continue  # drain until the LLM worker finishes
            request = {"input": text, "voice": voice, "language": language, "response_format": "pcm"}
            body = json.dumps({**request, "stream_format": "sse"}).encode()
            deadline = time.monotonic() + TTS_BUSY_WAIT
            try:
                while True:
                    try:
                        connection, response = turn.request(
                            "TTS", f"{config['tts_url']}/audio/speech", body, "application/json"
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
                    if not response.getheader("Content-Type", "").startswith("text/event-stream"):
                        raise BackendError("TTS", response.status, "answered without an event stream")
                    if stream_tts(turn, response, index, text):
                        finish(connection, response)
                finally:
                    turn.close(connection, response)
                index += 1
            except Exception as exc:
                failed = True
                if not turn.cancelled.is_set():
                    turn.emit(type="error", stage="tts", message=str(exc))
    finally:
        turn.emit(type="tts_done")


def finish(connection, response):
    """Read the rest of a stream after its last event (the chunked end), so its connection can go back to the pool
    and TTS sees the stream finish instead of an abandoned one."""
    with contextlib.suppress(OSError, http.client.HTTPException, ValueError):
        if connection.sock is not None:
            connection.sock.settimeout(DRAIN_TIMEOUT)
        response.read()


def stream_tts(turn, response, index, text):
    """Forward each streamed piece of 16-bit PCM as an `audio` event; the chunk's text rides on its first piece.

    True once the stream's `speech.audio.done` arrives, False when the turn was cancelled first."""
    sample_rate = int(response.getheader("X-Sample-Rate") or TTS_SAMPLE_RATE)
    part = 0
    for line in response:
        if turn.cancelled.is_set():
            return False
        if not line.startswith(b"data:"):
            continue  # the blank line ending each event
        event = json.loads(line[5:])
        if event["type"] == "speech.audio.delta":
            pcm = event["audio"]
            seconds = len(base64.b64decode(pcm)) / 2 / sample_rate
            turn.emit(
                type="audio",
                index=index,
                part=part,
                text=text if part == 0 else None,
                pcm=pcm,
                sample_rate=sample_rate,
                seconds=round(seconds, 2),
            )
            part += 1
        elif event["type"] == "error":
            raise BackendError("TTS", 500, event["error"]["message"])
        elif event["type"] == "speech.audio.done":
            return True
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
        self.pool = Pool()
        self.languages = None  # TTS's languages, read from its /health on first use

    def tts_languages(self):
        """Languages the TTS service accepts, read from its /health once it answers; until then each turn asks again."""
        if self.languages is None:
            try:
                health = tts_get(self.pool, self.config, "/health")
                if not isinstance(health, dict):
                    raise ValueError("TTS /health is not a JSON object")
                self.languages = set(health.get("languages") or ())
            except (OSError, ValueError, BackendError):
                return set()
        return self.languages

    @api.get("/")
    def page(self):
        page = (ROOT / "chat.html").read_text().replace("{{VERSION}}", html.escape(VERSION))
        return reply(200, page.encode(), "text/html; charset=utf-8")

    @api.get("/health")
    def health(self):
        c = self.config
        return reply(
            200,
            {
                "status": "ok",
                "version": VERSION,
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
            listed = await asyncio.to_thread(tts_get, self.pool, self.config, "/audio/voices")
            voices = [v["id"] for v in listed["voices"]]
            if not all(isinstance(v, str) for v in voices):
                raise ValueError("TTS /audio/voices has a voice without a string id")
        except (OSError, ValueError, TypeError, KeyError, BackendError) as exc:
            return reply(502, {"error": f"TTS unavailable: {exc}"})
        default = self.config["voice"]
        return reply(
            200, {"voices": voices, "default": default if default in voices else (voices[0] if voices else "")}
        )

    @api.post("/api/chat")
    async def chat(self, request: Request):
        started = time.perf_counter()
        try:
            req = json.loads(await read_body(request, MAX_REQUEST_BYTES))
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
        turn = Turn(asyncio.get_running_loop(), self.config["tokens"], self.pool)
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
