"""serve_chat's routes and turn pipeline against fake STT, LLM, and TTS servers on loopback.

Everything but the backends is real: the FastAPI routes, the worker threads and their http.client requests, chunking,
and cancellation.
"""

import asyncio
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import pytest
from conftest import serve_client

import serve_chat


class Backends:
    """Fake STT (/v1), LLM (/llm), and TTS (/tts) on one loopback server, recording what they were sent."""

    def __init__(self):
        self.transcript = {"text": "Hello.", "language": "english"}
        self.stt_status = 200
        self.llm_status = 200
        self.reply = ["Hello there. ", "How are ", "you today?"]
        self.tts_busy = 0  # answer this many TTS requests with 503 first
        self.tts_wav = False  # answer with a whole WAV instead of NDJSON
        self.tts_hold = None  # an Event: after the first piece, wait for the client to hang up and set it
        self.discovery: list[int] = []  # statuses for the next TTS /health and /voices requests, then 200
        self.requests: list[tuple[str, Any]] = []
        backends = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, body, content_type="application/json", headers=None):
                body = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                backends.requests.append(("GET " + self.path, self.headers.get("Authorization")))
                if backends.discovery:
                    return self.send(backends.discovery.pop(0), {"error": "busy"})
                if self.path == "/tts/health":
                    return self.send(200, {"status": "ok", "languages": ["auto", "chinese", "english"]})
                if self.path == "/tts/voices":
                    return self.send(200, ["alice", "bob"])
                self.send(404, {"error": "Not found"})

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path == "/v1/audio/transcriptions":
                    backends.requests.append(("stt", body))
                    if backends.stt_status != 200:
                        return self.send(backends.stt_status, {"error": {"message": "STT broke"}})
                    return self.send(200, backends.transcript)
                payload = json.loads(body)
                if self.path == "/llm/chat/completions":
                    backends.requests.append(("llm", payload))
                    if backends.llm_status != 200:
                        return self.send(backends.llm_status, {"error": {"message": "bad model"}})
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for delta in backends.reply:
                        event = {"choices": [{"delta": {"content": delta}}]}
                        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    return None
                backends.requests.append(("tts", payload))
                if backends.tts_busy:
                    backends.tts_busy -= 1
                    return self.send(503, {"error": "busy"})
                if backends.tts_wav:
                    return self.send(200, b"RIFF-wav", "audio/wav", {"X-RTF": "0.5", "X-Audio-Seconds": "1.25"})
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                pcm = base64.b64encode(b"\0\0" * 2400).decode()  # 0.1 s at 24 kHz
                for _ in range(2):
                    self.wfile.write(json.dumps({"type": "audio", "sample_rate": 24000, "pcm": pcm}).encode() + b"\n")
                    self.wfile.flush()
                    if backends.tts_hold is not None:
                        self.connection.recv(1)  # returns once the client shuts the connection
                        backends.tts_hold.set()
                        return None
                self.wfile.write(b'{"type": "done"}\n')
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, args=(0.05,), daemon=True).start()

    def sent(self, kind):
        return [body for name, body in self.requests if name == kind]


@pytest.fixture
def backends():
    backends = Backends()
    yield backends
    backends.server.shutdown()


def make_replica(backends, **overrides):
    replica = object.__new__(cast(Any, serve_chat.Chat).func_or_class)
    replica.config = {
        "stt_url": f"{backends.url}/v1",
        "llm_url": f"{backends.url}/llm",
        "llm_model": "tiny-llm",
        "llm_max_tokens": 64,
        "tts_url": f"{backends.url}/tts",
        "voice": "bob",
        "system_prompt": "Be brief.",
        "tokens": {"STT": "", "LLM": "", "TTS": ""},
        **overrides,
    }
    replica.languages = None
    return replica


@pytest.fixture
def replica(backends):
    return make_replica(backends)


@pytest.fixture
def client(replica):
    yield serve_client(serve_chat, replica)
    serve_chat.api.dependency_overrides.clear()


def chat(client, **body):
    response = client.post("/api/chat", json=body)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines()]


def types(events):
    return [e["type"] for e in events]


WAV = base64.b64encode(b"RIFF....WAVEfmt ").decode()


def test_spoken_strips_markdown_links_and_emoji():
    text = "## Title\n- **Bold** [link](https://x.y) and https://a.b/c 😀 done"
    assert serve_chat.spoken(text) == "Title Bold link and done"


def test_chunker_cuts_sentences_as_text_streams():
    chunker = serve_chat.Chunker()
    assert chunker.feed("Pi is 3.14, about") == []  # "3.14" is not a stop, and the comma comes too early
    assert chunker.feed(" that. Next") == ["Pi is 3.14, about that."]
    assert chunker.feed("?!") == []  # the next token may continue the run of punctuation
    assert chunker.feed(" And", final=False) == ["Next?!"]
    assert chunker.feed(" done", final=True) == ["And done"]


def test_chunker_first_chunk_may_end_at_a_comma():
    chunker = serve_chat.Chunker()
    assert chunker.feed("Well, that is a long start, and more") == ["Well, that is a long start,"]
    assert chunker.feed(", still going", final=True) == ["and more, still going"]  # later commas don't cut


def test_chunker_cjk_stops():
    assert serve_chat.Chunker().feed("你好。今天怎么样？好", final=True) == ["你好。", "今天怎么样？", "好"]  # noqa: RUF001


@pytest.mark.parametrize(
    ("text", "detected", "language"),
    [
        ("こんにちは", None, "japanese"),
        ("안녕", "english", "korean"),
        ("你好", None, "chinese"),
        ("Hi", "french", "french"),
    ],
)
def test_reply_language(text, detected, language):
    assert serve_chat.reply_language(text, detected) == language
    assert serve_chat.reply_language("Hi") is None


def test_system_prompt_names_the_reply_language():
    assert serve_chat.system_prompt("P", "chinese").endswith("is in Chinese; reply in Chinese.")
    assert "whatever language earlier turns used" in serve_chat.system_prompt("P", None)


def test_history_keeps_valid_recent_turns():
    messages: list[Any] = [{"role": "user", "content": f"m{i}"} for i in range(25)]
    messages += [{"role": "system", "content": "x"}, {"role": "assistant", "content": ""}, "junk"]
    past = serve_chat.history(messages)
    assert len(past) == serve_chat.HISTORY_MESSAGES
    assert past[-1] == {"role": "user", "content": "m24"}
    with pytest.raises(ValueError):
        serve_chat.history("nope")


def test_settings(monkeypatch):
    for name, value in {
        "CHAT_STT_URL": "http://stt/v1/",
        "CHAT_LLM_URL": "http://llm/v1",
        "CHAT_LLM_MODEL": "big",
        "CHAT_LLM_MAX_TOKENS": "100",
        "CHAT_TTS_URL": "http://tts/",
        "CHAT_VOICE": "alice",
        "CHAT_SYSTEM_PROMPT": "Talk.",
        "CHAT_LLM_TOKEN": "secret",
    }.items():
        monkeypatch.setenv(name, value)
    assert serve_chat.settings() == {
        "stt_url": "http://stt/v1",
        "llm_url": "http://llm/v1",
        "llm_model": "big",
        "llm_max_tokens": 100,
        "tts_url": "http://tts",
        "voice": "alice",
        "system_prompt": "Talk.",
        "tokens": {"STT": "", "LLM": "secret", "TTS": ""},
    }


def test_page_uses_relative_urls(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "fetch('api/chat'" in response.text and "fetch('api/voices')" in response.text
    assert "fetch('/" not in response.text  # absolute URLs would miss the route prefix


def test_health(client, backends):
    assert client.get("/health").json() == {
        "status": "ok",
        "stt": f"{backends.url}/v1",
        "llm": f"{backends.url}/llm",
        "llm_model": "tiny-llm",
        "tts": f"{backends.url}/tts",
        "voice": "bob",
    }


def test_voices(client):
    assert client.get("/api/voices").json() == {"voices": ["alice", "bob"], "default": "bob"}


def test_voices_default_to_the_first_when_unset(backends):
    client = serve_client(serve_chat, make_replica(backends, voice=""))
    try:
        assert client.get("/api/voices").json() == {"voices": ["alice", "bob"], "default": "alice"}
    finally:
        serve_chat.api.dependency_overrides.clear()


def test_voices_when_tts_is_down(backends):
    client = serve_client(serve_chat, make_replica(backends, tts_url="http://127.0.0.1:9"))
    try:
        response = client.get("/api/voices")
    finally:
        serve_chat.api.dependency_overrides.clear()
    assert response.status_code == 502
    assert response.json()["error"].startswith("TTS unavailable")


def test_tokens_are_sent_to_their_backend(backends):
    tokens = {"STT": "", "LLM": "", "TTS": "tts-pat"}
    client = serve_client(serve_chat, make_replica(backends, tokens=tokens))
    try:
        client.get("/api/voices")
    finally:
        serve_chat.api.dependency_overrides.clear()
    assert ("GET /tts/voices", "Bearer tts-pat") in backends.requests


def test_unknown_route_is_json_404(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json() == {"error": "Not found"}


def test_text_turn(client, backends):
    events = chat(client, text="Hi", history=[{"role": "assistant", "content": "Earlier."}])
    # Text and audio interleave as the threads run (the first delta is already a sentence); each keeps its own order.
    assert [e["delta"] for e in events if e["type"] == "text"] == backends.reply
    audio = [e for e in events if e["type"] == "audio"]
    assert [(e["index"], e["part"]) for e in audio] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [e["text"] for e in audio] == ["Hello there.", None, "How are you today?", None]
    assert audio[0]["sample_rate"] == 24000 and audio[0]["seconds"] == 0.1
    assert "first_audio" in audio[0] and "first_audio" not in audio[1]
    done = events[-1]
    assert done["type"] == "done" and done["reply"] == "Hello there. How are you today?"
    assert {"llm_first_token", "llm_done", "first_audio"} <= done["timings"].keys()

    (llm,) = backends.sent("llm")
    assert llm["model"] == "tiny-llm" and llm["max_tokens"] == 64 and llm["stream"] is True
    assert llm["chat_template_kwargs"] == {"enable_thinking": False}
    assert llm["messages"][0]["content"].startswith("Be brief.\nReply in the language of the user's latest message")
    assert llm["messages"][1:] == [{"role": "assistant", "content": "Earlier."}, {"role": "user", "content": "Hi"}]
    assert [(t["text"], t["voice"], t["language"], t["stream"]) for t in backends.sent("tts")] == [
        ("Hello there.", "bob", "auto", True),
        ("How are you today?", "bob", "auto", True),
    ]
    assert backends.sent("stt") == []


def test_voice_and_language_follow_the_request(client, backends):
    backends.reply = ["你好。"]
    chat(client, text="你好吗", voice="alice")
    (llm,) = backends.sent("llm")
    assert llm["messages"][0]["content"].endswith("reply in Chinese.")
    assert [(t["voice"], t["language"]) for t in backends.sent("tts")] == [("alice", "chinese")]


def test_audio_turn(client, backends):
    events = chat(client, audio=WAV)
    assert events[0] == {"type": "transcript", "text": "Hello.", "language": "english", "seconds": events[0]["seconds"]}
    assert events[-1]["type"] == "done" and "stt" in events[-1]["timings"]
    (stt,) = backends.sent("stt")
    assert b'name="response_format"\r\n\r\nverbose_json' in stt
    assert base64.b64decode(WAV) in stt
    (llm,) = backends.sent("llm")
    assert llm["messages"][-1] == {"role": "user", "content": "Hello."}
    assert llm["messages"][0]["content"].endswith("reply in English.")
    assert {t["language"] for t in backends.sent("tts")} == {"english"}


def test_empty_transcript_ends_the_turn(client, backends):
    backends.transcript = {"text": "  ", "language": "english"}
    events = chat(client, audio=WAV)
    assert types(events) == ["transcript", "done"]
    assert events[-1]["reply"] == ""
    assert backends.sent("llm") == []


def test_stt_error(client, backends):
    backends.stt_status = 500
    assert chat(client, audio=WAV) == [{"type": "error", "stage": "stt", "message": "STT returned 500: STT broke"}]


def test_llm_error_still_finishes(client, backends):
    backends.llm_status = 400
    events = chat(client, text="Hi")
    assert types(events) == ["error", "done"]
    assert events[0] == {"type": "error", "stage": "llm", "message": "LLM returned 400: bad model"}
    assert len(backends.sent("llm")) == 1  # 4xx is not retried
    assert backends.sent("tts") == []


def test_busy_tts_is_retried(client, backends):
    backends.tts_busy = 2
    events = chat(client, text="Hi")
    assert "error" not in types(events)
    assert len([e for e in events if e["type"] == "audio"]) == 4
    assert len(backends.sent("tts")) == 4  # two 503s, then one request per chunk


def test_wav_tts_backend(client, backends):
    backends.tts_wav = True
    events = [e for e in chat(client, text="Hi") if e["type"] == "audio"]
    assert events[0]["wav"] == base64.b64encode(b"RIFF-wav").decode()
    assert (events[0]["seconds"], events[0]["rtf"], events[0]["text"]) == (1.25, 0.5, "Hello there.")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"", "Request must be between"),
        (b"[", "Expecting value"),
        (b'{"text": "  "}', "Send audio or text"),
        (b'{"audio": "not base64!"}', "base64"),
        (b'{"text": "Hi", "history": "x"}', "history must be a list"),
        (b'{"text": "Hi", "voice": 3}', "Select a voice"),
        (b'["Hi"]', "attribute"),
    ],
)
def test_invalid_requests(client, body, message):
    response = client.post("/api/chat", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert message in response.json()["error"]


def test_turn_needs_a_voice(backends):
    client = serve_client(serve_chat, make_replica(backends, voice=""))
    try:
        response = client.post("/api/chat", json={"text": "Hi"})
    finally:
        serve_chat.api.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json() == {"error": "Select a voice"}


def test_disconnect_cancels_backend_requests(backends, replica):
    """A client that goes away mid-reply cancels the turn, which shuts its in-flight TTS connection."""
    backends.tts_hold = threading.Event()

    async def main():
        turn = serve_chat.Turn(asyncio.get_running_loop(), replica.config["tokens"])
        events, first = [], asyncio.Event()

        async def consume():
            async for item in replica.converse(turn, time.perf_counter(), None, "Hi", "bob", []):
                events.append(json.loads(item))
                if events[-1]["type"] == "audio":
                    first.set()

        consumer = asyncio.ensure_future(consume())
        await asyncio.wait_for(first.wait(), 5)
        consumer.cancel()  # what Serve does to the response when the client disconnects
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert turn.cancelled.is_set()
        assert await asyncio.to_thread(backends.tts_hold.wait, 5)  # TTS saw its client hang up
        assert "done" not in types(events)

    asyncio.run(main())


def test_busy_tts_voices_is_502(client, backends):
    backends.discovery = [503]
    response = client.get("/api/voices")
    assert response.status_code == 502
    assert response.json() == {"error": 'TTS unavailable: TTS returned 503: {"error": "busy"}'}


def test_tts_languages_are_retried_until_tts_answers(client, backends):
    backends.discovery = [503]
    chat(client, text="Bonjour")  # /health busy: the turn falls back to auto
    backends.reply = ["Bonjour."]
    chat(client, text="你好")  # asked again, now answered
    assert [t["language"] for t in backends.sent("tts")] == ["auto", "auto", "chinese"]
    assert [name for name, _ in backends.requests if name == "GET /tts/health"] == ["GET /tts/health"] * 2


def test_oversized_request_is_refused_by_content_length(client, monkeypatch):
    monkeypatch.setattr(serve_chat, "MAX_REQUEST_BYTES", 100)
    response = client.post("/api/chat", content=b"x" * 101, headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json() == {"error": "Request must be between 1 and 100 bytes"}


def test_oversized_chunked_request_is_refused_while_reading(client, monkeypatch):
    monkeypatch.setattr(serve_chat, "MAX_REQUEST_BYTES", 100)

    def chunks():  # no Content-Length: httpx sends it chunked
        for _ in range(50):
            yield b"x" * 10

    response = client.post("/api/chat", content=chunks(), headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json() == {"error": "Request must be between 1 and 100 bytes"}


def body_request(chunks, headers=()):
    """A Starlette request whose body arrives as `chunks`, recording how many of them were received."""
    from starlette.requests import Request

    received = []

    async def receive():
        received.append(1)
        more = len(received) < len(chunks)
        return {"type": "http.request", "body": chunks[len(received) - 1], "more_body": more}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/chat",
        "headers": [(k.encode(), v.encode()) for k, v in headers],
    }
    return Request(scope, receive), received


def test_read_body_stops_reading_past_the_limit():
    request, received = body_request([b"x" * 10] * 50)
    with pytest.raises(ValueError, match="between 1 and 100 bytes"):
        asyncio.run(serve_chat.read_body(request, 100))
    assert len(received) == 11  # stopped at the chunk that crossed the limit, not after all 50


def test_read_body_refuses_a_declared_oversize_without_reading():
    request, received = body_request([b"x" * 10], headers=[("content-length", "101")])
    with pytest.raises(ValueError, match="between 1 and 100 bytes"):
        asyncio.run(serve_chat.read_body(request, 100))
    assert received == []
