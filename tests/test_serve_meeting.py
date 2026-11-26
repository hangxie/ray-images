"""serve_meeting's page and stream relay, against serve_stt's real routes (with a fake Whisper) served on loopback.

Everything but Whisper is real: the meeting routes, the websockets client, uvicorn, and STT's stream and segmenter.
"""

import json
import re
import socket
import threading
import time
from typing import Any, cast

import numpy as np
import pytest
import uvicorn
from conftest import serve_client
from starlette.websockets import WebSocketDisconnect
from test_serve_stt import pcm, replica, silence, speech  # noqa: F401  # replica is a fixture
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

import serve_meeting
import serve_stt


class STTServer:
    """serve_stt's app under /v1 on a loopback port, recording each request's path, query and Authorization, and
    which WebSocket handlers have returned."""

    def __init__(self, whisper):
        self.requests: list[tuple[str, str, str]] = []
        self.finished: list[str] = []
        serve_client(serve_stt, whisper)  # binds the routes to the STT replica

        async def app(scope, receive, send):
            if scope["type"] in ("http", "websocket"):
                headers = dict(scope["headers"])
                path = scope["path"].removeprefix("/v1")
                self.requests.append(
                    (path, scope["query_string"].decode(), headers.get(b"authorization", b"").decode())
                )
                scope = {**scope, "path": path, "raw_path": path.encode()}
            await serve_stt.api(scope, receive, send)
            if scope["type"] == "websocket":
                self.finished.append(scope["path"])

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}/v1"
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [sock]}, daemon=True)
        self.thread.start()
        while not self.server.started:
            time.sleep(0.01)

    def stop(self):
        self.server.should_exit = True
        self.thread.join(10)
        serve_stt.api.dependency_overrides.clear()


@pytest.fixture
def stt(replica):  # noqa: F811
    server = STTServer(replica)
    yield server
    server.stop()


def make_replica(stt_url, token=""):
    meeting = object.__new__(cast(Any, serve_meeting.Meeting).func_or_class)
    meeting.config = {"stt_url": stt_url, "stt_token": token}
    meeting.languages = None
    return meeting


@pytest.fixture
def client(stt):
    yield serve_client(serve_meeting, make_replica(stt.url, "stt-pat"))
    serve_meeting.api.dependency_overrides.clear()


def page_languages(page):
    found = re.search(r"const LANGUAGE_CODES = (\{.*?\});", page)
    assert found
    return json.loads(found.group(1))


def test_settings(monkeypatch):
    monkeypatch.setenv("MEETING_STT_URL", "http://stt:8000/v1/")
    monkeypatch.setenv("MEETING_STT_TOKEN", "secret")
    assert serve_meeting.settings() == {"stt_url": "http://stt:8000/v1", "stt_token": "secret"}
    monkeypatch.delenv("MEETING_STT_URL")
    monkeypatch.delenv("MEETING_STT_TOKEN")
    assert serve_meeting.settings() == {"stt_url": "http://localhost:8000/v1", "stt_token": ""}


@pytest.mark.parametrize(
    ("stt_url", "language", "url"),
    [
        ("http://stt:8000/v1", "", "ws://stt:8000/v1/audio/stream"),
        ("https://stt.example/v1", "chinese", "wss://stt.example/v1/audio/stream?language=chinese"),
        ("http://stt/v1", "a b&c", "ws://stt/v1/audio/stream?language=a+b%26c"),
    ],
)
def test_stream_url(stt_url, language, url):
    assert serve_meeting.stream_url(stt_url, language) == url


@pytest.mark.parametrize(("code", "sent"), [(1000, 1000), (1008, 1008), (4000, 4000), (1006, 1011), (None, 1011)])
def test_close_code(code, sent):
    assert serve_meeting.close_code(code) == sent


def test_page_uses_relative_urls(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["cache-control"] == "no-store"
    assert "new URL('api/stream', location.href)" in response.text
    assert "fetch(" not in response.text  # the page uses only its stream
    assert f'<span id="version">{serve_meeting.VERSION}</span>' in response.text


def test_page_language_menu_has_stt_languages(client, stt):
    codes = page_languages(client.get("/").text)
    assert codes == TO_LANGUAGE_CODE and codes["mandarin"] == "zh"
    presets = re.findall(r'<option value="([a-z]+)">', client.get("/").text)
    assert presets == ["english", "chinese", "spanish", "french", "korean", "japanese", "other"]
    assert all(p in codes for p in presets[:-1])  # each preset is a name STT accepts
    assert [r for r in stt.requests if r[0] == "/health"] == [("/health", "", "Bearer stt-pat")]  # read once


def test_page_loads_while_stt_is_down():
    meeting = make_replica("http://127.0.0.1:9/v1")
    client = serve_client(serve_meeting, meeting)
    try:
        response = client.get("/")
    finally:
        serve_meeting.api.dependency_overrides.clear()
    assert response.status_code == 200
    assert page_languages(response.text) == {}
    assert meeting.languages is None  # asked again on the next page


def test_health(client, stt):
    assert client.get("/health").json() == {"status": "ok", "version": serve_meeting.VERSION, "stt": stt.url}


def test_unknown_route_is_json_404(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json() == {"error": "Not found"}


def test_stream_is_relayed_to_stt(client, stt, replica):  # noqa: F811
    audio = np.concatenate([speech(5.5), silence(3.0), speech(1.0)])
    events: list[dict] = []
    with client.websocket_connect("/api/stream?language=french") as ws:
        for i in range(0, len(audio), 1600):
            ws.send_bytes(pcm(audio[i : i + 1600]))
        ws.send_text(json.dumps({"type": "stop"}))
        while not events or events[-1]["type"] != "done":
            events.append(ws.receive_json())
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 1000
    assert [e["type"] for e in events] == ["segment", "segment", "done"]
    assert [(e["text"], e["language"], e["paragraph"]) for e in events[:2]] == [
        ("Hello world.", "french", True),
        ("Hello world.", "french", True),
    ]
    assert ("/audio/stream", "language=french", "Bearer stt-pat") in stt.requests
    assert sum(a.size for a, _ in replica.calls) > 6 * 16000  # the audio reached Whisper


def test_stt_errors_and_close_reach_the_page(client):
    with client.websocket_connect("/api/stream?language=klingon") as ws:
        assert ws.receive_json() == {"type": "error", "message": "Unknown language: klingon"}
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 1008


def test_stt_down_closes_the_stream():
    client = serve_client(serve_meeting, make_replica("http://127.0.0.1:9/v1"))
    try:
        with client.websocket_connect("/api/stream") as ws:
            event = ws.receive_json()
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
    finally:
        serve_meeting.api.dependency_overrides.clear()
    assert event["type"] == "error" and event["message"].startswith("STT unavailable")
    assert (closed.value.code, closed.value.reason) == (1011, "STT unavailable")


def test_page_leaving_closes_the_stt_stream(client, stt):
    with client.websocket_connect("/api/stream") as ws:
        ws.send_bytes(pcm(speech(1.0)))
    for _ in range(500):
        if stt.finished:
            break
        time.sleep(0.01)
    assert stt.finished == ["/audio/stream"]
