"""caller_log: one log line naming the caller for requests that carry the proxy's or authorizer's identity headers."""

import logging

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import serve_chat
import serve_meeting
import serve_stt
import serve_tts
from caller_log import CallerLog, clean


@pytest.fixture
def client():
    api = FastAPI()
    api.add_middleware(CallerLog)

    @api.get("/ok")
    def ok():
        return {"ok": True}

    @api.get("/boom")
    def boom():
        raise RuntimeError("boom")

    @api.websocket("/ws")
    async def echo(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text(await websocket.receive_text())
        await websocket.close()

    @api.websocket("/refuse")
    async def refuse(websocket: WebSocket):
        await websocket.close(code=1008)

    return TestClient(api, raise_server_exceptions=False)


def lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "ray.serve"]


@pytest.mark.parametrize("module", [serve_tts, serve_stt, serve_chat, serve_meeting])
def test_every_app_logs_callers(module):
    assert any(m.cls is CallerLog for m in module.api.user_middleware)


def test_http_request_with_identity(client, caplog):
    caplog.set_level(logging.INFO, "ray.serve")
    headers = {"X-Consumer": "open-webui", "X-Auth-Request-Email": "a@example.com", "Authorization": "Bearer secret"}
    assert client.get("/ok", headers=headers).status_code == 200
    [line] = lines(caplog)
    assert line.startswith("GET /ok 200 ")
    assert line.endswith(" consumer=open-webui email=a@example.com")
    assert "secret" not in caplog.text


def test_request_without_identity_is_not_logged(client, caplog):
    caplog.set_level(logging.INFO, "ray.serve")
    assert client.get("/ok", headers={"Authorization": "Bearer secret"}).status_code == 200
    assert lines(caplog) == []


def test_single_header_and_error_status(client, caplog):
    caplog.set_level(logging.INFO, "ray.serve")
    assert client.get("/missing", headers={"X-Auth-Request-Email": "a@example.com"}).status_code == 404
    assert client.get("/boom", headers={"X-Consumer": "svc"}).status_code == 500
    first, second = lines(caplog)
    assert first.startswith("GET /missing 404 ") and first.endswith(" email=a@example.com")
    assert second.startswith("GET /boom ") and second.endswith(" consumer=svc")
    assert " failed " in second  # Starlette turns the exception into a 500 outside user middleware


def test_websocket_with_identity(client, caplog):
    caplog.set_level(logging.INFO, "ray.serve")
    with client.websocket_connect("/ws", headers={"X-Consumer": "svc"}) as ws:
        ws.send_text("hi")
        assert ws.receive_text() == "hi"
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/refuse", headers={"X-Consumer": "svc"}):
        pass
    accepted, refused = lines(caplog)
    assert accepted.startswith("WS /ws accepted ") and accepted.endswith(" consumer=svc")
    assert refused.startswith("WS /refuse refused ")


def test_reverse_proxy_email(client, caplog):
    """OAuth2 Proxy in front of an app (the TTS page) forwards the user as X-Forwarded-Email."""
    caplog.set_level(logging.INFO, "ray.serve")
    assert client.get("/ok", headers={"X-Forwarded-Email": "page@example.com"}).status_code == 200
    both = {"X-Forwarded-Email": "page@example.com", "X-Auth-Request-Email": "auth@example.com"}
    assert client.get("/ok", headers=both).status_code == 200
    page, preferred = lines(caplog)
    assert page.startswith("GET /ok 200 ") and page.endswith(" email=page@example.com")
    assert preferred.endswith(" email=auth@example.com")


def test_values_stay_on_one_line():
    assert clean(b"a b\tc\x7fd") == "a?b?c?d"
    assert clean("é".encode()) == "??"
    assert len(clean(b"x" * 1000)) == 200
