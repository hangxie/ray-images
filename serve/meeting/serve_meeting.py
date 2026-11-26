"""Ray Serve meeting notes: the page that transcribes a meeting live, relaying its audio to serve_stt's stream.

`GET /` (meeting.html), `GET /health`, and a WebSocket at `/api/stream` that opens STT's `audio/stream` with a bearer
token and relays messages both ways unchanged, so the events are STT's. It holds no models, so replicas need a CPU and
the system Python only, like serve_chat; a page change redeploys this app instead of reloading Whisper. Access control
belongs to the deployment in front, as for the other apps.
"""

import asyncio
import html
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import cast
from urllib.parse import urlencode, urlsplit, urlunsplit

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse, Response
from ray import serve
from starlette.exceptions import HTTPException
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

ROOT = Path(__file__).resolve().parent
VERSION_FILE = ROOT.parent / "VERSION"  # written by the image build, shared by all the apps
VERSION = VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else "dev"
BACKEND_TIMEOUT = 10  # for opening STT's stream; the stream itself has no deadline
# For STT's /health, which the page waits on while STT hasn't answered yet (e.g. while Whisper loads): the page then
# loads without the language list instead of hanging.
HEALTH_TIMEOUT = 2
MAX_EVENT_BYTES = 1024 * 1024  # STT's events are small JSON objects

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
    """Replica settings, read on the head and passed through bind(), as in serve_chat."""
    env = os.environ.get
    return {
        "stt_url": env("MEETING_STT_URL", "http://localhost:8000/v1").rstrip("/"),
        "stt_token": env("MEETING_STT_TOKEN", ""),  # sent as `Authorization: Bearer`; empty sends none
    }


def auth(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def stream_url(stt_url, language):
    """STT's `audio/stream` as a ws:// or wss:// URL, with the page's language."""
    parts = urlsplit(f"{stt_url}/audio/stream")
    scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode({"language": language}) if language else "", ""))


def close_code(code):
    """A close code to pass on to the browser; codes that can't be sent (1005, 1006, 1015) become 1011."""
    return (
        code if code in (1000, 1001, 1002, 1003, 1007, 1008, 1009, 1010, 1011) or 3000 <= (code or 0) < 5000 else 1011
    )


@serve.deployment(
    num_replicas=int(os.environ.get("MEETING_NUM_REPLICAS", "1")),
    # A meeting is one WebSocket held for as long as it lasts and mostly waits on STT, so a replica holds many.
    max_ongoing_requests=int(os.environ.get("MEETING_MAX_ONGOING_REQUESTS", "16")),
)
@serve.ingress(api)
class Meeting:
    def __init__(self, settings):
        self.config = settings
        self.languages = None  # Whisper's, name -> code, from STT's /health on first use

    def stt_health(self):
        request = urllib.request.Request(f"{self.config['stt_url']}/health", headers=auth(self.config["stt_token"]))
        with urllib.request.urlopen(request, timeout=HEALTH_TIMEOUT) as response:
            return json.loads(response.read())

    def stt_languages(self):
        """Whisper's languages for the page's menu, read from STT once it answers; until then each page asks again."""
        if self.languages is None:
            try:
                languages = self.stt_health().get("languages")
                if not isinstance(languages, dict):
                    raise ValueError("STT /health has no languages")
                self.languages = languages
            except (OSError, ValueError, AttributeError) as exc:
                logger.warning("STT languages unavailable: %s", exc)
                return {}
        return self.languages

    @api.get("/")
    async def page(self):
        languages = await asyncio.to_thread(self.stt_languages)
        page = (ROOT / "meeting.html").read_text().replace("{{VERSION}}", html.escape(VERSION))
        page = page.replace("{{LANGUAGES}}", json.dumps(languages, sort_keys=True).replace("</", "<\\/"))
        return reply(200, page.encode(), "text/html; charset=utf-8")

    @api.get("/health")
    def health(self):
        return reply(200, {"status": "ok", "version": VERSION, "stt": self.config["stt_url"]})

    @api.websocket("/api/stream")
    async def stream(self, websocket: WebSocket):
        """STT's `audio/stream`, relayed: the page's messages go to STT and STT's events come back as they are, and
        when either side closes the other is closed too. STT's close code and reason reach the page."""
        await websocket.accept()
        url = stream_url(self.config["stt_url"], websocket.query_params.get("language", ""))
        try:
            stt = await connect(
                url,
                additional_headers=auth(self.config["stt_token"]),
                open_timeout=BACKEND_TIMEOUT,
                max_size=MAX_EVENT_BYTES,
            )
        except (OSError, TimeoutError, InvalidHandshake) as exc:
            logger.warning("STT stream unavailable: %s", exc)
            await websocket.send_json({"type": "error", "message": f"STT unavailable: {exc}"})
            await websocket.close(1011, "STT unavailable")
            return

        async def upload():
            """The page's audio and control messages, until the page goes away."""
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    await stt.send(message["bytes"])
                elif message.get("text") is not None:
                    await stt.send(message["text"])

        async def download():
            """STT's events, until STT closes the stream."""
            try:
                async for event in stt:
                    if isinstance(event, bytes):
                        await websocket.send_bytes(event)
                    else:
                        await websocket.send_text(event)
            except ConnectionClosed:
                pass

        tasks = [asyncio.create_task(upload()), asyncio.create_task(download())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            page_left = tasks[0].done() and tasks[0].exception() is None
            if not page_left:  # STT ended the stream, or went away while the page was sending; tell the page how
                reason = stt.close_reason or ("" if stt.close_code == 1000 else "STT connection lost")
                await websocket.close(close_code(stt.close_code), reason)
        except Exception:
            logger.exception("Meeting relay failed")
        finally:
            for task in tasks:
                task.cancel()
            try:
                await stt.close()
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                stt.transport.abort()  # in case the close was cancelled, e.g. as the page went away; else a no-op


# mypy ignores class decorators (python/mypy#3135), so it sees Meeting as the plain class rather than a Deployment
app = cast(serve.Deployment, Meeting).bind(settings())
