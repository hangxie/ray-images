"""Log who made each request, as the deployment's proxy or authorizer reports it.

The image has no auth (AGENTS.md, No auth in the image); the homelab Gateway's bearer check sets `X-Consumer` and
OAuth2 Proxy sets `X-Auth-Request-Email`. Requests carrying either get one log line with the caller, the request and
its outcome; Serve's own access log already covers the rest. `Authorization` is never read.

Stdlib only, since the head imports it with the system Python.
"""

import logging
import time

logger = logging.getLogger("ray.serve")

# Header, as ASGI gives it (lower case), and the name it is logged under.
IDENTITY_HEADERS = ((b"x-consumer", "consumer"), (b"x-auth-request-email", "email"))
MAX_VALUE_LENGTH = 200


def clean(value):
    """A header value made safe for one log line: printable ASCII without spaces, at most MAX_VALUE_LENGTH long."""
    text = value.decode("latin-1")[:MAX_VALUE_LENGTH]
    return "".join(c if "!" <= c <= "~" else "?" for c in text)


def identity(scope):
    """The caller's identity fields, in IDENTITY_HEADERS order, from the request's headers."""
    found = {}
    for name, value in scope.get("headers", ()):
        for header, field in IDENTITY_HEADERS:
            if name.lower() == header and field not in found:
                found[field] = clean(value)
    return [(field, found[field]) for _, field in IDENTITY_HEADERS if field in found]


class CallerLog:
    """ASGI middleware: log the caller of each HTTP request and WebSocket that carries an identity header."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        caller = identity(scope) if scope["type"] in ("http", "websocket") else []
        if not caller:
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        outcome = "-"

        async def watch(message):
            nonlocal outcome
            if message["type"] == "http.response.start":
                outcome = str(message["status"])
            elif message["type"] == "websocket.accept":
                outcome = "accepted"
            elif message["type"] == "websocket.close" and outcome == "-":
                outcome = "refused"
            await send(message)

        try:
            await self.app(scope, receive, watch)
        except BaseException:
            outcome = "failed" if outcome == "-" else outcome
            raise
        finally:
            method = scope.get("method", "WS")
            fields = " ".join(f"{field}={value}" for field, value in caller)
            elapsed = time.perf_counter() - started
            logger.info("%s %s %s %.2fs %s", method, scope["path"], outcome, elapsed, fields)
