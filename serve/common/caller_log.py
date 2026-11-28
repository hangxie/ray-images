"""Log who made each request, from the identity headers the deployment's proxy sets.

`X-Consumer` (Gateway bearer check), `X-Auth-Request-Email` or `X-Forwarded-Email` (OAuth2 Proxy). `Authorization` is
never read. Stdlib only: the head imports it.
"""

import logging
import time

logger = logging.getLogger("ray.serve")

# ASGI (lower-case) header -> logged name; for a shared name, the first header present wins.
IDENTITY_HEADERS = (
    (b"x-consumer", "consumer"),
    (b"x-auth-request-email", "email"),
    (b"x-forwarded-email", "email"),
)
MAX_VALUE_LENGTH = 200


def clean(value):
    """A header value made safe for one log line: printable ASCII without spaces, at most MAX_VALUE_LENGTH long."""
    text = value.decode("latin-1")[:MAX_VALUE_LENGTH]
    return "".join(c if "!" <= c <= "~" else "?" for c in text)


def identity(scope):
    """The caller's identity fields, in IDENTITY_HEADERS order, from the request's headers."""
    headers: dict[bytes, bytes] = {}
    for name, value in scope.get("headers", ()):
        headers.setdefault(name.lower(), value)
    found: dict[str, str] = {}
    for header, field in IDENTITY_HEADERS:
        if header in headers and field not in found:
            found[field] = clean(headers[header])
    return list(found.items())


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
