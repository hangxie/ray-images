"""Shared client for txt2mp3.py and txt2m4b.py: spreads paragraphs over every configured TTS endpoint.

Each endpoint is a serve_tts app, called through its OpenAI-style API (`GET /v1/health`, `POST /v1/audio/speech`); its
URL is the app's base, with or without `/v1`. Endpoints come from a TOML file (--endpoints, TTS_ENDPOINTS, or
~/.config/mlx-tts/endpoints.toml):

  [[endpoint]]
  name      = "k8s"
  url       = "https://tts.homelab.xiehang.com"
  token_env = "TTS_TOKEN_K8S"     # or token_file = "...", or token = "..."
  slots     = 2

Each endpoint runs one worker per slot, all taking paragraphs from one queue in input order, so faster endpoints
simply take more of them. Every endpoint is assumed to serve the same model and voices.
"""

import collections
import http.client
import json
import os
import re
import stat
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request

DEFAULT_CONFIG = os.path.expanduser("~/.config/mlx-tts/endpoints.toml")
# Fixed generation settings, sent with every paragraph and recorded in the output tags.
SETTINGS = {
    "seed": 0,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.5,
    "max_tokens": 4096,
    "volume": -20.0,
}
REQUEST_TIMEOUT = 900
PROBE_TIMEOUT = 10
BUSY_SECONDS = 5
# Backoff for an endpoint that stopped answering, doubling from the first to the last value.
BACKOFF = (5, 300)
# The run fails once every endpoint has been failing this long, or once one paragraph has failed this many times and
# on every endpoint still in the run, so one broken endpoint cannot fail a paragraph that the others can synthesize.
STALL_SECONDS = 900
MAX_ATTEMPTS = 4


def paragraphs(path):
    """Each non-empty line is a paragraph; full-width spaces, used to indent Chinese text, count as whitespace."""
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            text = " ".join(line.replace("　", " ").split())
            if text:
                yield text


def add_arguments(ap):
    env = os.environ.get
    ap.add_argument(
        "--endpoints",
        default=env("TTS_ENDPOINTS"),
        help=f"endpoint config file (default: {DEFAULT_CONFIG} when it exists and --tts-url is unset)",
    )
    ap.add_argument(
        "--tts-url",
        default=env("TTS_URL"),
        help="single endpoint instead of a config; a TTS_TOKEN environment variable is sent as a bearer token",
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=env("PARALLEL"),
        help="paragraphs sent at once to --tts-url (default: 1); a config sets slots per endpoint",
    )


class Endpoint:
    def __init__(self, name, url, token=None, slots=1):
        self.name, self.slots, self.token = name, slots, token
        self.base = url.rstrip("/").removesuffix("/v1/audio/speech").removesuffix("/v1")
        self.health = None
        self.dropped = None  # reason the endpoint was taken out for the rest of the run
        self.failing_since = None  # start of the current run of failures
        self.retry_at = 0.0
        self.backoff = BACKOFF[0]
        self.done, self.audio, self.synthesis = 0, 0.0, 0.0
        self.busy = False

    def headers(self):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


def load_endpoints(args, error):
    """Return endpoints from the config file, or the single --tts-url endpoint."""
    path = args.endpoints
    if path is None and args.tts_url is None and os.path.exists(DEFAULT_CONFIG):
        path = DEFAULT_CONFIG
    if path is None:
        if args.tts_url is None:
            error(f"set --tts-url (or TTS_URL), or list endpoints in {DEFAULT_CONFIG}")
        if args.parallel is not None and int(args.parallel) < 1:
            error("--parallel must be at least 1")
        return [Endpoint("tts", args.tts_url, os.environ.get("TTS_TOKEN"), int(args.parallel or 1))]
    if args.parallel is not None:
        error(f"--parallel and PARALLEL apply only to --tts-url; set slots per endpoint in {path}")
    try:
        with open(path, "rb") as f:
            config = tomllib.load(f)
        mode = os.stat(path).st_mode
    except (OSError, tomllib.TOMLDecodeError) as exc:
        error(f"cannot read {path}: {exc}")
    endpoints: list[Endpoint] = []
    for i, entry in enumerate(config.get("endpoint", []), 1):
        where = f"{path}: endpoint {i}"
        unknown = set(entry) - {"name", "url", "slots", "token", "token_env", "token_file"}
        if unknown:
            error(f"{where}: unknown keys {', '.join(sorted(unknown))}")
        if not isinstance(entry.get("url"), str):
            error(f"{where}: url is required")
        slots = entry.get("slots", 1)
        if not isinstance(slots, int) or slots < 0:
            error(f"{where}: slots must be a non-negative integer")
        if sum(key in entry for key in ("token", "token_env", "token_file")) > 1:
            error(f"{where}: set only one of token, token_env and token_file")
        token = entry.get("token")
        if token and mode & (stat.S_IRWXG | stat.S_IRWXO):
            print(f"warning: {path} holds a token but is readable by others; chmod 600 it", file=sys.stderr)
        if "token_env" in entry:
            token = os.environ.get(entry["token_env"])
            if not token:
                error(f"{where}: environment variable {entry['token_env']} is not set")
        if "token_file" in entry:
            try:
                with open(os.path.expanduser(entry["token_file"])) as f:
                    token = f.read().strip()
            except OSError as exc:
                error(f"{where}: {exc}")
        name = entry.get("name", entry["url"])
        if any(e.name == name for e in endpoints):
            error(f"{where}: name {name} is used twice")
        # slots = 0 keeps an endpoint in the file while leaving it out of runs.
        if slots:
            endpoints.append(Endpoint(name, entry["url"], token, slots))
    if not endpoints:
        error(f"{path} has no endpoint with slots")
    return endpoints


class NoRedirect(urllib.request.HTTPRedirectHandler):
    # A missing token makes the homelab gateway redirect to its SSO login; following it would return a 200 page.
    def redirect_request(self, *args):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


def http_error(exc):
    try:
        error = json.loads(exc.read()).get("error", "")
        return (error.get("message", "") if isinstance(error, dict) else error) or exc.reason
    except (ValueError, AttributeError, OSError):
        return exc.reason


class Job:
    def __init__(self, text, path, label):
        self.text, self.path, self.label = text, path, label
        self.failed_on: collections.Counter[str] = collections.Counter()
        self.done = False


PROBE = object()


class Pool:
    """Synthesize jobs across endpoints; each finished job's WAV is written to job.path."""

    def __init__(self, endpoints, request):
        self.endpoints, self.request = endpoints, request
        self.cond = threading.Condition()
        self.queue: collections.deque[Job] = collections.deque()
        self.remaining = 0
        self.error = None
        self.started = time.monotonic()

    def health(self, e):
        with OPENER.open(
            urllib.request.Request(e.base + "/v1/health", headers=e.headers()), timeout=PROBE_TIMEOUT
        ) as resp:
            e.health = json.loads(resp.read())
        return e.health

    def probe(self):
        """Check every endpoint's /health and return the replies; an endpoint that does not answer joins later."""
        replies = []
        for e in self.endpoints:
            try:
                replies.append(self.health(e))
            except (urllib.error.URLError, OSError, ValueError) as exc:
                print(
                    f"warning: endpoint {e.name} ({e.base}) is not answering: {exc}; retrying it during the run",
                    file=sys.stderr,
                )
                self.fail(e)
        if not replies:
            raise SystemExit("error: no endpoint is reachable")
        return replies

    def start(self, jobs):
        self.queue.extend(jobs)
        self.remaining = len(self.queue)
        for e in self.endpoints:
            for _ in range(e.slots):
                threading.Thread(target=self.worker, args=(e,), daemon=True).start()

    def wait(self, jobs):
        """Block until the given jobs are done; raise if the run failed."""
        with self.cond:
            while self.error is None and not all(j.done for j in jobs):
                self.cond.wait()
            if self.error is not None:
                raise SystemExit(f"error: {self.error}")

    def stop(self, error):
        with self.cond:
            if self.error is None:
                self.error = error
            self.cond.notify_all()

    def fail(self, e):
        now = time.monotonic()
        if e.failing_since is None:
            e.failing_since = now
        e.retry_at = now + e.backoff
        e.backoff = min(e.backoff * 2, BACKOFF[1])

    def take(self, e):
        """Next job for one of an endpoint's workers, PROBE to recheck a failing endpoint, or None to stop."""
        with self.cond:
            while True:
                live = [x for x in self.endpoints if not x.dropped]
                now = time.monotonic()
                if self.error is None and not live:
                    self.error = "every endpoint was dropped"
                elif self.error is None and all(
                    x.failing_since is not None and now - x.failing_since > STALL_SECONDS for x in live
                ):
                    self.error = f"no endpoint has answered for {STALL_SECONDS // 60} minutes"
                if self.error is not None or e.dropped or not self.remaining:
                    self.cond.notify_all()
                    return None
                if now < e.retry_at:
                    self.cond.wait(e.retry_at - now)
                elif e.failing_since is not None:
                    # Recheck without holding a paragraph, which other endpoints can take meanwhile.
                    e.retry_at = now + PROBE_TIMEOUT
                    return PROBE
                elif job := self.next_job(e, live):
                    return job
                else:
                    # Idle until a job is requeued or the last one finishes.
                    self.cond.wait(BUSY_SECONDS)

    def next_job(self, e, live):
        """The first queued job, skipping those this endpoint already failed while another endpoint has not."""
        for job in self.queue:
            if not job.failed_on[e.name] or all(job.failed_on[x.name] for x in live):
                self.queue.remove(job)
                return job
        return None

    def requeue(self, job):
        with self.cond:
            self.queue.appendleft(job)
            self.cond.notify_all()

    def worker(self, e):
        try:
            while (job := self.take(e)) is not None:
                if job is PROBE:
                    self.recheck(e)
                else:
                    self.send(e, job)
        except BaseException as exc:
            self.stop(f"endpoint {e.name} worker crashed: {exc!r}")

    def recheck(self, e):
        try:
            self.health(e)
        except (urllib.error.URLError, OSError, ValueError):
            with self.cond:
                self.fail(e)
            return
        with self.cond:
            if e.failing_since is not None:
                print(f"[{e.name}] /v1/health answers again; resuming", flush=True)
            e.failing_since, e.retry_at = None, 0.0
            self.cond.notify_all()

    def send(self, e, job):
        name = f"[{e.name}] {job.label}"
        body = json.dumps({"input": job.text, "response_format": "wav", **self.request}).encode()
        print(f"{name} {job.text[:60]}", flush=True)
        try:
            req = urllib.request.Request(e.base + "/v1/audio/speech", body, e.headers())
            with OPENER.open(req, timeout=REQUEST_TIMEOUT) as resp:
                audio, headers = resp.read(), resp.headers
            if not headers.get_content_type().startswith("audio/"):
                return self.drop(e, job, f"replied {headers.get_content_type()} instead of audio")
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308, 401, 403):
                return self.drop(e, job, f"HTTP {exc.code}; check its token")
            if exc.code == 404:
                return self.drop(e, job, "HTTP 404; check its url")
            if exc.code in (429, 503):
                # Busy: a full Ray Serve queue, such as while voice chat or the page is using every replica. Hand the
                # paragraph to another endpoint and look again shortly; being busy is not a failure.
                with self.cond:
                    if not e.busy:
                        print(f"[{e.name}] busy (HTTP {exc.code}); retrying every {BUSY_SECONDS} s", flush=True)
                    e.busy = True
                    e.retry_at = time.monotonic() + BUSY_SECONDS
                return self.requeue(job)
            if 400 <= exc.code < 500:
                # Bad text or settings fail the same way on every endpoint.
                return self.stop(f"{job.label} failed on {e.name} (HTTP {exc.code}): {http_error(exc)}")
            return self.retry(e, job, f"HTTP {exc.code}: {http_error(exc)}")
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            return self.retry(e, job, str(getattr(exc, "reason", exc)) or type(exc).__name__)
        with open(job.path, "wb") as f:
            f.write(audio)
        seconds = float(headers["X-Audio-Seconds"])
        print(f"{name} {headers['X-Audio-Seconds']} s audio, RTF {headers['X-RTF']}", flush=True)
        if headers.get("X-Token-Limited") == "true":
            print(f"warning: {job.label} reached the token limit; check for truncation or repetition", file=sys.stderr)
        with self.cond:
            e.backoff, e.busy = BACKOFF[0], False
            e.done += 1
            e.audio += seconds
            e.synthesis += float(headers["X-Synthesis-Seconds"])
            job.done = True
            self.remaining -= 1
            self.cond.notify_all()

    def retry(self, e, job, reason):
        with self.cond:
            job.failed_on[e.name] += 1
            live = [x for x in self.endpoints if not x.dropped]
            if job.failed_on.total() >= MAX_ATTEMPTS and all(job.failed_on[x.name] for x in live):
                self.error = f"{job.label} failed {job.failed_on.total()} times; last on {e.name}: {reason}"
            else:
                print(
                    f"[{e.name}] {reason}; requeueing {job.label} and backing off {e.backoff} s",
                    file=sys.stderr,
                    flush=True,
                )
                self.fail(e)
                self.queue.appendleft(job)
            self.cond.notify_all()

    def drop(self, e, job, reason):
        with self.cond:
            if not e.dropped:
                print(
                    f"warning: dropping endpoint {e.name} ({e.base}) for this run: {reason}",
                    file=sys.stderr,
                    flush=True,
                )
            e.dropped = reason
            self.queue.appendleft(job)
            self.cond.notify_all()

    def summary(self):
        for e in self.endpoints:
            rtf = f", RTF {e.synthesis / e.audio:.3f}" if e.audio else ""
            dropped = f", dropped: {e.dropped}" if e.dropped else ""
            print(f"  {e.name}: {e.done} paragraphs, {e.audio / 60:.1f} min audio{rtf}{dropped}")
        print(f"  {(time.monotonic() - self.started) / 60:.1f} min wall time")


def check_speed(replies, speed, error):
    for health in replies:
        if not health["speed"]["min"] <= speed <= health["speed"]["max"]:
            error(f"--speed must be between {health['speed']['min']} and {health['speed']['max']}")


def models(replies):
    """The models the reachable endpoints report, for the output tags."""
    return ",".join(sorted({h["model"] for h in replies}))


def without_model(comment):
    """A tag minus its model field: endpoints name the same model differently, e.g. MLX and CUDA checkpoints."""
    return re.sub(r"^model=[^;]*;", "", comment)
