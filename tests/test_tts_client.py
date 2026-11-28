"""scripts/tts_client.py against serve_tts's real routes (fake model) on loopback."""

import socket
import threading
import time
import wave
from types import SimpleNamespace

import pytest
import uvicorn
from conftest import serve_client
from test_serve_tts_api import make_replica, voices_dir  # noqa: F401  # voices_dir is a fixture

import serve_tts
import tts_client


@pytest.fixture
def tts_url(voices_dir, no_cuda):  # noqa: F811
    serve_client(serve_tts, make_replica(voices_dir))  # binds the routes to the replica
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(serve_tts.api, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    server.should_exit = True
    thread.join(10)
    serve_tts.api.dependency_overrides.clear()


def pool(url, voice="alice"):
    args = SimpleNamespace(endpoints=None, tts_url=url, parallel=None)
    request = {"voice": voice, "language": "auto", "speed": 1.0, **tts_client.SETTINGS}
    return tts_client.Pool(tts_client.load_endpoints(args, pytest.fail), request)


@pytest.mark.parametrize("suffix", ["", "/v1"])
def test_paragraphs_are_synthesized(tts_url, tmp_path, suffix):
    run = pool(tts_url + suffix)
    replies = run.probe()
    assert tts_client.models(replies) == "Qwen/tiny"
    tts_client.check_speed(replies, 1.0, pytest.fail)
    jobs = [tts_client.Job(text, str(tmp_path / f"{i}.wav"), f"[{i}/2]") for i, text in enumerate(["One.", "Two."], 1)]
    run.start(jobs)
    run.wait(jobs)
    for job in jobs:
        with wave.open(job.path) as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
            assert w.getnframes() == 36000  # the fake model's 1.5 s tone
    endpoint = run.endpoints[0]
    assert endpoint.done == 2 and endpoint.audio == pytest.approx(3.0)


def test_rejected_request_stops_the_run(tts_url, tmp_path):
    run = pool(tts_url, voice="nobody")
    run.probe()
    jobs = [tts_client.Job("One.", str(tmp_path / "1.wav"), "[1/1]")]
    run.start(jobs)
    with pytest.raises(SystemExit, match=r"\[1/1\] failed on tts \(HTTP 400\): Select a reference voice"):
        run.wait(jobs)


def test_an_endpoint_is_required(monkeypatch, tmp_path):
    monkeypatch.setattr(tts_client, "DEFAULT_CONFIG", str(tmp_path / "missing.toml"))
    errors = []

    def error(message):
        errors.append(message)
        raise SystemExit(message)

    with pytest.raises(SystemExit):
        tts_client.load_endpoints(SimpleNamespace(endpoints=None, tts_url=None, parallel=None), error)
    assert errors == [f"set --tts-url (or TTS_URL), or list endpoints in {tmp_path / 'missing.toml'}"]
