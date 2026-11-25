"""serve_tts's HTTP surface (site-up's page, txt2mp3.sh and voice chat depend on it), on a replica with a fake model.

The routes, request validation, generate() and generate_stream() are the real ones; only Qwen3TTSModel is replaced.
"""

import asyncio
import base64
import io
import json
import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import soundfile as sf
import torch
from conftest import CODE_GROUPS, serve_client, tone, wav_bytes

import serve_tts

RATE = 24000


class FakeTTS:
    """Qwen3TTSModel as the replica uses it: one token per word, and a tone per synthesis."""

    def __init__(self, decoder=None):
        self.calls = []
        self.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text: text.split()))
        self.model = SimpleNamespace(
            speech_tokenizer=SimpleNamespace(
                model=SimpleNamespace(decoder=decoder), get_output_sample_rate=lambda: RATE
            )
        )
        self.audio = tone(seconds=1.5, rate=RATE, amplitude=0.05)

    def create_voice_clone_prompt(self, ref_audio, ref_text):
        self.calls.append(("prompt", ref_audio, ref_text))
        return [SimpleNamespace(ref_code=torch.zeros((5, CODE_GROUPS), dtype=torch.long))]

    def generate_voice_clone(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return [self.audio], RATE


def make_replica(voices_dir, tts=None):
    replica = object.__new__(cast(Any, serve_tts.TTS).func_or_class)
    replica.model_id = "Qwen/tiny"
    replica.voices_dir = voices_dir
    replica.tts = tts or FakeTTS()
    replica.job = None
    replica.cuda_graphs = True
    replica.stream_context = 300
    replica.prompts = {}
    replica.languages = ["auto", "chinese", "english"]
    replica.sampling = dict(serve_tts.SAMPLING)
    replica.device = "CPU / CUDA none"
    return replica


@pytest.fixture
def voices_dir(tmp_path):
    (tmp_path / "alice.wav").write_bytes(wav_bytes(tone(seconds=1.0, rate=RATE, amplitude=0.1), RATE))
    (tmp_path / "alice.txt").write_text("Hello there.", encoding="utf-8")
    (tmp_path / "bob.wav").write_bytes(wav_bytes(tone(seconds=1.0, rate=RATE), RATE))  # no transcript
    (tmp_path / "empty.wav").write_bytes(wav_bytes(tone(seconds=1.0, rate=RATE), RATE))
    (tmp_path / "empty.txt").write_text("  \n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def replica(voices_dir, no_cuda):
    return make_replica(voices_dir)


@pytest.fixture
def client(replica):
    yield serve_client(serve_tts, replica)
    serve_tts.api.dependency_overrides.clear()


def tts(client, **body):
    return client.post("/api/tts", json={"text": "Hello world.", "voice": "alice", **body})


def test_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["cache-control"] == "no-store"
    assert b"<html" in response.content.lower()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "Qwen/tiny"
    assert body["languages"] == ["auto", "chinese", "english"]
    assert body["sampling"]["top_k"] == {"default": 50, "min": 0, "max": 1024, "integer": True}
    assert body["sampling"]["max_tokens"]["default"] is None
    assert body["speed"] == {"default": 1.0, "min": 0.5, "max": 2.0}
    assert body["volume"] == {"default": -20.0, "min": -40.0, "max": -9.0}


def test_voices_need_a_transcript(client):
    assert client.get("/voices").json() == ["alice", "empty"]


def test_unknown_route_is_json_404(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json() == {"error": "Not found"}
    assert response.headers["cache-control"] == "no-store"


def test_synthesize_wav(client, replica):
    response = tts(client, seed=7, language="english", temperature=0.5, top_k=20)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["cache-control"] == "no-store"
    audio, rate = sf.read(io.BytesIO(response.content))
    assert rate == RATE
    assert audio.size == replica.tts.audio.size
    assert float(response.headers["X-Audio-Seconds"]) == pytest.approx(1.5)
    assert float(response.headers["X-Loudness-LUFS"]) == pytest.approx(-20.0, abs=0.1)
    assert response.headers["X-Loudness-Limited"] == "false"
    assert response.headers["X-Token-Limited"] == "false"
    assert response.headers["X-Peak-CUDA-GB"] == "1.50"
    for header in ("X-Synthesis-Seconds", "X-RTF"):
        float(response.headers[header])

    (_, call) = replica.tts.calls[-1]
    assert call["text"] == "Hello world."
    assert call["language"] == "english"
    assert call["do_sample"] is True and call["temperature"] == 0.5
    assert call["top_k"] == 20 and call["top_p"] == 1.0 and call["repetition_penalty"] == 1.05
    assert call["max_new_tokens"] == serve_tts.MIN_SPEECH_TOKEN_BUDGET  # 2 words is under the minimum budget


def test_synthesize_defaults(client, replica):
    assert tts(client).status_code == 200
    (_, call) = replica.tts.calls[-1]
    assert call["language"] == "Auto"
    assert call["temperature"] == 0.9


def test_greedy_leaves_temperature_unset(client, replica):
    assert tts(client, temperature=0).status_code == 200
    (_, call) = replica.tts.calls[-1]
    assert call["do_sample"] is False and call["temperature"] is None


def test_speed_and_volume(client, replica):
    response = tts(client, speed=2.0, volume=-30)
    assert response.status_code == 200
    assert float(response.headers["X-Audio-Seconds"]) == pytest.approx(0.75)
    assert float(response.headers["X-Loudness-LUFS"]) == pytest.approx(-30.0, abs=0.1)


def test_loud_target_is_limited(client, replica):
    replica.tts.audio = replica.tts.audio.copy()
    replica.tts.audio[100] = 0.5  # a click that would clip at the gain the tone needs
    response = tts(client, volume=-9)
    assert response.status_code == 200
    assert response.headers["X-Loudness-Limited"] == "true"


def test_token_budget_scales_with_text(client, replica):
    text = " ".join(["word"] * 100)
    assert tts(client, text=text).status_code == 200
    (_, call) = replica.tts.calls[-1]
    assert call["max_new_tokens"] == 100 * serve_tts.SPEECH_TOKEN_BUDGET_PER_TEXT_TOKEN


def test_explicit_max_tokens(client, replica):
    assert tts(client, max_tokens=500).status_code == 200
    assert replica.tts.calls[-1][1]["max_new_tokens"] == 500


def test_text_too_long_for_max_tokens(client):
    response = tts(client, text=" ".join(["word"] * 30), max_tokens=100)
    assert response.status_code == 413
    assert response.json() == {"error": "Text is 30 tokens; at most 25 fit in max_tokens", "tokens": 30, "limit": 25}


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"text": ""}, "Enter between 1 and 20000 characters"),
        ({"text": "   "}, "Enter between 1 and 20000 characters"),
        ({"text": "x" * 20_001}, "Enter between 1 and 20000 characters"),
        ({"text": 5}, "Enter between 1 and 20000 characters"),
        ({"voice": "bob"}, "Select a reference voice with an exact transcript"),
        ({"voice": "../alice"}, "Select a reference voice with an exact transcript"),
        ({"seed": -1}, "Seed must be between 0 and 4294967295"),
        ({"seed": 2**32}, "Seed must be between 0 and 4294967295"),
        ({"stream": "yes"}, "stream must be true or false"),
        ({"stream": True, "speed": 1.5}, "speed is not supported when streaming"),
        ({"speed": 0.4}, "speed must be between 0.5 and 2.0"),
        ({"speed": 2.1}, "speed must be between 0.5 and 2.0"),
        ({"volume": -41}, "volume must be between -40.0 and -9.0 LUFS"),
        ({"volume": -8}, "volume must be between -40.0 and -9.0 LUFS"),
        ({"language": "klingon"}, "Language must be one of: auto, chinese, english"),
        ({"temperature": 1.6}, "temperature must be between 0.0 and 1.5"),
        ({"top_k": -1}, "top_k must be between 0 and 1024"),
        ({"top_p": 0.01}, "top_p must be between 0.05 and 1.0"),
        ({"repetition_penalty": 0.9}, "repetition_penalty must be between 1.0 and 3.0"),
        ({"max_tokens": 0}, "max_tokens must be between 1 and 4096"),
        ({"max_tokens": 4097}, "max_tokens must be between 1 and 4096"),
        ({"speed": "fast"}, "could not convert string to float: 'fast'"),
    ],
)
def test_invalid_requests(client, replica, body, message):
    response = tts(client, **body)
    assert response.status_code == 400
    assert response.json() == {"error": message}
    assert response.headers["cache-control"] == "no-store"
    assert not [c for c in replica.tts.calls if c[0] == "generate"]


@pytest.mark.parametrize("content", [b"", b"x" * 100_001, b"not json", b"[]", b'{"text": "hi"}'])
def test_malformed_bodies(client, content):
    response = client.post("/api/tts", content=content)
    assert response.status_code == 400
    assert "error" in response.json()


def test_empty_transcript_fails_per_request(client):
    response = tts(client, voice="empty")
    assert response.status_code == 500
    assert response.json() == {"error": "Reference transcript is empty"}


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (np.zeros(0, dtype=np.float32), "Model generated no audio"),
        (np.full(100, np.nan, dtype=np.float32), "Model generated non-finite audio"),
    ],
)
def test_bad_model_output_is_500(client, replica, audio, message):
    replica.tts.audio = audio
    response = tts(client)
    assert response.status_code == 500
    assert response.json() == {"error": message}


def test_prompts_are_cached_until_files_change(client, replica, voices_dir):
    assert tts(client).status_code == 200
    assert tts(client).status_code == 200
    prompts = [c for c in replica.tts.calls if c[0] == "prompt"]
    assert prompts == [("prompt", str(voices_dir / "alice.wav"), "Hello there.")]

    (voices_dir / "alice.txt").write_text("Changed.", encoding="utf-8")
    assert tts(client).status_code == 200
    prompts = [c for c in replica.tts.calls if c[0] == "prompt"]
    assert prompts[-1] == ("prompt", str(voices_dir / "alice.wav"), "Changed.")


def test_disconnect_stops_generation(replica):
    """A cancelled request (client gone) sets the job's stop event and waits for generation to end."""
    started, stopped = threading.Event(), threading.Event()

    def generate(text, voice, seed, speed, volume, language, sampling, job):
        started.set()
        assert job.stop.wait(5)
        stopped.set()
        raise serve_tts.ClientGone

    replica.generate = generate

    async def main():
        body = json.dumps({"text": "Hello", "voice": "alice"}).encode()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        from starlette.requests import Request

        request = Request({"type": "http", "method": "POST", "path": "/api/tts", "headers": []}, receive)
        task = asyncio.ensure_future(replica.synthesize(request))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()  # generation finished before the handler gave the replica back

    asyncio.run(main())


def ndjson(response):
    return [json.loads(line) for line in response.text.splitlines()]


def test_stream(voices_dir, no_cuda, tiny_decoder):
    replica = make_replica(voices_dir, FakeTTS(tiny_decoder))
    generated = torch.randint(0, 32, (30, CODE_GROUPS))

    def run(text, voice, seed, language, sampling, job):
        """What the talker hooks do during generate_voice_clone: hand over frames, flush, finish."""
        replica.started = time.perf_counter()
        for frame in generated:
            job.stream.frame(frame)
        job.token_limited = True
        job.stream.flush()
        raise serve_tts.StreamFinished

    replica.run = run
    client = serve_client(serve_tts, replica)
    try:
        response = tts(client, stream=True, volume=-25)
    finally:
        serve_tts.api.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = ndjson(response)
    assert [e["type"] for e in events] == ["audio", "audio", "audio", "done"]  # 12 + 12 + 6 frames
    pcm = b"".join(base64.b64decode(e["pcm"]) for e in events[:-1])
    assert len(pcm) == 2 * 30 * int(tiny_decoder.total_upsample)
    assert all(e["sample_rate"] == RATE for e in events[:-1])
    done = events[-1]
    assert done["audio_seconds"] == pytest.approx(30 * int(tiny_decoder.total_upsample) / RATE, abs=0.01)
    assert done["token_limited"] is True
    assert done["peak_cuda_gb"] == 1.5
    reference = serve_tts.integrated_loudness(tone(seconds=1.0, rate=RATE, amplitude=0.1), RATE)
    assert done["gain_db"] == pytest.approx(-25 - reference, abs=0.1)
    assert {"synthesis_seconds", "first_audio_seconds", "rtf", "clipped_samples"} <= done.keys()


def test_stream_error_event(voices_dir, no_cuda, tiny_decoder):
    replica = make_replica(voices_dir, FakeTTS(tiny_decoder))

    def run(*args):
        replica.started = 0.0
        raise RuntimeError("CUDA out of memory")

    replica.run = run
    client = serve_client(serve_tts, replica)
    try:
        events = ndjson(tts(client, stream=True))
    finally:
        serve_tts.api.dependency_overrides.clear()
    assert events == [{"type": "error", "error": "CUDA out of memory"}]


def test_stream_without_hook_is_an_error(voices_dir, no_cuda, tiny_decoder):
    """generate_voice_clone returning normally means the talker hooks did not run."""
    replica = make_replica(voices_dir, FakeTTS(tiny_decoder))
    replica.run = lambda *a: setattr(replica, "started", 0.0)
    client = serve_client(serve_tts, replica)
    try:
        events = ndjson(tts(client, stream=True))
    finally:
        serve_tts.api.dependency_overrides.clear()
    assert events == [{"type": "error", "error": "Generation ended without streaming"}]


def test_stream_disconnect_stops_generation(voices_dir, no_cuda, tiny_decoder):
    """A streamed request whose client goes away sets the job's stop event and waits for generation to end."""
    replica = make_replica(voices_dir, FakeTTS(tiny_decoder))
    stopped = threading.Event()

    def run(text, voice, seed, language, sampling, job):
        """Streams one piece, then waits as the talker hook would until the disconnect stops it."""
        replica.started = time.perf_counter()
        for frame in torch.randint(0, 32, (serve_tts.STREAM_FRAMES, CODE_GROUPS)):
            job.stream.frame(frame)
        assert job.stop.wait(5)
        stopped.set()
        raise serve_tts.ClientGone

    replica.run = run
    sampling = {name: d for name, (_, d, _, _) in replica.sampling.items()}

    async def main():
        response = replica.stream_speech("Hello", "alice", 0, -23.0, "auto", sampling)
        events, first = [], asyncio.Event()

        async def consume():
            async for line in response.body_iterator:
                events.append(json.loads(line))
                first.set()

        consumer = asyncio.ensure_future(consume())
        await asyncio.wait_for(first.wait(), 5)
        consumer.cancel()  # what Starlette does to the response body when the client disconnects
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert stopped.is_set()  # generation finished before the response gave the replica back
        assert [e["type"] for e in events] == ["audio"]  # no done or error after the disconnect

    asyncio.run(main())
