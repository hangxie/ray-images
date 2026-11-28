"""serve_stt's transcription API and live stream, on a replica with a fake Whisper."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from conftest import serve_client, tone, wav_bytes
from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE

import serve_stt


@pytest.fixture
def replica(monkeypatch):
    monkeypatch.setattr(serve_stt, "PARTIAL_SECONDS", 1e9)  # no previews, except in test_stream_previews
    replica = object.__new__(cast(Any, serve_stt.STT).func_or_class)
    replica.model_id = "openai/whisper-tiny"
    replica.device = "CPU / CUDA none"
    replica.languages, replica.language_codes = LANGUAGES, TO_LANGUAGE_CODE
    replica.calls = []
    replica.gpu = asyncio.Lock()
    replica.detector = lambda: loud  # exact cuts; Silero itself is tested on real speech below

    def transcribe(audio, language, prompt=""):
        replica.calls.append((audio, language))
        return "Hello world.", language or "fr"

    replica.transcribe = transcribe
    return replica


@pytest.fixture
def client(replica):
    yield serve_client(serve_stt, replica)
    serve_stt.api.dependency_overrides.clear()


def transcribe(client, audio=None, **fields):
    audio = wav_bytes(tone(seconds=1.0, rate=16000), 16000) if audio is None else audio
    return client.post("/v1/audio/transcriptions", files={"file": ("a.wav", audio, "audio/wav")}, data=fields)


def test_health_and_models(client):
    assert client.get("/v1/health").json() == {
        "status": "ok",
        "version": serve_stt.VERSION,
        "model": "openai/whisper-tiny",
        "device": "CPU / CUDA none",
        "languages": TO_LANGUAGE_CODE,  # for serve_meeting's language menu
    }
    assert TO_LANGUAGE_CODE["mandarin"] == "zh"
    assert client.get("/v1/models").json() == {
        "object": "list",
        "data": [{"id": "openai/whisper-tiny", "object": "model", "owned_by": "local"}],
    }


def test_unknown_route_is_openai_error(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json() == {"error": {"message": "Not found"}}


def test_json_transcription_detects_language(client, replica):
    response = transcribe(client, model="whisper-1")
    assert response.status_code == 200, response.text
    assert response.json() == {"text": "Hello world.", "language": "french", "duration": 1.0}
    assert response.headers["cache-control"] == "no-store"
    assert float(response.headers["X-Audio-Seconds"]) == 1.0
    for header in ("X-Transcribe-Seconds", "X-RTF"):
        float(response.headers[header])
    audio, language = replica.calls[-1]
    assert language is None
    assert audio.dtype == np.float32 and audio.size == 16000


@pytest.mark.parametrize(("given", "code"), [("en", "en"), ("English", "en"), (" german ", "de"), ("zh", "zh")])
def test_language_by_code_or_name(client, replica, given, code):
    assert transcribe(client, language=given).status_code == 200
    assert replica.calls[-1][1] == code


def test_text_format(client):
    response = transcribe(client, response_format="text")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.text == "Hello world."


def test_verbose_json_is_accepted(client):
    assert transcribe(client, response_format="verbose_json").json()["text"] == "Hello world."


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000])
def test_audio_is_resampled_to_16k_mono(client, replica, rate):
    stereo = np.stack([tone(seconds=2.0, rate=rate), tone(seconds=2.0, rate=rate, freq=440)], axis=1)
    assert transcribe(client, wav_bytes(stereo, rate)).status_code == 200
    audio, _ = replica.calls[-1]
    assert audio.dtype == np.float32
    assert audio.size == 32000


def test_flac_upload(client, replica):
    import io

    import soundfile as sf

    out = io.BytesIO()
    sf.write(out, tone(seconds=1.0, rate=16000), 16000, format="FLAC")
    assert transcribe(client, out.getvalue()).status_code == 200


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"response_format": "srt"}, "response_format must be json, verbose_json, or text"),
        ({"language": "klingon"}, "Unknown language: klingon"),
        ({"audio": b"not audio"}, "Unsupported or corrupt audio"),
        ({"audio": wav_bytes(np.zeros(0, dtype=np.float32), 16000)}, "Audio must be between 0 and 300 seconds"),
        ({"audio": wav_bytes(np.zeros(301 * 8000, dtype=np.float32), 8000)}, "Audio must be between 0 and 300 seconds"),
    ],
)
def test_invalid_requests(client, replica, kwargs, message):
    response = transcribe(client, **kwargs)
    assert response.status_code == 400
    assert response.json()["error"]["message"].startswith(message)
    assert not replica.calls


def test_missing_file(client):
    response = client.post("/v1/audio/transcriptions", files={"other": ("a.wav", b"x")})
    assert response.status_code == 400
    assert response.json() == {"error": {"message": "Missing audio file field"}}


def test_not_multipart(client):
    response = client.post("/v1/audio/transcriptions", json={"file": "x"})
    assert response.status_code == 400
    assert response.json() == {"error": {"message": "Request must be multipart/form-data"}}


def test_empty_body(client):
    response = client.post("/v1/audio/transcriptions", content=b"")
    assert response.status_code == 400


def test_transcription_failure_is_500(client, replica):
    def fail(audio, language, prompt=""):
        raise RuntimeError("CUDA error")

    replica.transcribe = fail
    response = transcribe(client)
    assert response.status_code == 500
    assert response.json() == {"error": {"message": "CUDA error"}}


def test_multipart_fields_keeps_binary_payload():
    payload = bytes(range(256)) * 4
    boundary = "xyz"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.bin"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + payload
        + (
            f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\nen\r\n--{boundary}--\r\n'
        ).encode()
    )
    fields = serve_stt.multipart_fields(f"multipart/form-data; boundary={boundary}", body)
    assert fields == {"file": payload, "language": b"en"}


def test_page_uses_relative_urls(client):
    """A plain recording page; live features belong to serve_meeting."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["cache-control"] == "no-store"
    assert "fetch('v1/audio/transcriptions'" in response.text
    assert "fetch('/" not in response.text  # absolute URLs would miss a route prefix
    assert "WebSocket" not in response.text and "localStorage" not in response.text
    assert f'<span id="version">{serve_stt.VERSION}</span>' in response.text


FRAME = serve_stt.Segmenter.FRAME / 16000  # seconds


def speech(seconds, amplitude=0.1):
    """A tone: speech to the `loud` detector, which the cutting-rule tests use."""
    return tone(seconds=seconds, rate=16000, amplitude=amplitude)


def silence(seconds):
    return np.zeros(round(seconds * 16000), dtype=np.float32)


def loud(frame):
    """A stand-in for Silero VAD with exact answers: a frame louder than -40 dBFS is speech."""
    return 0.9 if np.sqrt(np.mean(np.square(frame))) > 0.01 else 0.0


def segment(audio, segmenter=None, step=1600):
    """Pieces of `audio` fed in 0.1 s chunks, as the page sends them, then flushed."""
    segmenter = segmenter or serve_stt.Segmenter(loud)
    pieces = [p for i in range(0, len(audio), step) for p in segmenter.feed(audio[i : i + step])]
    return pieces + segmenter.flush()


def spans(pieces):
    return [(round(p["start"], 2), round(p["end"], 2), p["paragraph"]) for p in pieces]


def test_short_speech_is_one_piece_trimmed_to_speech():
    pieces = segment(np.concatenate([silence(1.0), speech(2.0), silence(1.0)]))
    # Speech in frames 31 (0.992 s, which holds its first 8 ms) to 93; 10 frames (0.32 s) of margin each side.
    assert spans(pieces) == [(round(21 * FRAME, 2), round(104 * FRAME, 2), True)]
    assert pieces[0]["audio"].size == 83 * 512


def test_pause_after_min_seconds_ends_a_piece():
    # The 0.2 s pause at 3 s is too early; the one at 6 s ends the piece once 8 quiet frames (0.256 s) have gone by.
    audio = np.concatenate([speech(3.0), silence(0.2), speech(2.8), silence(0.5), speech(2.0)])
    pieces = spans(segment(audio))
    assert len(pieces) == 2 and pieces[0][:1] == (0.0,) and pieces[1][2] is False
    assert 6.0 < pieces[0][1] < 6.3 and pieces[0][1] == pieces[1][0]


def test_long_speech_is_cut_at_the_frame_least_like_speech_by_max_seconds():
    audio = speech(20.0, amplitude=0.2)
    dip = round(9.0 / FRAME)  # one frame that isn't speech, too short for a pause
    audio[dip * 512 : (dip + 1) * 512] = 0
    assert spans(segment(audio)) == [
        (0.0, round((dip + 1) * FRAME, 2), True),
        (round((dip + 1) * FRAME, 2), 20.0, False),
    ]


def test_silence_is_dropped_and_a_long_gap_starts_a_paragraph():
    audio = np.concatenate([speech(1.0), silence(12.0), speech(1.0), silence(1.0), speech(1.0)])
    pieces = spans(segment(audio, serve_stt.Segmenter(loud, min_seconds=1.0)))
    assert [(round(start), paragraph) for start, _, paragraph in pieces] == [(0, True), (13, True), (15, False)]


def test_no_speech_gives_no_pieces():
    assert segment(silence(30.0)) == []
    assert segment(np.concatenate([silence(5.0), speech(0.1), silence(5.0)])) == []


def test_short_runs_of_speech_do_not_count():
    # 0.2 s blips (a beat, a clatter) from 3 s to 6 s, between sentences at 0 s and 7 s: they make no piece, and the
    # sentences are still apart enough for two paragraphs.
    blips = np.concatenate([np.concatenate([speech(0.2), silence(0.4)]) for _ in range(5)])
    audio = np.concatenate([speech(2.0), silence(1.0), blips, silence(1.0), speech(2.0)])
    pieces = spans(segment(audio, serve_stt.Segmenter(loud, min_seconds=1.0)))
    assert [p[2] for p in pieces] == [True, True]
    assert pieces[0][1] < 2.4 and pieces[1][0] > 6.6


CLIP = Path(__file__).with_name("data") / "speech.flac"  # two sentences 0.5 s apart, from macOS `say`


def music(seconds, level_db, rng):
    """Music with a beat, which a loudness detector took for continuous speech."""
    t = np.arange(round(seconds * 16000)) / 16000
    chords = [(220.0, 277.2, 329.6), (196.0, 246.9, 293.7), (174.6, 220.0, 261.6), (196.0, 246.9, 293.7)]
    bar = (t // 2).astype(int) % len(chords)
    beat = 0.05 + 0.95 * np.exp(-((t % 0.5) * 12))
    out = sum(np.sin(2 * np.pi * np.array([c[k] for c in chords])[bar] * t) for k in range(3)) * beat
    out = out + rng.standard_normal(t.size) * np.exp(-(((t + 0.25) % 0.5) * 40)) * 0.5
    return (out * 10 ** (level_db / 20) / np.sqrt(np.mean(out**2))).astype(np.float32)


@pytest.fixture(scope="module")
def silero():
    return cast(Any, serve_stt.STT).func_or_class.detector


def test_silero_finds_paragraphs_under_music(silero):
    import soundfile as sf

    clip, _ = sf.read(CLIP, dtype="float32")
    voice = np.concatenate([clip, silence(3.0), clip, silence(3.0), clip])
    starts = [0.0, clip.size / 16000 + 3, 2 * (clip.size / 16000 + 3)]
    rng = np.random.default_rng(0)
    for level in (None, -45, -35, -25, -20):
        audio = voice if level is None else voice + music(voice.size / 16000, level, rng)
        pieces = segment(audio, serve_stt.Segmenter(silero(None)))
        paragraphs = [p["start"] for p in pieces if p["paragraph"]]
        assert len(paragraphs) == 3 and all(abs(p - s) < 0.5 for p, s in zip(paragraphs, starts, strict=True)), (
            level,
            spans(pieces),
        )


def test_silero_drops_music_noise_and_silence(silero):
    rng = np.random.default_rng(1)
    for audio in (music(30.0, -20, rng), (rng.standard_normal(30 * 16000) * 0.03).astype(np.float32), silence(30.0)):
        assert segment(audio, serve_stt.Segmenter(silero(None))) == []


def pcm(audio):
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def stream(client, audio, query=""):
    """Events of a live transcription of `audio`, sent in 0.1 s chunks."""
    events: list[dict] = []
    with client.websocket_connect(f"/v1/audio/stream{query}") as ws:
        for i in range(0, len(audio), 1600):
            ws.send_bytes(pcm(audio[i : i + 1600]))
        ws.send_text(json.dumps({"type": "stop"}))
        while not events or events[-1]["type"] != "done":
            events.append(ws.receive_json())
    return events


def test_stream_transcribes_pieces_with_context(client, replica):
    prompts = []

    def transcribe(audio, language, prompt=""):
        prompts.append((prompt, language))
        return f"Piece {len(prompts)}.", language or "fr"

    replica.transcribe = transcribe
    audio = np.concatenate([speech(5.5), silence(3.0), speech(2.0), silence(0.5), speech(1.0)])
    events = stream(client, audio)
    assert [e.pop("seconds") >= 0 for e in events[:-1]] == [True] * 3
    assert events == [
        {"type": "segment", "start": 0.0, "end": 5.76, "text": "Piece 1.", "language": "french", "paragraph": True},
        {"type": "segment", "start": 8.16, "end": 10.78, "text": "Piece 2.", "language": "french", "paragraph": True},
        {"type": "segment", "start": 10.78, "end": 12.0, "text": "Piece 3.", "language": "french", "paragraph": False},
        {"type": "done"},
    ]
    # Each piece gets the text so far as its prompt, and the language found in the first.
    assert prompts == [("", None), ("Piece 1.", "fr"), ("Piece 1. Piece 2.", "fr")]


def test_stream_previews_the_piece_being_spoken(client, replica, monkeypatch):
    monkeypatch.setattr(serve_stt, "PARTIAL_SECONDS", 1.0)
    seen = []

    def transcribe(audio, language, prompt=""):
        seen.append(audio.size / 16000)
        return f"{audio.size / 16000:.1f} s.", "en"

    replica.transcribe = transcribe
    events = stream(client, np.concatenate([speech(8.0), silence(1.0), speech(3.0)]))
    partials = [e for e in events if e["type"] == "partial"]
    segments = [e for e in events if e["type"] == "segment"]
    assert [s["start"] for s in segments] == [0.0, 8.67]
    assert len(partials) >= 3
    for p in partials:  # each previews a piece that later ends, and comes before it
        segment_at = next(i for i, e in enumerate(events) if e["type"] == "segment" and e["start"] == p["start"])
        assert events.index(p) < segment_at
    texts = [float(p["text"].split()[0]) for p in partials if p["start"] == 0.0]
    assert texts == sorted(texts) and texts[-1] < 8.3  # growing previews of the first piece, never past its end


def test_stream_language_is_given(client, replica):
    events = stream(client, speech(1.0), "?language=English")
    assert events[0]["language"] == "english" and replica.calls[-1][1] == "en"


def test_stream_skips_empty_text(client, replica):
    replica.transcribe = lambda audio, language, prompt="": ("", "en")
    assert stream(client, speech(1.0)) == [{"type": "done"}]


def test_stream_reports_failed_pieces_and_goes_on(client, replica):
    prompts = []

    def transcribe(audio, language, prompt=""):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise RuntimeError("CUDA error")
        return "Later.", "en"

    replica.transcribe = transcribe
    events = stream(client, np.concatenate([speech(5.5), silence(3.0), speech(1.0)]))
    assert [e["type"] for e in events] == ["error", "segment", "done"]
    assert events[0]["message"] == "CUDA error"
    assert prompts == ["", ""]


def test_stream_rejects_unknown_language(client, replica):
    with client.websocket_connect("/v1/audio/stream?language=klingon") as ws:
        assert ws.receive_json() == {"type": "error", "message": "Unknown language: klingon"}
    assert not replica.calls


def test_cancelled_transcription_keeps_the_gpu_until_it_finishes(replica):
    import threading

    release, finished = threading.Event(), []

    def transcribe(audio, language, prompt=""):
        release.wait(5)
        finished.append(True)
        return "Done.", "en"

    replica.transcribe = transcribe

    async def main():
        replica.gpu = asyncio.Lock()
        task = asyncio.create_task(replica.run(silence(1.0), None))
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        assert replica.gpu.locked() and not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished and not replica.gpu.locked()

    asyncio.run(main())


def test_stream_ignores_other_text_messages(client, replica):
    with client.websocket_connect("/v1/audio/stream") as ws:
        ws.send_text("not json")
        ws.send_text(json.dumps(["stop"]))
        ws.send_bytes(pcm(speech(1.0)) + b"\x00")  # the odd byte is dropped
        ws.send_text(json.dumps({"type": "stop"}))
        assert ws.receive_json()["type"] == "segment"
        assert ws.receive_json() == {"type": "done"}
    assert replica.calls[0][0].size == 16000


CALLER = {"X-Consumer": "svc", "X-Auth-Request-Email": "a@example.com", "Authorization": "Bearer secret"}


def caller_lines(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "ray.serve" and r.getMessage().endswith(" email=a@example.com")
    ]


def test_caller_is_logged(client, caplog):
    caplog.set_level(logging.INFO, "ray.serve")
    assert client.get("/v1/health", headers=CALLER).status_code == 200
    with client.websocket_connect("/v1/audio/stream", headers=CALLER) as ws:
        ws.send_text(json.dumps({"type": "stop"}))
        assert ws.receive_json() == {"type": "done"}
    health, stream = caller_lines(caplog)
    assert health.startswith("GET /v1/health 200 ") and health.endswith(" consumer=svc email=a@example.com")
    assert stream.startswith("WS /v1/audio/stream accepted ")
    assert "secret" not in caplog.text
