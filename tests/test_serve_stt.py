"""serve_stt's OpenAI-compatible transcription API, on a replica with a fake Whisper.

The routes, multipart parsing, audio loading and validation are the real ones; transcribe() needs CUDA.
"""

from typing import Any, cast

import numpy as np
import pytest
from conftest import serve_client, tone, wav_bytes
from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE

import serve_stt


@pytest.fixture
def replica():
    replica = object.__new__(cast(Any, serve_stt.STT).func_or_class)
    replica.model_id = "openai/whisper-tiny"
    replica.device = "CPU / CUDA none"
    replica.languages, replica.language_codes = LANGUAGES, TO_LANGUAGE_CODE
    replica.calls = []

    def transcribe(audio, language):
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
    return client.post("/audio/transcriptions", files={"file": ("a.wav", audio, "audio/wav")}, data=fields)


def test_health_and_models(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "version": serve_stt.VERSION,
        "model": "openai/whisper-tiny",
        "device": "CPU / CUDA none",
    }
    assert client.get("/models").json() == {
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
    response = client.post("/audio/transcriptions", files={"other": ("a.wav", b"x")})
    assert response.status_code == 400
    assert response.json() == {"error": {"message": "Missing audio file field"}}


def test_not_multipart(client):
    response = client.post("/audio/transcriptions", json={"file": "x"})
    assert response.status_code == 400
    assert response.json() == {"error": {"message": "Request must be multipart/form-data"}}


def test_empty_body(client):
    response = client.post("/audio/transcriptions", content=b"")
    assert response.status_code == 400


def test_transcription_failure_is_500(client, replica):
    def fail(audio, language):
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
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["cache-control"] == "no-store"
    assert "fetch('audio/transcriptions'" in response.text
    assert "fetch('/" not in response.text  # absolute URLs would miss the /v1 route prefix
    assert f'<span id="version">{serve_stt.VERSION}</span>' in response.text
