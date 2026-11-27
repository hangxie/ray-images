"""The uv environment the tests run in matches the voice image: its Dockerfile's version ARGs and the transformers
version its build asserts. Both images build on the same Ray."""

import re
from importlib.metadata import version
from pathlib import Path

import pytest

DOCKER = Path(__file__).parents[1] / "docker"
VOICE = (DOCKER / "voice" / "Dockerfile").read_text(encoding="utf-8")
VLLM = (DOCKER / "vllm" / "Dockerfile").read_text(encoding="utf-8")


def find(pattern, dockerfile=VOICE):
    match = re.search(pattern, dockerfile, re.MULTILINE)
    assert match, f"{pattern!r} not in the Dockerfile"
    return match[1]


def arg(name, dockerfile=VOICE):
    return find(rf"^ARG {name}=(\S+)$", dockerfile)


@pytest.mark.parametrize(
    ("package", "name"),
    [
        ("ray", "RAY_VERSION"),
        ("torch", "TORCH_VERSION"),
        ("torchaudio", "TORCHAUDIO_VERSION"),
        ("qwen-tts", "QWEN_TTS_VERSION"),
        ("silero-vad", "SILERO_VAD_VERSION"),
    ],
)
def test_voice_dockerfile_args(package, name):
    assert version(package).split("+")[0] == arg(name)


def test_voice_transformers():
    asserted = find(r"transformers\.__version__ == '([^']+)'")
    assert version("transformers") == asserted


def test_images_share_ray():
    assert arg("RAY_VERSION", VLLM) == arg("RAY_VERSION")
