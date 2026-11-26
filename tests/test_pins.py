"""The uv environment the tests run in matches the image: the Dockerfile's version ARGs, the transformers version its
TTS venv check asserts, and the torch/torchaudio/fastapi versions vLLM pins in the system environment.

vLLM's pins come from PyPI, so that test is marked `network` and runs only in `make pins`."""

import json
import re
import urllib.request
from importlib.metadata import version
from pathlib import Path

import pytest
from packaging.requirements import Requirement

DOCKERFILE = (Path(__file__).parents[1] / "Dockerfile").read_text(encoding="utf-8")


def find(pattern):
    match = re.search(pattern, DOCKERFILE, re.MULTILINE)
    assert match, f"{pattern!r} not in the Dockerfile"
    return match[1]


def arg(name):
    return find(rf"^ARG {name}=(\S+)$")


@pytest.mark.parametrize(
    ("package", "name"),
    [("ray", "RAY_VERSION"), ("qwen-tts", "QWEN_TTS_VERSION"), ("silero-vad", "SILERO_VAD_VERSION")],
)
def test_dockerfile_args(package, name):
    assert version(package) == arg(name)


def test_tts_venv_transformers():
    asserted = find(r"transformers\.__version__ == '([^']+)'")
    assert version("transformers") == asserted


@pytest.mark.network
def test_vllm_pins():
    # Fails rather than skips when PyPI can't be read: a 404 is a VLLM_VERSION the image build can't install either
    url = f"https://pypi.org/pypi/vllm/{arg('VLLM_VERSION')}/json"
    with urllib.request.urlopen(url, timeout=10) as response:
        requires = json.load(response)["info"]["requires_dist"]
    pins = {r.name: r for r in map(Requirement, requires) if r.marker is None or r.marker.evaluate()}
    for package in ("torch", "torchaudio", "fastapi"):
        assert pins[package].specifier.contains(version(package), prereleases=True), (package, str(pins[package]))
