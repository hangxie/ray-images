"""What the head needs from the Serve apps: it imports them with the system Python (no qwen-tts, transformers 5.x),
reads settings there and passes them through bind(), and pickles the FastAPI app."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from ray import cloudpickle

import serve_chat
import serve_stt
import serve_tts

APP = Path(__file__).resolve().parent.parent / "tts" / "app"
REPLICA_ONLY = ["torch", "torchaudio", "transformers", "qwen_tts", "soundfile", "accelerate", "librosa", "gradio"]
# The head's environment as far as the apps can tell: a finder that refuses the replica-only packages. (Entries of None
# in sys.modules would not do: scipy looks torch up there and expects a module.)
HEAD = """
import sys


class Head:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in {blocked!r}:
            raise ImportError(f"{{name}} is not installed on the head")


sys.meta_path.insert(0, Head())
import serve_chat, serve_stt, serve_tts

print(serve_tts.app, serve_stt.app, serve_chat.app)
"""


def test_apps_import_without_replica_packages():
    """Module-level imports stay within what the system environment has (AGENTS.md, Serve app imports)."""
    code = HEAD.format(blocked=REPLICA_ONLY)
    result = subprocess.run([sys.executable, "-c", code], cwd=APP, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", [serve_tts, serve_stt, serve_chat])
def test_app_pickles(module):
    """Serve pickles the ingress app and the replica class, so nothing unpicklable (e.g. threading.local) may live at
    module level. The ingress wrapper is a dynamic subclass, so it goes by value with the globals its methods use."""
    restored = cloudpickle.loads(cloudpickle.dumps(module.api))
    assert {r.path for r in restored.routes} == {r.path for r in module.api.routes}
    cloudpickle.loads(cloudpickle.dumps(module.app._bound_deployment.func_or_class))


def test_tts_settings(monkeypatch):
    for name, value in {
        "TTS_MODEL": "Qwen/other",
        "TTS_MODEL_DIR": "/models/qwen",
        "TTS_VOICES_DIR": "/voices",
        "TTS_CUDA_GRAPHS": "0",
        "TTS_KV_CACHE_TOKENS": "2048",
        "TTS_STREAM_CONTEXT_FRAMES": "100",
    }.items():
        monkeypatch.setenv(name, value)
    assert serve_tts.settings() == {
        "model": "Qwen/other",
        "model_dir": "/models/qwen",
        "voices_dir": "/voices",
        "cuda_graphs": False,
        "kv_cache_tokens": 2048,
        "stream_context_frames": 100,
    }


def test_tts_settings_defaults(monkeypatch):
    for name in [n for n in os.environ if n.startswith("TTS_")]:
        monkeypatch.delenv(name)
    assert serve_tts.settings() == {
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "model_dir": None,
        "voices_dir": str(APP / "voices"),
        "cuda_graphs": True,
        "kv_cache_tokens": 3072,
        "stream_context_frames": 300,
    }


def test_stt_settings(monkeypatch):
    monkeypatch.delenv("STT_MODEL", raising=False)
    monkeypatch.setenv("STT_MODEL_DIR", "/models/whisper")
    assert serve_stt.settings() == {"model": "openai/whisper-large-v3-turbo", "model_dir": "/models/whisper"}


@pytest.mark.parametrize(("module", "variable"), [(serve_tts, "TTS_PYTHON"), (serve_stt, "STT_PYTHON")])
def test_replica_runtime_env(monkeypatch, module, variable):
    """A deployment's runtime_env replaces the application's, so HF_* variables are carried over explicitly."""
    for name in [n for n in os.environ if n.startswith("HF_")]:
        monkeypatch.delenv(name)
    monkeypatch.delenv(variable, raising=False)
    assert module.replica_runtime_env() == {"py_executable": "/home/ray/tts-venv/bin/python"}

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", "/cache")
    monkeypatch.setenv("TRANSFORMERS_VERBOSITY", "info")
    monkeypatch.setenv(variable, "/venv/bin/python")
    assert module.replica_runtime_env() == {
        "py_executable": "/venv/bin/python",
        "env_vars": {"HF_HUB_OFFLINE": "1", "HF_HOME": "/cache"},
    }


def test_tts_deployment_options():
    deployment = cast(Any, serve_tts.TTS)
    assert deployment.ray_actor_options["num_gpus"] == 1
    assert deployment.ray_actor_options["resources"] == {"tts": 1}
    assert "py_executable" in deployment.ray_actor_options["runtime_env"]
    assert deployment.max_ongoing_requests == 1  # one synthesis per replica; the job slot relies on it


def test_stt_deployment_options():
    deployment = cast(Any, serve_stt.STT)
    assert deployment.ray_actor_options["num_gpus"] == 1
    assert "py_executable" in deployment.ray_actor_options["runtime_env"]
    assert deployment.max_ongoing_requests == 1


def test_chat_deployment_options():
    """Chat holds no model: no GPU, and the system Python (no py_executable), like the head."""
    deployment = cast(Any, serve_chat.Chat)
    assert not deployment.ray_actor_options.get("num_gpus")
    assert "runtime_env" not in deployment.ray_actor_options
    assert deployment.max_ongoing_requests == 16  # each open turn is one ongoing request


def test_apps_bind_head_settings():
    """Settings reach replicas only as bind() arguments."""
    assert serve_tts.app._bound_deployment.init_args == (serve_tts.settings(),)
    assert serve_stt.app._bound_deployment.init_args == (serve_stt.settings(),)
    assert serve_chat.app._bound_deployment.init_args == (serve_chat.settings(),)
