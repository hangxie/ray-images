"""tts/check_requirements.py, the venv's stand-in for `pip check`, against this environment's installed packages."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import check_requirements

SCRIPT = Path(__file__).resolve().parent.parent / "tts" / "check_requirements.py"


def test_satisfied():
    assert check_requirements.problems(["qwen-tts==0.1.1", "scipy", "soundfile"]) == []


def test_extras_are_followed():
    assert check_requirements.problems(["ray[serve]"]) == []


def test_missing_package():
    assert check_requirements.problems(["no-such-package-xyz"]) == [
        "requested requires no-such-package-xyz, which is not installed"
    ]


def test_wrong_version():
    (problem,) = check_requirements.problems(["pytest<1"])
    assert problem.startswith("requested requires pytest<1, but ")


def test_transitive_version(monkeypatch):
    """A dependency's own requirement on a third package is checked against what is installed."""
    real = check_requirements.distribution

    fake = SimpleNamespace(version="1.0", requires=["numpy<1", 'pytest; extra == "test"'], metadata={"Name": "fake"})
    monkeypatch.setattr(check_requirements, "distribution", lambda name: fake if name == "fake" else real(name))
    (problem,) = check_requirements.problems(["fake"])
    assert problem.startswith("fake 1.0 requires numpy<1, but ")


def test_cli_exit_status():
    ok = subprocess.run([sys.executable, SCRIPT, "scipy"], capture_output=True, text=True)
    assert ok.returncode == 0
    assert ok.stdout.strip() == "Requirements satisfied: scipy"
    bad = subprocess.run([sys.executable, SCRIPT, "scipy<0.1"], capture_output=True, text=True)
    assert bad.returncode == 1
    assert "requires scipy<0.1" in bad.stdout
