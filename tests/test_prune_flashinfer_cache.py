"""scripts/prune_flashinfer_cache.py on synthetic CUDA fat binaries (the real cache only exists in the image)."""

import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest

stub = ModuleType("flashinfer_jit_cache")
stub.__file__ = "/nonexistent/__init__.py"
sys.modules.setdefault("flashinfer_jit_cache", stub)
import prune_flashinfer_cache as prune  # noqa: E402

ENTRY_HEADER = 64


def fatbin(*entries):
    """A fat binary: magic, version, header size, payload size, then (kind, arch) entries with small payloads."""
    body = b""
    for kind, arch in entries:
        payload = b"\0" * 16
        header = struct.pack("<HHIQ", kind, 0, ENTRY_HEADER, len(payload)) + b"\0" * 12 + struct.pack("<I", arch)
        body += header.ljust(ENTRY_HEADER, b"\0") + payload
    return prune.MAGIC + struct.pack("<HHQ", 1, 16, len(body)) + body


def elf(*fatbins):
    return b"\x7fELF" + b"\0" * 100 + b"padding".join(fatbins) + b"\0" * 50


def test_entries_parses_every_fat_binary():
    blob = elf(fatbin((prune.SASS, 80), (prune.PTX, 80)), fatbin((prune.SASS, 90)))
    assert list(prune.entries(blob)) == [(prune.SASS, 80), (prune.PTX, 80), (prune.SASS, 90)]


def test_entries_stops_at_unknown_kinds():
    assert list(prune.entries(elf(fatbin((prune.SASS, 86), (7, 86), (prune.SASS, 90))))) == [(prune.SASS, 86)]


def test_entries_without_fat_binary():
    assert list(prune.entries(b"\x7fELF no device code")) == []


@pytest.mark.parametrize(
    ("found", "arch", "runs"),
    [
        ({(prune.SASS, 86)}, 86, True),
        ({(prune.SASS, 80)}, 86, True),  # same major, older minor
        ({(prune.SASS, 89)}, 86, False),  # newer minor
        ({(prune.SASS, 75)}, 86, False),  # SASS does not cross majors
        ({(prune.SASS, 90)}, 86, False),
        ({(prune.PTX, 75)}, 86, True),  # the driver JIT-compiles older PTX
        ({(prune.PTX, 90)}, 86, False),
        ({(prune.SASS, 90), (prune.PTX, 80)}, 86, True),
        ({(prune.SASS, 100)}, 100, True),
    ],
)
def test_runs_on(found, arch, runs):
    assert prune.runs_on(found, arch) is runs


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "jit_cache"
    modules = {
        "sm86": elf(fatbin((prune.SASS, 86))),
        "sm80": elf(fatbin((prune.SASS, 80))),
        "ptx75": elf(fatbin((prune.PTX, 75))),
        "sm90": elf(fatbin((prune.SASS, 90))),
        "sm100": elf(fatbin((prune.SASS, 100), (prune.SASS, 103))),
        "host_only": b"\x7fELF no device code",
    }
    for name, blob in modules.items():
        (root / name / "lib").mkdir(parents=True)
        (root / name / "lib" / f"{name}.so").write_bytes(blob)
        (root / name / "README").write_text("not a library")
    monkeypatch.setattr(prune, "flashinfer_jit_cache", SimpleNamespace(__file__=str(tmp_path / "__init__.py")))
    return root


def remaining(root):
    return sorted(p.name for p in root.iterdir())


def test_prune_keeps_runnable_and_unrecognized(cache, capsys):
    prune.main(["8.6"])
    assert remaining(cache) == ["host_only", "ptx75", "sm80", "sm86"]
    assert "kept 4 modules, removed 2 unusable on sm_86" in capsys.readouterr().out


def test_prune_several_archs(cache):
    prune.main(["8.6", "9.0"])
    assert remaining(cache) == ["host_only", "ptx75", "sm80", "sm86", "sm90"]


def test_prune_all_keeps_everything(cache, capsys):
    prune.main(["all"])
    assert len(remaining(cache)) == 6
    assert "keeping every module" in capsys.readouterr().out


def test_prune_refuses_to_empty_cache(tmp_path, monkeypatch):
    root = tmp_path / "jit_cache"
    (root / "sm90").mkdir(parents=True)
    (root / "sm90" / "a.so").write_bytes(elf(fatbin((prune.SASS, 90))))
    monkeypatch.setattr(prune, "flashinfer_jit_cache", SimpleNamespace(__file__=str(tmp_path / "__init__.py")))
    with pytest.raises(SystemExit, match="refusing to leave an empty JIT cache"):
        prune.main(["8.6"])
