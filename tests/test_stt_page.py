"""stt.html's script, as `GET /` serves it, run in node against fake browser APIs (tests/stt_page.js)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from test_serve_stt import client, replica  # noqa: F401  # fixtures

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_page_script(client, tmp_path):  # noqa: F811
    page = client.get("/").text
    found = re.search(r"<script>(.*)</script>", page, re.S)
    assert found
    script = tmp_path / "stt.js"
    script.write_text(found.group(1))
    harness = Path(__file__).with_name("stt_page.js")
    result = subprocess.run([str(NODE), str(harness), str(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
