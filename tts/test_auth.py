"""CPU HTTP checks for TTS authentication; run with the image's system Python."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from serve_tts import TTS, read_api_key


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "token"
        self.path.write_text("test-token\n")
        replica = SimpleNamespace(api_key_file=str(self.path))
        app = FastAPI()

        @app.post("/api/tts")
        async def synthesize(request: Request):
            return await TTS.func_or_class.synthesize(replica, request)

        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def request(self, authorization=None):
        headers = {"Authorization": authorization} if authorization is not None else {}
        # A valid token reaches payload validation (400); a rejected token returns 401 first.
        return self.client.post("/api/tts", content=b"not json", headers=headers)

    def test_missing_wrong_and_malformed_tokens(self):
        for authorization in (None, "", "Bearer", "Bearer wrong", "Basic test-token", "Bearer test-token extra"):
            with self.subTest(authorization=authorization):
                response = self.request(authorization)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.headers["www-authenticate"], "Bearer")
                self.assertEqual(response.headers["cache-control"], "no-store")
                self.assertEqual(response.json(), {"error": "Missing or invalid API token"})

    def test_valid_token_reaches_payload_validation(self):
        for scheme in ("Bearer", "bearer", "BEARER"):
            with self.subTest(scheme=scheme):
                self.assertEqual(self.request(f"{scheme} test-token").status_code, 400)

    def test_rotation_is_visible_without_replica_restart(self):
        self.path.write_text("replacement-token")
        self.assertEqual(self.request("Bearer test-token").status_code, 401)
        self.assertEqual(self.request("Bearer replacement-token").status_code, 400)

    def test_absent_or_empty_secret_fails_closed(self):
        self.path.write_text(" \n")
        with self.assertRaises(RuntimeError):
            read_api_key(self.path)
        self.assertEqual(self.request("Bearer test-token").status_code, 500)
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            read_api_key(self.path)
        self.assertEqual(self.request("Bearer test-token").status_code, 500)



if __name__ == "__main__":
    unittest.main()
