"""
CORS allowlist regression guard.

The Railway image runs `uvicorn app_genetics:app` (see Dockerfile CMD with
PYTHONPATH=/app/genetics-module), so app_genetics.ALLOWED_ORIGINS is the
authoritative production CORS configuration. app/main.py carries its own list
but is not the served application.

Preview/acceptance deployments are admitted by enumerating one exact origin at a
time. That is deliberate: a "*" or a *.vercel.app regex would admit every
deployment of every Vercel account, which is why this test asserts the absence
of both as well as the presence of the specific origins.

The list is read out of the source with ast rather than by importing
app_genetics, which would construct the whole application (R engine probe,
every router). The parsed list is then handed to a real CORSMiddleware so the
assertions exercise Starlette's actual origin matching, not a reimplementation
of it.

Run from inside genetics-module/:
    python -m pytest test_cors_allowed_origins.py -v
"""

import ast
import io
import os
import unittest

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_genetics.py")

PRODUCTION_ORIGINS = [
    "https://vivasensestat.com",
    "https://www.vivasensestat.com",
]
# Temporary acceptance-testing deployment. Delete this entry here and in
# app_genetics.py together when acceptance is signed off.
ACCEPTANCE_ORIGIN = "https://vivasense-stat-c1x9emuvk-fayeun-lawerences-projects.vercel.app"

REJECTED_ORIGINS = [
    "https://evil.example.com",
    # A different Vercel deployment: proves no *.vercel.app pattern crept in.
    "https://vivasense-stat-someotherbuild-fayeun-lawerences-projects.vercel.app",
    # Scheme and subdomain must both match exactly.
    "http://www.vivasensestat.com",
    "https://api.vivasensestat.com",
]


def _read_source() -> str:
    return io.open(MODULE_PATH, encoding="utf-8").read()


def _parse_allowed_origins() -> list:
    """Extract the ALLOWED_ORIGINS literal from app_genetics.py without importing it."""
    tree = ast.parse(_read_source(), filename=MODULE_PATH)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ALLOWED_ORIGINS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("ALLOWED_ORIGINS not found in app_genetics.py")


def _client(origins) -> TestClient:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/genetics/upload-preview")
    async def _preview():  # pragma: no cover - only the preflight is exercised
        return {"ok": True}

    return TestClient(app)


class CorsAllowlistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.origins = _parse_allowed_origins()
        cls.client = _client(cls.origins)

    def _preflight(self, origin: str):
        return self.client.options(
            "/genetics/upload-preview",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-vivasense-mode",
            },
        )

    def assertPreflightAllowed(self, origin: str):
        res = self._preflight(origin)
        self.assertEqual(res.status_code, 200, f"{origin}: preflight status {res.status_code}")
        self.assertEqual(
            res.headers.get("access-control-allow-origin"),
            origin,
            f"{origin}: preflight was not granted an Access-Control-Allow-Origin echo",
        )

    def assertPreflightRejected(self, origin: str):
        res = self._preflight(origin)
        self.assertIsNone(
            res.headers.get("access-control-allow-origin"),
            f"{origin}: unexpectedly allowed by CORS",
        )

    def test_production_domains_remain_allowed(self):
        for origin in PRODUCTION_ORIGINS:
            with self.subTest(origin=origin):
                self.assertIn(origin, self.origins)
                self.assertPreflightAllowed(origin)

    def test_acceptance_preview_origin_is_allowed(self):
        self.assertIn(ACCEPTANCE_ORIGIN, self.origins)
        self.assertPreflightAllowed(ACCEPTANCE_ORIGIN)

    def test_unrelated_origins_remain_rejected(self):
        for origin in REJECTED_ORIGINS:
            with self.subTest(origin=origin):
                self.assertNotIn(origin, self.origins)
                self.assertPreflightRejected(origin)

    def test_no_wildcard_and_no_origin_regex(self):
        """A wildcard or *.vercel.app regex would silently defeat every case above."""
        self.assertNotIn("*", self.origins)
        for origin in self.origins:
            self.assertNotIn("*", origin, f"pattern-like entry in allowlist: {origin}")
        self.assertNotIn(
            "allow_origin_regex",
            _read_source(),
            "CORSMiddleware must not be configured with allow_origin_regex",
        )


if __name__ == "__main__":
    unittest.main()
