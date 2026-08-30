"""Hit a rate-limited endpoint repeatedly and report the bucket state from health."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
KEY = (Path(__file__).resolve().parents[1] / "data" / ".dev-api-key").read_text().strip()


def post(path: str, body: dict) -> int:
    request = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(), method="POST"
    )
    request.add_header("Content-Type", "application/json")
    request.add_header("X-API-Key", KEY)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def health() -> dict:
    with urllib.request.urlopen(f"{BASE}/api/system/health", timeout=60) as response:
        return json.load(response)


def main() -> None:
    codes = [
        post("/api/retrieval/search", {"question": "Section 73 damages for breach"})
        for _ in range(16)
    ]
    print("codes:", " ".join(str(c) for c in codes))
    print("200s:", codes.count(200), " 429s:", codes.count(429))
    guardrails = health().get("guardrails", {})
    print("guardrails from health:")
    print(json.dumps(guardrails, indent=2))


if __name__ == "__main__":
    main()
