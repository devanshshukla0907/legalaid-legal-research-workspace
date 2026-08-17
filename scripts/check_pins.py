"""Strictly verify that every pinned dependency version exists on PyPI.

Used during setup to avoid committing a lockfile that cannot be installed.
"""

from __future__ import annotations

import json
import sys
import urllib.request

PINS = [
    ("fastapi", "0.120.4"),
    ("uvicorn", "0.38.0"),
    ("pydantic", "2.12.4"),
    ("pydantic-settings", "2.13.0"),
    ("sqlalchemy", "2.0.44"),
    ("httpx", "0.28.1"),
    ("python-multipart", "0.0.20"),
    ("pymupdf", "1.26.5"),
    ("python-docx", "1.2.0"),
    ("faiss-cpu", "1.13.2"),
    ("numpy", "2.4.2"),
    ("pyyaml", "6.0.3"),
    ("sse-starlette", "3.0.2"),
    ("structlog", "25.5.0"),
    ("pytest", "9.0.2"),
    ("pytest-asyncio", "1.3.0"),
    ("mypy", "1.19.0"),
    ("ruff", "0.15.4"),
]


def main() -> int:
    failures: list[str] = []
    for name, version in PINS:
        url = f"https://pypi.org/pypi/{name}/json"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.load(response)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: lookup failed ({exc})")
            continue
        releases = payload.get("releases", {})
        if version in releases and releases[version]:
            print(f"OK    {name}=={version}")
        elif version in releases:
            failures.append(f"{name}=={version} exists but has no files")
        else:
            latest = payload["info"]["version"]
            failures.append(f"{name}=={version} NOT FOUND (latest {latest})")

    for failure in failures:
        print(f"FAIL  {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
