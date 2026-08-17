"""Resolve India Code handles for the MVP corpus by probing candidate handles.

Prints the act title and section count for each handle so the manifest records verified
identifiers rather than guesses.

Usage: python scripts/find_handles.py 2187 1362 ...
"""

from __future__ import annotations

import sys

from legalaid.ingestion.india_code import (
    BASE_URL,
    IndiaCodeError,
    _get,
    parse_act_title,
    parse_pdf_url,
    parse_section_index,
)

import time

CANDIDATES = [
    # handle, what we expect it to be
    ("2187", "Indian Contract Act, 1872"),
    ("15240", "Bharatiya Nyaya Sanhita, 2023"),
    ("20062", "Bharatiya Nyaya Sanhita, 2023 (alt)"),
    ("2263", "Specific Relief Act, 1963"),
    ("13114", "Consumer Protection Act, 2019"),
    ("1999", "Information Technology Act, 2000"),
    ("2143", "Arbitration and Conciliation Act, 1996"),
    ("2191", "Code of Civil Procedure, 1908"),
]

# The site throttles bursts with 403s, so pace the probes.
PROBE_DELAY = 4.0


def probe(handle: str, expected: str = "") -> None:
    url = f"{BASE_URL}/handle/123456789/{handle}"
    try:
        page = _get(url, timeout=45, retries=2).decode("utf-8", "replace")
    except IndiaCodeError as exc:
        print(f"{handle:>7}  FAIL  {exc}")
        return
    title = parse_act_title(page) or "(no title)"
    refs = parse_section_index(page)
    pdf = parse_pdf_url(page)
    note = f"  [expected: {expected}]" if expected else ""
    print(f"{handle:>7}  {len(refs):>4} sections  {title}{note}")
    if refs:
        print(f"          actid={refs[0].actid}")
        print(f"          first={refs[0].section_number!r} {refs[0].heading[:60]!r}")
        print(f"          last ={refs[-1].section_number!r} {refs[-1].heading[:60]!r}")
        print(f"          pdf  ={pdf}")


def main() -> None:
    args = sys.argv[1:]
    pairs = [(h, "") for h in args] if args else CANDIDATES
    seen: set[str] = set()
    for handle, expected in pairs:
        if handle in seen:
            continue
        seen.add(handle)
        probe(handle, expected)
        time.sleep(PROBE_DELAY)


if __name__ == "__main__":
    main()
