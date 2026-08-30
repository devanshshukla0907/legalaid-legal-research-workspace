"""Probe India Code handles for the practitioner-core acts before adding them to the manifest.

Every handle in `ingestion/corpus.py` must be verified rather than guessed: a wrong handle
silently ingests the wrong statute. This prints the title and section count read back from the
live act page so the manifest records what was actually confirmed.

Usage:
    python scripts/probe_handles.py            # the candidate list below
    python scripts/probe_handles.py 2187 2263  # specific handles
"""

from __future__ import annotations

import sys
import time

from legalaid.ingestion.india_code import (
    BASE_URL,
    IndiaCodeError,
    _get,
    parse_act_title,
    parse_section_index,
)

# Candidates for the practitioner core. Several are guesses at this point - that is the whole
# reason for probing rather than committing them straight into the manifest.
CANDIDATES: list[tuple[str, str]] = [
    ("2263", "Specific Relief Act, 1963"),
    ("13114", "Consumer Protection Act, 2019"),
    ("2143", "Arbitration and Conciliation Act, 1996"),
    ("2183", "Limitation Act, 1963"),
    ("2338", "Transfer of Property Act, 1882"),
    ("2189", "Negotiable Instruments Act, 1881"),
    ("2079", "Companies Act, 2013"),
    ("2188", "Indian Partnership Act, 1932"),
    ("20064", "Bharatiya Nagarik Suraksha Sanhita, 2023"),
    ("13115", "Insolvency and Bankruptcy Code, 2016"),
    ("1362", "Right to Information Act, 2005"),
    ("2119", "Sale of Goods Act, 1930"),
]

PROBE_DELAY = 4.0


def probe(handle: str, expected: str = "") -> None:
    url = f"{BASE_URL}/handle/123456789/{handle}"
    try:
        page = _get(url, timeout=45, retries=2).decode("utf-8", "replace")
    except IndiaCodeError as exc:
        print(f"{handle:>7}  FAIL   {str(exc)[:70]}")
        return
    title = parse_act_title(page) or "(no title)"
    refs = parse_section_index(page)
    match = "OK " if expected and expected.split(",")[0].lower() in title.lower() else "?? "
    print(f"{handle:>7}  {match} {len(refs):>4} sections  {title}")
    if expected:
        print(f"{'':>7}       expected: {expected}")
    if refs:
        print(f"{'':>7}       actid={refs[0].actid}  first={refs[0].section_number!r}  "
              f"last={refs[-1].section_number!r}")


def main() -> None:
    args = sys.argv[1:]
    pairs = [(h, "") for h in args] if args else CANDIDATES
    for handle, expected in pairs:
        probe(handle, expected)
        time.sleep(PROBE_DELAY)


if __name__ == "__main__":
    main()
