"""Fetch a few named sections of an act and show the real statutory text.

Proves the ingestion route end-to-end without paying for a full act fetch.

Usage:
  python scripts/fetch_act_sample.py contract_act_1872 73 74 75
"""

from __future__ import annotations

import sys
import time

from legalaid.ingestion.corpus import get_act
from legalaid.ingestion.india_code import (
    BASE_URL,
    _get,
    fetch_section_text,
    parse_act_title,
    parse_section_index,
)


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "contract_act_1872"
    wanted = [s.upper() for s in (sys.argv[2:] or ["73", "74"])]

    spec = get_act(key)
    url = f"{BASE_URL}/handle/123456789/{spec.handle}"
    page = _get(url, timeout=60).decode("utf-8", "replace")

    title = parse_act_title(page)
    refs = parse_section_index(page)
    print(f"act        : {title}")
    print(f"handle     : {spec.handle}   manifest title: {spec.title}")
    print(f"sections   : {len(refs)}   expected: {spec.expected_sections}")
    if spec.expected_sections and len(refs) != spec.expected_sections:
        print("  WARNING: section count drifted from the manifest")

    selected = [r for r in refs if r.section_number.upper() in wanted]
    print(f"selected   : {[r.section_number for r in selected]}\n")

    for ref in selected:
        section = fetch_section_text(ref, timeout=60)
        print("=" * 78)
        print(f"Section {section.section_number} — {section.heading}")
        print(f"source: {section.source_url}")
        print(f"chars : {len(section.text)}")
        print("-" * 78)
        print(section.text[:1100])
        if section.footnotes:
            print("\n[amendment footnotes]")
            print(section.footnotes[:400])
        print()
        time.sleep(1.5)


if __name__ == "__main__":
    main()
