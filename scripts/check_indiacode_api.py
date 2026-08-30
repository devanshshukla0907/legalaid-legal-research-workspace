"""Verify the DSpace-based India Code client against the live site.

Checks the things that would silently corrupt the corpus if wrong: that the CENTRAL copy is
selected rather than a state adoption, that section numbers and text come through, and that a
known provision reads correctly.
"""

from __future__ import annotations

import sys

from legalaid.ingestion import india_code_api as ic

WANTED = [
    "Indian Contract Act, 1872",
    "Specific Relief Act, 1963",
    "Limitation Act, 1963",
    "Negotiable Instruments Act, 1881",
    "Transfer of Property Act, 1882",
    "Sale of Goods Act, 1930",
    "Arbitration and Conciliation Act, 1996",
    "Consumer Protection Act, 2019",
    "Right to Information Act, 2005",
    "Indian Partnership Act, 1932",
]


def main() -> int:
    print("=== resolving acts (central copies only) ===")
    found: list[ic.ActRef] = []
    for title in WANTED:
        try:
            refs = ic.find_acts(title, limit=6)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {title:<42} {type(exc).__name__}: {str(exc)[:50]}")
            continue
        if not refs:
            print(f"  MISS  {title:<42} no central act found")
            continue
        ref = refs[0]
        found.append(ref)
        print(
            f"  OK    {title:<42} {ref.title[:40]:<42} year={ref.year} "
            f"commenced={ref.enforcement_date} repealed={ref.repealed}"
        )
        print(f"        act_id={ref.act_id}  handle={ref.handle}")

    if not found:
        print("\nNo acts resolved; the API shape may have changed again.")
        return 1

    print("\n=== fetching sections for the Contract Act ===")
    contract = next((r for r in found if "contract act" in r.title.lower()), found[0])
    sections = ic.fetch_sections(contract.act_id, delay=0.3, progress=False)
    print(f"  sections fetched: {len(sections)}")
    with_text = [s for s in sections if s.text.strip()]
    print(f"  with text       : {len(with_text)}")
    numbered = [s for s in sections if s.section_number]
    print(f"  with a number   : {len(numbered)}")

    print("\n=== spot check: section 73 must be compensation for breach ===")
    s73 = next((s for s in sections if s.section_number == "73"), None)
    if s73 is None:
        print("  FAIL  section 73 not found")
        return 1
    print(f"  heading: {s73.heading}")
    print(f"  url    : {s73.source_url}")
    print(f"  text   : {s73.text[:260]}")
    ok = "compensation" in s73.heading.lower() or "compensation" in s73.text.lower()
    print(f"  verdict: {'PASS' if ok else 'FAIL - unexpected content'}")

    print("\n=== a few sections, to confirm ordering and numbering ===")
    for section in sections[:5]:
        print(f"  order={section.order_no:<5} s.{section.section_number:<6} {section.heading[:56]}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
