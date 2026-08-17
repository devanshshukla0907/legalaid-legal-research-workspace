"""Does raw_html in the metadata carry enough text to screen candidates without PDFs?"""

from __future__ import annotations

from legalaid.ingestion.case_curation import score_candidate
from legalaid.ingestion.case_law import load_year_metadata


def main() -> None:
    rows = load_year_metadata(2023)
    row = rows[0]
    raw = str(row.get("raw_html") or "")
    print(f"raw_html length: {len(raw)}")
    print(repr(raw[:900]))

    nc = str(row.get("nc_display") or "")
    print(f"\nnc_display: {nc[:200]!r}")

    # If raw_html is substantive, scoring on it avoids downloading every PDF.
    scored = [score_candidate(r, str(r.get("raw_html") or "")) for r in rows]
    useful = [s for s in scored if s.score > 0]
    print(f"\ncandidates scoring > 0 using raw_html: {len(useful)} of {len(rows)}")
    for s in sorted(useful, key=lambda c: -c.score)[:12]:
        print(f"  {s.score:>4}  {str(s.row.get('title'))[:70]}")
        print(f"        {s.reason}")


if __name__ == "__main__":
    main()
