"""Why is the HEADNOTE regex not matching real judgment text?"""

from __future__ import annotations

import re

from legalaid.ingestion.case_law import clean_judgment_text, load_year_metadata, _get


def main() -> None:
    rows = load_year_metadata(2023)
    row = rows[0]
    path = str(row["path"])
    raw = _get(f"data/pdf/year=2023/english/{path}_EN.pdf")

    import pymupdf

    doc = pymupdf.open(stream=raw, filetype="pdf")
    pages = [doc[i].get_text() for i in range(doc.page_count)]
    doc.close()
    text = clean_judgment_text("\n\n".join(pages))

    print(f"cleaned length: {len(text)}")
    for marker in ("HEADNOTE", "Headnote", "CASE LAW", "Issue for consideration", "HELD"):
        idx = text.find(marker)
        print(f"  {marker!r:28} at {idx}")

    i = text.find("HEADNOTE")
    if i >= 0:
        print("\n--- 1400 chars from HEADNOTE ---")
        print(text[i : i + 1400])

    from legalaid.ingestion.case_law import _HEADNOTE_RE, parse_judgment_structure

    m = _HEADNOTE_RE.search(text)
    print(f"\nmodule regex matched: {bool(m)}")
    if m:
        print(f"captured {len(m.group(1))} chars")

    headnote, issues, provisions = parse_judgment_structure(text)
    print(f"parse_judgment_structure -> headnote={headnote and len(headnote)} "
          f"issues={issues and len(issues)} provisions={provisions}")

    # What structural markers actually exist after the headnote?
    print("\n--- ALL-CAPS lines (candidate terminators) ---")
    for line in text.split("\n"):
        s = line.strip()
        if 3 <= len(s) <= 60 and s == s.upper() and any(c.isalpha() for c in s):
            print(f"  {s!r}")


if __name__ == "__main__":
    main()
