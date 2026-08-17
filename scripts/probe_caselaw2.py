"""Verify judgment PDFs download and extract to usable text."""

from __future__ import annotations

import io
import urllib.error
import urllib.request

import pyarrow.parquet as pq

BASE = "https://indian-supreme-court-judgments.s3.amazonaws.com"
UA = {"User-Agent": "LegalAId/0.1 academic-legal-research"}


def fetch(path: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(f"{BASE}/{path}", headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main() -> None:
    raw = fetch("metadata/parquet/year=2023/metadata.parquet")
    rows = pq.read_table(io.BytesIO(raw)).to_pylist()

    # `path` looks like 2023_16_872_887; PDFs live under data/pdf/year=YYYY/english/.
    sample = rows[0]
    print("sample path field:", sample["path"])

    # Discovered from a bucket listing: objects are named "{path}_EN.pdf".
    candidates = [
        f"data/pdf/year=2023/english/{sample['path']}_EN.pdf",
        f"data/pdf/year=2023/english/{sample['path']}.pdf",
    ]
    pdf_bytes = None
    for cand in candidates:
        try:
            pdf_bytes = fetch(cand)
            print(f"OK  {cand} -> {len(pdf_bytes):,} bytes, magic={pdf_bytes[:5]!r}")
            break
        except urllib.error.HTTPError as exc:
            print(f"FAIL {cand} -> HTTP {exc.code}")

    if pdf_bytes is None:
        # Fall back to listing the prefix to learn the real naming.
        print("\nlisting bucket prefix to discover naming...")
        listing = urllib.request.Request(
            "https://indian-supreme-court-judgments.s3.amazonaws.com"
            "?list-type=2&prefix=data/pdf/year=2023/english/&max-keys=5",
            headers=UA,
        )
        with urllib.request.urlopen(listing, timeout=90) as r:
            body = r.read().decode("utf-8", "replace")
        print(body[:1500])
        return

    # Extract text with the already-installed PDF library.
    import pymupdf

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    print(f"\npages: {doc.page_count}")
    text = doc[0].get_text()
    print(f"page 1 chars: {len(text)}")
    print("--- first 900 chars ---")
    print(text[:900])


if __name__ == "__main__":
    main()
