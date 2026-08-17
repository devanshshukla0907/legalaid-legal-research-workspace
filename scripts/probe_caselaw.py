"""Probe the Supreme Court open dataset metadata to design a curated subset."""

from __future__ import annotations

import io
import urllib.request

import pyarrow.parquet as pq

BASE = "https://indian-supreme-court-judgments.s3.amazonaws.com"
UA = {"User-Agent": "LegalAId/0.1 academic-legal-research"}


def fetch(path: str) -> bytes:
    req = urllib.request.Request(f"{BASE}/{path}", headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> None:
    raw = fetch("metadata/parquet/year=2023/metadata.parquet")
    print(f"parquet bytes: {len(raw):,}")
    table = pq.read_table(io.BytesIO(raw))
    print(f"rows: {table.num_rows}  columns: {table.num_columns}")
    print("schema:")
    for name, dtype in zip(table.schema.names, table.schema.types, strict=True):
        print(f"  {name:<22} {dtype}")

    df = table.to_pylist()
    print("\n--- first 3 records (truncated) ---")
    for row in df[:3]:
        for key in ("title", "petitioner", "respondent", "citation", "decision_date",
                    "disposal_nature", "judge", "path", "description"):
            value = str(row.get(key) or "")
            print(f"  {key:<18} {value[:110]}")
        print()

    # How many look like contract / commercial disputes we can align with the demo matter?
    import re

    pattern = re.compile(
        r"contract|agreement|breach|compensation|damages|arbitrat|consumer|"
        r"specific performance|recovery|suit",
        re.I,
    )
    hits = [
        r for r in df
        if pattern.search(f"{r.get('title') or ''} {r.get('description') or ''}")
    ]
    print(f"contract/commercial-looking judgments in 2023: {len(hits)} of {len(df)}")
    for row in hits[:8]:
        print(f"  - {str(row.get('title'))[:95]}")
        print(f"    path={row.get('path')}")

    natures = {}
    for row in df:
        key = str(row.get("disposal_nature") or "-")
        natures[key] = natures.get(key, 0) + 1
    print("\ndisposal_nature counts:")
    for key, count in sorted(natures.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {count:>5}  {key}")


if __name__ == "__main__":
    main()
