"""Scan a range of India Code handles and report which acts they hold.

Direct handle probing is more reliable than the site's DSpace search, which returns an empty
result table for these queries. Results are cached to JSON so a scan is only paid for once.

Usage:
  python scripts/scan_handles.py 2140 2160        # scan an inclusive range
  python scripts/scan_handles.py --report         # show what has been discovered so far
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from legalaid.ingestion.india_code import (
    BASE_URL,
    IndiaCodeError,
    _get,
    parse_act_title,
    parse_section_index,
)

CACHE = Path("data/knowledge_sources/india_code/handle_scan.json")

WANTED = re.compile(
    r"specific relief|consumer protection|arbitration|sakshya|nagarik suraksha|"
    r"contract|nyaya sanhita|information technology|civil procedure|constitution",
    re.I,
)


def load() -> dict[str, dict]:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def save(data: dict[str, dict]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def probe(handle: str) -> dict:
    url = f"{BASE_URL}/handle/123456789/{handle}"
    try:
        page = _get(url, timeout=45, retries=2).decode("utf-8", "replace")
    except IndiaCodeError as exc:
        return {"handle": handle, "ok": False, "error": str(exc)[:120]}
    title = parse_act_title(page)
    refs = parse_section_index(page)
    return {
        "handle": handle,
        "ok": True,
        "title": title,
        "sections": len(refs),
        "actid": refs[0].actid if refs else None,
    }


def report(data: dict[str, dict]) -> None:
    usable = [
        v for v in data.values() if v.get("ok") and v.get("sections") and v.get("title")
    ]
    usable.sort(key=lambda v: -v["sections"])
    print(f"\n{len(usable)} handles with sections (of {len(data)} scanned)\n")
    for v in usable:
        flag = "  <== WANTED" if WANTED.search(v["title"] or "") else ""
        print(f"  {v['handle']:>7}  {v['sections']:>4}  {v['title'][:70]}{flag}")


def main() -> None:
    args = sys.argv[1:]
    data = load()

    if not args or args[0] == "--report":
        report(data)
        return

    start, end = int(args[0]), int(args[1] if len(args) > 1 else args[0])
    for n in range(start, end + 1):
        handle = str(n)
        if handle in data and data[handle].get("ok"):
            continue
        info = probe(handle)
        data[handle] = info
        if info.get("ok") and info.get("sections"):
            flag = "  <== WANTED" if WANTED.search(info.get("title") or "") else ""
            print(f"  {handle:>7}  {info['sections']:>4}  {(info.get('title') or '')[:70]}{flag}")
        save(data)
        time.sleep(3.5)

    report(data)


if __name__ == "__main__":
    main()
