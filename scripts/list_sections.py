"""List provisions in the KB so evaluation ground truth matches what actually exists."""

from __future__ import annotations

from sqlalchemy import select

from legalaid.db.base import session_scope
from legalaid.db.models import LegalProvision

WANTED = {
    "Indian Contract Act, 1872": [
        "2", "10", "23", "25", "39", "55", "56", "62", "63", "73", "74", "75",
        "124", "126", "148", "151", "152", "182",
    ],
    "Information Technology Act, 2000": ["3", "4", "10A", "65B"],
    "Bharatiya Sakshya Adhiniyam, 2023": ["61", "62", "63"],
    "Code of Civil Procedure, 1908": ["9", "11", "151"],
    "Bharatiya Nyaya Sanhita, 2023": ["316", "318"],
}

with session_scope() as session:
    rows = session.scalars(select(LegalProvision)).all()

have: dict[str, dict[str, str]] = {}
for r in rows:
    have.setdefault(r.act_name, {})[r.section_number] = r.section_heading or ""

for act, sections in WANTED.items():
    print(f"\n=== {act} ===")
    for s in sections:
        heading = have.get(act, {}).get(s)
        mark = "OK " if heading else "MISSING"
        print(f"  {mark} s.{s:<6} {(heading or '')[:66]}")

print("\n=== all Order/Rule style sections present in CPC ===")
for s, h in sorted(have.get("Code of Civil Procedure, 1908", {}).items()):
    print(f"  s.{s:<8} {h[:66]}")

print("\n=== BNS sections present ===")
for s, h in sorted(have.get("Bharatiya Nyaya Sanhita, 2023", {}).items()):
    print(f"  s.{s:<8} {h[:66]}")

print("\n=== BSA sections present ===")
for s, h in sorted(have.get("Bharatiya Sakshya Adhiniyam, 2023", {}).items()):
    print(f"  s.{s:<8} {h[:66]}")

print("\n=== IT Act sections present ===")
for s, h in sorted(have.get("Information Technology Act, 2000", {}).items()):
    print(f"  s.{s:<8} {h[:66]}")
