"""Report tables and row counts in the LegalAId SQLite database."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

db = Path(sys.argv[1] if len(sys.argv) > 1 else "data/legalaid.db")
if not db.exists():
    print(f"missing: {db}")
    raise SystemExit(1)

conn = sqlite3.connect(str(db))
tables = [
    r[0]
    for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
]
print(f"{len(tables)} tables in {db}")
for t in tables:
    n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    print(f"  {n:>7}  {t}")
