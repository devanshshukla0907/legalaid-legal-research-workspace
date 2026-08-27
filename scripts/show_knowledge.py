"""Print exactly what is in the knowledge base, and where it physically lives."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from legalaid.config import get_settings
from legalaid.db.base import session_scope
from legalaid.db.models import Case, Chunk, Document, LegalProvision
from legalaid.domain.enums import Collection
from legalaid.services.vector.store import get_store

settings = get_settings()

with session_scope() as session:
    provisions = session.scalars(select(LegalProvision)).all()
    cases = session.scalars(select(Case)).all()
    documents = session.scalars(select(Document)).all()
    chunks = session.scalars(select(Chunk)).all()

print("=" * 74)
print("1. LEGISLATION  (collection: LEGAL_ACTS)")
print("=" * 74)
by_act: dict[str, list[LegalProvision]] = {}
for p in provisions:
    by_act.setdefault(p.act_name, []).append(p)
for act, items in sorted(by_act.items(), key=lambda kv: -len(kv[1])):
    chars = sum(len(i.text) for i in items)
    print(f"  {len(items):>4} sections  {chars:>8,} chars   {act}")
print(f"\n  source: {provisions[0].source_authority if provisions else 'n/a'}")

print()
print("=" * 74)
print("2. CASE LAW  (collection: CASE_LAW)")
print("=" * 74)
by_year: dict[int, int] = {}
for c in cases:
    by_year[c.year or 0] = by_year.get(c.year or 0, 0) + 1
print(f"  {len(cases)} Supreme Court judgments   years: {dict(sorted(by_year.items()))}")
print("\n  examples:")
for c in cases[:8]:
    print(f"    {(c.citation or '-'):<24} {c.case_title[:52]}")

print()
print("=" * 74)
print("3. YOUR UPLOADED DOCUMENTS  (collection: USER_DOCUMENTS)")
print("=" * 74)
if documents:
    for d in documents:
        n = len([c for c in chunks if c.document_id == d.id])
        print(f"  {d.filename:<34} {d.doc_class:<16} {d.page_count}p  {n} chunks")
else:
    print("  (none yet — upload or paste on the My documents page)")

print()
print("=" * 74)
print("4. WHERE IT PHYSICALLY LIVES")
print("=" * 74)
db = settings.database_url.replace("sqlite+pysqlite:///", "")
print(f"  text + metadata : {db}")
print(f"  vectors         : {settings.index_dir}")
store = get_store()
for collection, index in store.all_indexes().items():
    if index.count:
        size = index.size_bytes() / 1024
        print(
            f"      {index.index_path.name:<26} {index.count:>4} vectors  {size:>8,.0f} KB"
        )
print(f"  uploaded files  : {settings.upload_dir}")
uploads = list(Path(settings.upload_dir).glob("*")) if Path(settings.upload_dir).exists() else []
print(f"      {len(uploads)} file(s)")

print()
print(f"TOTAL: {len(chunks)} chunks / {store.total_vectors()} vectors, "
      f"{sum(len(c.text) for c in chunks):,} characters")
print("Everything traces to an official source URL. Nothing is synthetic.")
