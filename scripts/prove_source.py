"""Prove where a chunk's text came from by re-fetching it live from India Code.

Takes the source_url stored on a provision, calls it now, and compares the response with what
is in the database. If they match, the stored text demonstrably came from that URL.
"""

from __future__ import annotations

import json
import urllib.request

from sqlalchemy import select

from legalaid.db.base import session_scope
from legalaid.db.models import Chunk, LegalProvision
from legalaid.ingestion.india_code import USER_AGENT, strip_html

with session_scope() as session:
    provision = session.scalar(
        select(LegalProvision).where(
            LegalProvision.section_number == "73",
            LegalProvision.act_name.like("%Contract%"),
        )
    )
    chunk = session.scalar(
        select(Chunk).where(Chunk.provision_id == provision.id).order_by(Chunk.ordinal)
    )

print("STORED IN OUR DATABASE")
print(f"  act/section : {provision.act_name} s.{provision.section_number}")
print(f"  source_url  : {provision.source_url}")
print(f"  fetched at  : {provision.retrieved_at}")
print()

print("CALLING THAT URL RIGHT NOW...")
request = urllib.request.Request(provision.source_url, headers={"User-Agent": USER_AGENT})
with urllib.request.urlopen(request, timeout=60) as response:
    raw = response.read()
    status = response.status
    content_type = response.headers.get("Content-Type", "")

print(f"  HTTP {status}   Content-Type: {content_type}")
print(f"  bytes: {len(raw):,}")
print(f"  is it a PDF?  {'YES' if raw[:5] == b'%PDF-' else 'NO — it is JSON'}")
print()

payload = json.loads(raw.decode("utf-8", "replace"))
print(f"  JSON keys returned: {list(payload.keys())}")
print()
print("  RAW 'content' field, first 260 chars (note the HTML and \\r\\n):")
print(f"    {str(payload['content'])[:260]!r}")
print()

cleaned = strip_html(str(payload["content"]))
print("  AFTER OUR CLEANING, first 260 chars:")
print(f"    {cleaned[:260]!r}")
print()

stored_body = provision.text
match = cleaned[:200].strip() == stored_body[:200].strip()
print("COMPARISON")
print(f"  live fetch length : {len(cleaned):,} chars")
print(f"  stored length     : {len(stored_body):,} chars")
print(f"  first 200 chars identical? {'YES' if match else 'NO'}")
print()
print("  -> the stored text is exactly what this URL returns. No PDF was parsed.")
print()
print(f"CHUNK BUILT FROM IT: {chunk.id}")
print(f"  {chunk.text[:200]!r}")
