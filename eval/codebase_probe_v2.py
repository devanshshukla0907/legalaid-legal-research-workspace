"""Exercise 6, re-measured after building the CODE collection differentiator (T5.3).

eval/codebase_probe.py established the baseline: 0/61 ground-truth files found across five
multi-file codebase questions, because COMBINED_LEGAL_RESEARCH never searches anything but law,
case law and user documents. That script is left untouched as the historical "before" record.

This script asks the same five questions, against the same static ground truth, but through
RetrievalMode.CODE_SEARCH - the new domain that indexes this repository's own source code by
function/class boundary (services/knowledge/ingest_codebase.py). It reuses
codebase_probe.static_ground_truth() and codebase_probe.import_graph() directly rather than
duplicating them, so the ground truth cannot silently drift between the before and after runs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from codebase_probe import QUESTIONS, import_graph, static_ground_truth

from legalaid.domain.contracts import RetrievalRequest
from legalaid.domain.enums import RetrievalMode
from legalaid.services.retrieval.service import search

OUT = Path(__file__).parent / "results" / "codebase_probe_v2.json"


async def probe() -> list[dict]:
    out: list[dict] = []
    for qid, question, kind in QUESTIONS:
        response = await search(
            RetrievalRequest(question=question, mode=RetrievalMode.CODE_SEARCH, top_k=6)
        )
        chunks = [
            {
                "collection": c.collection.value if hasattr(c.collection, "value") else str(c.collection),
                "file_path": (c.metadata or {}).get("file_path"),
                "symbol": c.heading,
                "similarity": round(c.similarity, 4),
                "text_head": (c.text or "")[:110],
            }
            for c in response.chunks
        ]
        out.append(
            {
                "id": qid,
                "question": question,
                "kind": kind,
                "top_similarity": chunks[0]["similarity"] if chunks else None,
                "files_returned": sorted({c["file_path"] for c in chunks if c["file_path"]}),
                "chunks": chunks,
            }
        )
        print(f"  {qid} top_sim={out[-1]['top_similarity']} files={out[-1]['files_returned']}")
    return out


async def main() -> None:
    print("PART A' - asking the SAME five codebase questions through CODE_SEARCH")
    probes = await probe()

    print("\nPART B - ground truth (reused from codebase_probe.py, unchanged)")
    truth = static_ground_truth()
    graph = import_graph()

    print("\nPART C' - overlap between CODE_SEARCH results and the same ground truth")
    coverage = []
    total_truth = 0
    total_hits = 0
    for record in probes:
        files = truth.get(record["id"], [])
        returned = set(record["files_returned"])
        hits = [f for f in files if f in returned]
        total_truth += len(files)
        total_hits += len(hits)
        coverage.append(
            {
                "id": record["id"],
                "ground_truth_files": len(files),
                "files_found_in_context": len(hits),
                "coverage": round(len(hits) / len(files), 4) if files else 0.0,
                "hits": hits,
            }
        )
        print(f"  {record['id']}: {len(hits)}/{len(files)} ground-truth files present in CODE_SEARCH context")

    print(f"\nTOTAL: {total_hits}/{total_truth} ground-truth files retrieved "
          f"({total_hits/total_truth*100:.1f}%), versus 0/{total_truth} before CODE_SEARCH existed")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "mode": "CODE_SEARCH",
                "probes": probes,
                "coverage": coverage,
                "total_ground_truth_files": total_truth,
                "total_hits": total_hits,
                "total_coverage": round(total_hits / total_truth, 4) if total_truth else 0.0,
                "baseline_reference": "eval/results/codebase_probe.json (0/61 before CODE_SEARCH existed)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
