"""Run duplicate detection and citation extraction over the stored corpus.

Produces the reconciliation evidence the data-acquisition research called for: a duplicate rate
rather than an assumption of cleanliness, and a citation graph built from judgment text rather
than asserted to exist.

Writes eval/results/corpus_analysis.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from legalaid.db.base import session_scope
from legalaid.db.models import Case, Chunk, LegalProvision
from legalaid.ingestion.citations import build_edges, extract_section_references
from legalaid.ingestion.dedup import DedupItem, find_duplicates

OUT = Path(__file__).resolve().parents[1] / "eval" / "results" / "corpus_analysis.json"


def analyse_provisions() -> dict:
    with session_scope() as session:
        rows = session.scalars(select(LegalProvision)).all()
        items = [
            DedupItem(
                key=row.id,
                text=row.text or "",
                act=row.act_name or "",
                section=row.section_number or "",
                label=f"{row.act_name} s.{row.section_number}",
            )
            for row in rows
        ]
        labels = {row.id: f"{row.act_name} s.{row.section_number}" for row in rows}

    report = find_duplicates(items)
    payload = report.to_dict()
    # Replace opaque ids with readable labels in the reported groups.
    for bucket in ("exact_groups", "near_groups", "structural_groups"):
        for group in payload[bucket]:
            group["members"] = [labels.get(m, m) for m in group["members"]]
    return payload


def analyse_chunks() -> dict:
    with session_scope() as session:
        rows = session.scalars(select(Chunk)).all()
        items = [
            DedupItem(key=row.id, text=row.text or "", act=row.source_title or "",
                      section=row.section or "")
            for row in rows
        ]
    report = find_duplicates(items)
    return {
        "total": report.total,
        "exact_duplicates": report.exact_duplicates,
        "near_duplicates": report.near_duplicates,
        "duplicate_rate": report.duplicate_rate,
    }


def analyse_citations() -> dict:
    with session_scope() as session:
        rows = session.scalars(select(Case)).all()
        cases: list[tuple[str, str, list[str]]] = []
        titles: dict[str, str] = {}
        for row in rows:
            text = "\n".join(
                part
                for part in (
                    row.facts_summary, row.issues, row.arguments,
                    row.reasoning, row.decision, row.ratio,
                )
                if part
            )
            known = [c for c in (row.citation, row.neutral_citation, row.case_ref) if c]
            cases.append((row.id, text, known))
            titles[row.id] = row.case_title

    edges, stats = build_edges(cases)
    resolved = [e for e in edges if e.resolved]

    # Statutory references per judgment: the other half of the graph, linking a judgment to the
    # provisions it interprets.
    section_links = 0
    for _case_id, text, _known in cases:
        section_links += len(extract_section_references(text))

    return {
        "cases_analysed": len(cases),
        "citations_found": stats.get("citations_found", 0),
        "resolved_to_corpus": stats.get("resolved", 0),
        "unresolved": stats.get("unresolved", 0),
        "section_references_found": section_links,
        "treatment_breakdown": {
            key.removeprefix("treatment_"): value
            for key, value in sorted(stats.items())
            if key.startswith("treatment_")
        },
        "resolved_edges": [
            {
                "from": titles.get(edge.source_case_id, edge.source_case_id)[:70],
                "to": titles.get(edge.target_case_id or "", edge.target_citation)[:70],
                "treatment": edge.treatment.value,
                "evidence": edge.evidence[:150],
            }
            for edge in resolved[:25]
        ],
        "sample_unresolved": [
            {"citation": e.target_citation, "treatment": e.treatment.value}
            for e in edges
            if not e.resolved
        ][:20],
    }


def main() -> None:
    print("=== provisions: duplicate detection ===")
    provisions = analyse_provisions()
    print(f"  total              : {provisions['total']}")
    print(f"  exact duplicates   : {provisions['exact_duplicates']}")
    print(f"  near duplicates    : {provisions['near_duplicates']}")
    print(f"  structural dupes   : {provisions['structural_duplicates']}")
    print(f"  duplicate rate     : {provisions['duplicate_rate'] * 100:.2f}%")
    for group in provisions["near_groups"][:6]:
        print(f"    near: {group['members'][:3]}  ({group['detail']})")
    for group in provisions["structural_groups"][:6]:
        print(f"    structural: {group['members'][:3]}")

    print("\n=== chunks: duplicate detection ===")
    chunks = analyse_chunks()
    print(f"  total              : {chunks['total']}")
    print(f"  exact duplicates   : {chunks['exact_duplicates']}")
    print(f"  near duplicates    : {chunks['near_duplicates']}")
    print(f"  duplicate rate     : {chunks['duplicate_rate'] * 100:.2f}%")

    print("\n=== case law: citation graph ===")
    citations = analyse_citations()
    print(f"  cases analysed     : {citations['cases_analysed']}")
    print(f"  citations found    : {citations['citations_found']}")
    print(f"  resolved in corpus : {citations['resolved_to_corpus']}")
    print(f"  unresolved         : {citations['unresolved']}")
    print(f"  section references : {citations['section_references_found']}")
    print(f"  treatment          : {citations['treatment_breakdown']}")
    for edge in citations["resolved_edges"][:8]:
        print(f"    {edge['treatment']:<14} {edge['from'][:38]} -> {edge['to'][:38]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"provisions": provisions, "chunks": chunks, "citations": citations}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
