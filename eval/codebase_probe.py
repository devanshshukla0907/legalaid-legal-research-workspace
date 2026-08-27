"""Exercise 6 - can the existing LLM + RAG system answer multi-file codebase questions?

The honest hypothesis before running this: no, and for a structural reason rather than a tuning
one. The knowledge base contains Indian statute and case law. There is no code in it, no import
graph, no call graph, and the graph retriever's edge types are legal
(contains / cross_references / cites_section / considers_act) with no notion of "calls" or
"imports".

This script tests that hypothesis instead of asserting it, in three parts:

  A. Ask the retriever five genuine multi-file questions about this repository and record what
     comes back. If the corpus cannot answer them, the retrieved chunks will be statute.

  B. Establish the ground truth by static analysis of the repo, so there is something concrete
     to compare against - which files really do define the agents, what really imports the
     retrieval service, and so on.

  C. Quantify the gap: how many of the ground-truth files appear anywhere in the retrieved
     context. Expected answer is zero, which is the finding.
"""

from __future__ import annotations

import ast
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from legalaid.domain.contracts import RetrievalRequest
from legalaid.domain.enums import RetrievalMode
from legalaid.services.retrieval.service import search

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "backend" / "src" / "legalaid"
OUT = Path(__file__).parent / "results" / "codebase_probe.json"

QUESTIONS = [
    (
        "Q1",
        "Which files are involved in agent context isolation?",
        "multi_file_location",
    ),
    (
        "Q2",
        "Which modules import the retrieval service?",
        "dependency_graph",
    ),
    (
        "Q3",
        "What happens after a user submits a research request?",
        "control_flow",
    ),
    (
        "Q4",
        "Which components would be affected if the chunking strategy changed?",
        "impact_analysis",
    ),
    (
        "Q5",
        "Which tests cover the hybrid retrieval fusion logic?",
        "test_mapping",
    ),
]


# --------------------------------------------------------------------------- part B
def static_ground_truth() -> dict[str, list[str]]:
    """Answer the same five questions by parsing the repo, to have a real baseline."""
    truth: dict[str, list[str]] = {}

    def rel(path: Path) -> str:
        return str(path.relative_to(REPO)).replace("\\", "/")

    # Q1 - context isolation: files defining or using AgentContext.
    q1: list[str] = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "AgentContext" in text:
            q1.append(rel(path))
    truth["Q1"] = sorted(q1)

    # Q2 - modules importing the retrieval service.
    q2: list[str] = []
    for path in REPO.rglob("*.py"):
        if ".venv" in str(path) or "node_modules" in str(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "services.retrieval" in text or "from legalaid.services.retrieval" in text:
            q2.append(rel(path))
    truth["Q2"] = sorted(q2)

    # Q3 - control flow from the research endpoint through the orchestrator.
    q3 = [
        rel(SRC / "services" / "gateway" / "app.py"),
        rel(SRC / "agents" / "orchestrator.py"),
        rel(SRC / "agents" / "intake_router.py"),
        rel(SRC / "services" / "retrieval" / "service.py"),
        rel(SRC / "agents" / "legal_research.py"),
        rel(SRC / "agents" / "similar_cases.py"),
        rel(SRC / "agents" / "synthesis.py"),
        rel(SRC / "core" / "tracing.py"),
    ]
    truth["Q3"] = [p for p in q3 if (REPO / p).exists()]

    # Q4 - anything referencing chunking.
    q4: list[str] = []
    for path in REPO.rglob("*.py"):
        if ".venv" in str(path) or "node_modules" in str(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "chunking" in text or "chunk_strategy" in text:
            q4.append(rel(path))
    truth["Q4"] = sorted(q4)

    # Q5 - tests mentioning fusion / hybrid retrieval.
    q5: list[str] = []
    tests = REPO / "backend" / "tests"
    if tests.exists():
        for path in tests.rglob("test_*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if "fusion" in text or "rrf" in text.lower() or "hybrid" in text:
                q5.append(rel(path))
    truth["Q5"] = sorted(q5)

    return truth


def import_graph() -> dict[str, list[str]]:
    """Internal import edges, i.e. the structure a code-aware retriever would need."""
    edges: dict[str, list[str]] = defaultdict(list)
    for path in SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        module = str(path.relative_to(SRC)).replace("\\", "/")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("legalaid"):
                    edges[module].append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("legalaid"):
                        edges[module].append(alias.name)
    return {k: sorted(set(v)) for k, v in sorted(edges.items())}


# --------------------------------------------------------------------------- part A
async def probe() -> list[dict]:
    out: list[dict] = []
    for qid, question, kind in QUESTIONS:
        response = await search(
            RetrievalRequest(
                question=question, mode=RetrievalMode.COMBINED_LEGAL_RESEARCH, top_k=6
            )
        )
        chunks = [
            {
                "collection": c.collection.value
                if hasattr(c.collection, "value")
                else str(c.collection),
                "source_title": c.source_title,
                "section": c.section,
                "similarity": round(c.similarity, 4),
                "text_head": (c.text or "")[:110],
            }
            for c in response.chunks
        ]
        collections = sorted({c["collection"] for c in chunks})
        out.append(
            {
                "id": qid,
                "question": question,
                "kind": kind,
                "top_similarity": chunks[0]["similarity"] if chunks else None,
                "collections_returned": collections,
                "any_code_returned": any(c != "USER_DOCUMENTS" and "LEGAL" not in c
                                         and c != "CASE_LAW" for c in collections),
                "chunks": chunks,
            }
        )
        print(
            f"  {qid} top_sim={out[-1]['top_similarity']} "
            f"collections={collections}"
        )
    return out


async def main() -> None:
    print("PART A - asking the existing RAG system five codebase questions")
    probes = await probe()

    print("\nPART B - ground truth by static analysis")
    truth = static_ground_truth()
    for qid, files in truth.items():
        print(f"  {qid}: {len(files)} file(s)")
        for f in files[:4]:
            print(f"       {f}")
        if len(files) > 4:
            print(f"       ... +{len(files) - 4} more")

    graph = import_graph()
    print(f"\n  internal import edges parsed: {sum(len(v) for v in graph.values())} "
          f"across {len(graph)} modules")

    print("\nPART C - overlap between what RAG returned and the ground truth")
    coverage = []
    for record in probes:
        files = truth.get(record["id"], [])
        blob = json.dumps(record["chunks"]).lower()
        hits = [f for f in files if Path(f).name.lower() in blob]
        coverage.append(
            {
                "id": record["id"],
                "ground_truth_files": len(files),
                "files_found_in_context": len(hits),
                "coverage": round(len(hits) / len(files), 4) if files else 0.0,
            }
        )
        print(
            f"  {record['id']}: {len(hits)}/{len(files)} ground-truth files present "
            f"in retrieved context"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "probes": probes,
                "ground_truth": truth,
                "import_graph_modules": len(graph),
                "import_graph_edges": sum(len(v) for v in graph.values()),
                "coverage": coverage,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
