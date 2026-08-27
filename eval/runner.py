"""Run the evaluation dataset against several models under identical conditions.

Design decisions that matter for validity:

1. Retrieval runs ONCE per question and the identical context is replayed to every model.
   Retrieval is model-independent (it depends only on the embedding model and the corpus), so
   re-running it per model would add noise and trebles the cost for no information. It also
   guarantees the "same knowledge base, same conditions" requirement literally rather than
   approximately.

2. One agent (legal_research) per question rather than the full seven-agent pipeline. The full
   pipeline is 3-7 minutes per question; 27 questions x 3 models would be ~9 hours. The single
   agent still exercises retrieval, prompt rendering, schema-constrained decoding and the
   grounding audit, which is where model differences actually show up.

3. Structured output is used for scoring rather than prose. The agent returns cited sections
   with chunk ids, so correctness and hallucination can be computed by set membership instead
   of a subjective read.

Usage:
    python eval/runner.py                       # all models, all tasks
    python eval/runner.py --models gemma2:2b    # subset
    python eval/runner.py --limit 5             # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from legalaid.agents.legal_research import LegalResearchAgent, prepare
from legalaid.domain.contracts import RetrievalRequest, RetrievedChunk
from legalaid.domain.enums import RetrievalMode
from legalaid.services.retrieval.service import search

from dataset import TASKS, EvalTask

# Deliberately the three lightest locally available models, from three different families.
# codellama:7b-instruct (the Week 3 baseline) is 2x the size and ~3x slower; it is included
# only when --with-baseline is passed so the light comparison stays fast.
DEFAULT_MODELS = [
    "gemma2:2b",              # 1.6 GB, Google
    "qwen2.5:3b-instruct",    # 1.9 GB, Alibaba
    "llama3.2:3b",            # 2.0 GB, Meta
]
BASELINE_MODEL = "codellama:7b-instruct"  # 3.8 GB, Week 3 default

OUT_DIR = Path(__file__).parent / "results"


@dataclass
class RetrievalRecord:
    """What retrieval produced for one question - shared by every model."""

    task_id: str
    query_used: str
    chunk_ids: list[str] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    similarities: list[float] = field(default_factory=list)
    context_tokens: int = 0
    retrieval_ms: float = 0.0
    # Did retrieval surface the sections the task expects?
    expected_found: list[str] = field(default_factory=list)
    expected_missed: list[str] = field(default_factory=list)
    recall: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0


@dataclass
class ModelRecord:
    """What one model did with that context."""

    task_id: str
    model: str
    ok: bool = False
    error: str | None = None
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    parse_ok: bool = False
    attempts: int = 0
    cited_sections: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    evidence_sufficient: bool | None = None
    raw_output: dict[str, Any] | None = None
    raw_text_len: int = 0


def _section_key(act: str, number: str) -> str:
    """Normalise so 'Section 73' / 's.73' / '73' in 'Indian Contract Act, 1872' all match."""
    act_token = "".join(ch for ch in act.lower() if ch.isalnum())[:14]
    num = str(number).lower().replace("section", "").replace("s.", "").strip()
    return f"{act_token}:{num}"


def _expected_keys(task: EvalTask) -> list[str]:
    return [_section_key(act, num) for act, num in task.expected_sections]


async def run_retrieval(task: EvalTask) -> tuple[RetrievalRecord, list[RetrievedChunk]]:
    """Returns the scored record plus the raw chunks, which are replayed to every model."""
    started = time.perf_counter()
    response = await search(
        RetrievalRequest(
            question=task.question,
            mode=RetrievalMode.COMBINED_LEGAL_RESEARCH,
            top_k=8,
        )
    )
    elapsed = (time.perf_counter() - started) * 1000

    sections: list[dict[str, Any]] = []
    retrieved_keys: list[str] = []
    for chunk in response.chunks:
        meta = chunk.metadata or {}
        act = chunk.source_title or ""
        section = chunk.section or meta.get("section_number") or ""
        sections.append(
            {
                "chunk_id": chunk.chunk_id,
                "collection": chunk.collection.value
                if hasattr(chunk.collection, "value")
                else str(chunk.collection),
                "act": act,
                "section": section,
                "similarity": round(chunk.similarity, 4),
                "retrievers": meta.get("retrievers", []),
            }
        )
        if section:
            retrieved_keys.append(_section_key(act, str(section)))

    expected = _expected_keys(task)
    found = [k for k in expected if k in retrieved_keys]
    missed = [k for k in expected if k not in retrieved_keys]

    # Reciprocal rank of the first expected section, 0 if none retrieved.
    mrr = 0.0
    for rank, key in enumerate(retrieved_keys, start=1):
        if key in expected:
            mrr = 1.0 / rank
            break

    record = RetrievalRecord(
        task_id=task.id,
        query_used=task.question,
        chunk_ids=[c["chunk_id"] for c in sections],
        sections=sections,
        similarities=[c["similarity"] for c in sections],
        context_tokens=response.context_tokens,
        retrieval_ms=round(elapsed, 1),
        expected_found=found,
        expected_missed=missed,
        recall=round(len(found) / len(expected), 4) if expected else 1.0,
        precision_at_k=round(
            sum(1 for k in retrieved_keys if k in expected) / max(1, len(retrieved_keys)), 4
        )
        if expected
        else 0.0,
        mrr=round(mrr, 4),
    )
    return record, list(response.chunks)


async def run_model(
    task: EvalTask, model: str, chunks: list[RetrievedChunk]
) -> ModelRecord:
    """Run one model on one task, replaying the identical retrieved context."""
    record = ModelRecord(task_id=task.id, model=model)
    agent = LegalResearchAgent(model=model)
    started = time.perf_counter()
    try:
        context = prepare(agent, question=task.question, chunks=chunks)
        result = await agent.run(trace_id=None, context=context, max_attempts=2)
        record.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        record.ok = True
        record.parse_ok = result.parse_ok
        record.attempts = result.attempts
        record.input_tokens = result.input_tokens
        record.output_tokens = result.output_tokens
        record.raw_text_len = len(result.raw_text or "")
        payload = result.output or {}
        record.raw_output = payload
        record.evidence_sufficient = payload.get("evidence_sufficient")
        for item in payload.get("sections") or []:
            act = item.get("act") or ""
            number = item.get("section") or ""
            if number:
                record.cited_sections.append(_section_key(act, str(number)))
            if item.get("chunk_id"):
                record.cited_chunk_ids.append(str(item["chunk_id"]))
    except Exception as exc:  # noqa: BLE001 - a model failing is a datapoint, not a crash
        record.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        record.error = f"{type(exc).__name__}: {exc}"[:300]
    return record


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--with-baseline", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    models = args.models or list(DEFAULT_MODELS)
    if args.with_baseline and BASELINE_MODEL not in models:
        models.append(BASELINE_MODEL)

    tasks = TASKS[: args.limit] if args.limit else TASKS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"models : {', '.join(models)}")
    print(f"tasks  : {len(tasks)}")
    print(f"total  : {len(tasks) * len(models)} model calls\n")

    # ---------------------------------------------------------------- phase 1: retrieval
    print("PHASE 1 - retrieval (once per question, shared by all models)")
    retrievals: dict[str, RetrievalRecord] = {}
    chunk_cache: dict[str, list[RetrievedChunk]] = {}
    for index, task in enumerate(tasks, start=1):
        record, chunks = await run_retrieval(task)
        retrievals[task.id] = record
        chunk_cache[task.id] = chunks
        flag = "" if not record.expected_missed else f"  MISSED {record.expected_missed}"
        print(
            f"  [{index:>2}/{len(tasks)}] {task.id} recall={record.recall:.2f} "
            f"mrr={record.mrr:.2f} {record.retrieval_ms:>7.0f}ms{flag}"
        )
    (OUT_DIR / "retrieval.json").write_text(
        json.dumps({k: asdict(v) for k, v in retrievals.items()}, indent=2), encoding="utf-8"
    )
    print(f"  -> {OUT_DIR / 'retrieval.json'}\n")

    # ------------------------------------------------------------------ phase 2: models
    for model in models:
        print(f"PHASE 2 - {model}")
        out_path = OUT_DIR / f"model_{model.replace(':', '_').replace('/', '_')}.json"
        done: dict[str, dict] = {}
        if out_path.exists():
            done = {r["task_id"]: r for r in json.loads(out_path.read_text(encoding="utf-8"))}
            if done:
                print(f"  resuming, {len(done)} already recorded")

        records: list[dict] = list(done.values())
        for index, task in enumerate(tasks, start=1):
            if task.id in done:
                continue
            record = await run_model(task, model, chunk_cache[task.id])
            records.append(asdict(record))
            status = "ok " if record.parse_ok else ("ERR" if record.error else "bad")
            print(
                f"  [{index:>2}/{len(tasks)}] {task.id} {status} "
                f"{record.latency_ms:>7.0f}ms out={record.output_tokens or 0:>4} "
                f"cites={len(record.cited_sections)}"
            )
            # Persist after each call so a long run is never lost.
            out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"  -> {out_path}\n")

    print("done")


if __name__ == "__main__":
    asyncio.run(main())
