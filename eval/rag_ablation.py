"""E6: the same questions with and without retrieval.

The Week 4 evaluation measured hallucination only *with* retrieval, which is a measurement
without a control. This supplies the control: identical questions, identical model, identical
prompt shape, the only difference being whether retrieved statutory text is in the context.

Scoring is the interesting part. In the no-RAG condition there is no supplied context, so the
grounding audit has nothing to check citations against - the usual metric is undefined. Instead
each condition is scored on whether the sections it names actually exist in the corpus and say
what the model claims:

corpus_verified
    The model named a section that exists in the knowledge base, for the right act.
fabricated
    The model named a section that does not exist in that act at all. This is the strongest
    possible evidence of hallucination: not a judgement about phrasing, a lookup that fails.
correct
    The model named the section the task expects.

A no-RAG answer that happens to be right is still counted right. The claim being tested is not
"the model knows nothing" - it is that without retrieval the model cannot tell the difference
between what it knows and what it is inventing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sqlalchemy import select

from dataset import TASKS, EvalTask
from legalaid.db.base import session_scope
from legalaid.db.models import LegalProvision
from legalaid.domain.contracts import LLMGenerateRequest, RetrievalRequest
from legalaid.domain.enums import AgentKey, RetrievalMode
from legalaid.services.llm import service as llm_service
from legalaid.services.retrieval.service import search

OUT = Path(__file__).parent / "results" / "rag_ablation.json"

MODEL = "llama3.2:3b"  # best accuracy in the Week 4 comparison

# Section references in prose, with the act they are attributed to nearby.
_SECTION_RE = re.compile(r"\b(?:section|sec\.?|s\.)\s*(\d+[A-Za-z]{0,2})\b", re.I)

ACT_HINTS: dict[str, tuple[str, ...]] = {
    "indian contract act": ("contract act", "ica", "indian contract"),
    "code of civil procedure": ("cpc", "civil procedure"),
    "information technology act": ("it act", "information technology"),
    "bharatiya sakshya adhiniyam": ("bsa", "sakshya", "evidence act"),
    "bharatiya nyaya sanhita": ("bns", "nyaya"),
    "specific relief act": ("specific relief",),
    "limitation act": ("limitation",),
    "arbitration and conciliation act": ("arbitration",),
    "negotiable instruments act": ("negotiable instruments", "ni act"),
    "sale of goods act": ("sale of goods",),
    "transfer of property act": ("transfer of property", "tpa"),
    "consumer protection act": ("consumer protection",),
    "right to information act": ("rti", "right to information"),
    "indian partnership act": ("partnership",),
}

SYSTEM_NO_RAG = (
    "You are a legal assistant for Indian law. Answer the question concisely from your own "
    "knowledge. Name the specific sections and Acts that apply."
)
SYSTEM_WITH_RAG = (
    "You are a legal assistant for Indian law. Answer ONLY from the provided context. "
    "Name the specific sections and Acts that apply. If the context does not answer the "
    "question, say so explicitly rather than answering from memory."
)


def load_corpus_index() -> dict[str, set[str]]:
    """act-name token -> set of section numbers that really exist."""
    with session_scope() as session:
        rows = session.scalars(select(LegalProvision)).all()
    index: dict[str, set[str]] = {}
    for row in rows:
        index.setdefault((row.act_name or "").lower(), set()).add(
            (row.section_number or "").upper()
        )
    return index


@dataclass
class ConditionResult:
    task_id: str
    condition: str
    answer: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    sections_named: list[str] = field(default_factory=list)
    corpus_verified: list[str] = field(default_factory=list)
    fabricated: list[str] = field(default_factory=list)
    hit_expected: bool = False
    declined: bool = False
    error: str | None = None


def attribute_sections(answer: str, index: dict[str, set[str]]) -> tuple[list[str], list[str], list[str]]:
    """Return (named, verified, fabricated) section references found in the prose.

    A section is attributed to whichever act is mentioned nearest before it. Where no act can
    be identified the reference is recorded as named but neither verified nor fabricated -
    counting an unattributable reference as a fabrication would overstate the result.
    """
    lowered = answer.lower()
    named: list[str] = []
    verified: list[str] = []
    fabricated: list[str] = []

    for match in _SECTION_RE.finditer(answer):
        number = match.group(1).upper()
        window = lowered[max(0, match.start() - 170) : match.start() + 170]
        act_name = None
        for canonical, hints in ACT_HINTS.items():
            if canonical in window or any(hint in window for hint in hints):
                act_name = canonical
                break
        label = f"{act_name or 'unattributed'} s.{number}"
        named.append(label)
        if act_name is None:
            continue
        # Resolve the canonical name to the stored act name.
        stored = next((k for k in index if act_name in k), None)
        if stored is None:
            continue
        if number in index[stored]:
            verified.append(label)
        else:
            fabricated.append(label)
    return named, verified, fabricated


async def run_condition(
    task: EvalTask, *, with_rag: bool, index: dict[str, set[str]], model: str
) -> ConditionResult:
    context_block = ""
    if with_rag:
        response = await search(
            RetrievalRequest(
                question=task.question, mode=RetrievalMode.COMBINED_LEGAL_RESEARCH, top_k=6
            )
        )
        context_block = "\n\n".join(
            f"[{i}] {c.section or ''} {c.source_title}\n{(c.text or '')[:600]}"
            for i, c in enumerate(response.chunks[:6], start=1)
        )

    prompt = (
        f"CONTEXT\n{context_block}\n\nQUESTION\n{task.question}"
        if with_rag
        else task.question
    )
    started = time.perf_counter()
    result = ConditionResult(
        task_id=task.id,
        condition="with_rag" if with_rag else "no_rag",
        answer="",
        latency_ms=0.0,
        input_tokens=None,
        output_tokens=None,
    )
    try:
        generated = await llm_service.generate(
            LLMGenerateRequest(
                agent_key=AgentKey.DIRECT_LLM,
                system_prompt=SYSTEM_WITH_RAG if with_rag else SYSTEM_NO_RAG,
                user_prompt=prompt,
                model=model,
                max_tokens=420,
            )
        )
        result.answer = generated.text or ""
        result.input_tokens = generated.input_tokens
        result.output_tokens = generated.output_tokens
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"[:200]
    result.latency_ms = round((time.perf_counter() - started) * 1000, 1)

    named, verified, fabricated = attribute_sections(result.answer, index)
    result.sections_named = named
    result.corpus_verified = verified
    result.fabricated = fabricated

    expected = {num.upper() for _act, num in task.expected_sections}
    result.hit_expected = any(
        label.split("s.")[-1].strip().upper() in expected for label in verified
    )
    lowered = result.answer.lower()
    result.declined = any(
        phrase in lowered
        for phrase in (
            "does not answer", "not in the context", "cannot answer", "no information",
            "context does not", "insufficient", "not provided", "unable to answer",
        )
    )
    return result


def summarise(results: list[ConditionResult], condition: str, tasks: list[EvalTask]) -> dict:
    subset = [r for r in results if r.condition == condition]
    answerable_ids = {t.id for t in tasks if t.answerable}
    negative_ids = {t.id for t in tasks if not t.answerable}

    total_named = sum(len(r.sections_named) for r in subset)
    total_fabricated = sum(len(r.fabricated) for r in subset)
    total_verified = sum(len(r.corpus_verified) for r in subset)

    answerable = [r for r in subset if r.task_id in answerable_ids]
    negative = [r for r in subset if r.task_id in negative_ids]

    return {
        "condition": condition,
        "tasks": len(subset),
        "sections_named": total_named,
        "corpus_verified": total_verified,
        "fabricated": total_fabricated,
        "fabrication_rate": round(total_fabricated / total_named, 4) if total_named else 0.0,
        "verified_rate": round(total_verified / total_named, 4) if total_named else 0.0,
        "tasks_with_a_fabrication": sum(1 for r in subset if r.fabricated),
        "accuracy_answerable": round(
            sum(1 for r in answerable if r.hit_expected) / len(answerable), 4
        )
        if answerable
        else 0.0,
        "declined_on_unanswerable": sum(1 for r in negative if r.declined),
        "unanswerable_total": len(negative),
        "fabricated_on_unanswerable": sum(1 for r in negative if r.fabricated),
        "mean_latency_ms": round(statistics.fmean([r.latency_ms for r in subset]), 1)
        if subset
        else 0.0,
        "mean_output_tokens": round(
            statistics.fmean([r.output_tokens or 0 for r in subset]), 1
        )
        if subset
        else 0.0,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = TASKS[: args.limit] if args.limit else TASKS
    index = load_corpus_index()
    print(f"corpus index: {len(index)} acts, "
          f"{sum(len(v) for v in index.values())} sections")
    print(f"model: {args.model}   tasks: {len(tasks)}   conditions: 2\n")

    results: list[ConditionResult] = []
    for position, task in enumerate(tasks, start=1):
        for with_rag in (False, True):
            result = await run_condition(
                task, with_rag=with_rag, index=index, model=args.model
            )
            results.append(result)
            tag = "RAG" if with_rag else "raw"
            print(
                f"  [{position:>2}/{len(tasks)}] {task.id} {tag:<4} "
                f"named={len(result.sections_named):<2} "
                f"fabricated={len(result.fabricated):<2} "
                f"{'hit' if result.hit_expected else '   '} "
                f"{'declined' if result.declined else ''}"
            )

    no_rag = summarise(results, "no_rag", tasks)
    with_rag = summarise(results, "with_rag", tasks)

    print(f"\n{'metric':<34}{'no RAG':>12}{'with RAG':>12}")
    print("-" * 58)
    for key, label in [
        ("sections_named", "sections named"),
        ("corpus_verified", "verified in corpus"),
        ("fabricated", "fabricated"),
        ("fabrication_rate", "fabrication rate"),
        ("accuracy_answerable", "accuracy (answerable)"),
        ("declined_on_unanswerable", "declined on unanswerable"),
        ("fabricated_on_unanswerable", "fabricated on unanswerable"),
        ("mean_latency_ms", "mean latency ms"),
    ]:
        left, right = no_rag[key], with_rag[key]
        fmt = (lambda v: f"{v:.3f}") if isinstance(left, float) else (lambda v: f"{v}")
        print(f"{label:<34}{fmt(left):>12}{fmt(right):>12}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "model": args.model,
                "no_rag": no_rag,
                "with_rag": with_rag,
                "results": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
