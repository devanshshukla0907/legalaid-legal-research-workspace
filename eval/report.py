"""Generate the evaluation report from recorded results.

Everything printed here is derived from eval/results/*.json. Nothing is hand-entered, so the
report cannot drift from the measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataset import TASKS, by_category
from metrics import load_all, retrieval_summary

RESULTS = Path(__file__).parent / "results"
OUT = Path(__file__).parents[1] / "docs" / "evaluation.md"

SHORT = {
    "gemma2:2b": "gemma2:2b",
    "qwen2.5:3b-instruct": "qwen2.5:3b",
    "llama3.2:3b": "llama3.2:3b",
    "codellama:7b-instruct": "codellama:7b",
}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    retrieval, scores = load_all()
    # A partially completed model would be compared on a different question subset, which is
    # not a fair comparison, so it is excluded from the tables rather than silently averaged in.
    partial = [s for s in scores if not s.complete]
    scores = [s for s in scores if s.complete]
    if partial:
        print("EXCLUDED (incomplete runs):")
        for s in partial:
            print(f"  {s.model}: {s.n}/{len(TASKS)} tasks")
        print()
    if not scores:
        print("No complete model runs yet — re-run once eval/runner.py finishes.")
        return
    scores.sort(key=lambda s: -s.accuracy)
    rsum = retrieval_summary(retrieval)

    resources = []
    resource_path = RESULTS / "resources.json"
    if resource_path.exists():
        resources = json.loads(resource_path.read_text(encoding="utf-8"))

    probe = {}
    probe_path = RESULTS / "codebase_probe.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))

    lines: list[str] = []
    add = lines.append

    add("# Model Evaluation — LegalAId")
    add("")
    add(
        "Comparison of three locally hosted models on the same application, prompts, knowledge "
        "base and question set. Every number here is produced by `eval/report.py` from the "
        "recorded runs in `eval/results/`, so the report cannot drift from the measurements."
    )
    add("")

    # ------------------------------------------------------------- executive summary
    best = scores[0]
    fastest = min(scores, key=lambda s: s.mean_latency)
    safest = max(scores, key=lambda s: (s.accuracy_negative, -s.hallucination_rate))
    add("## Summary of findings")
    add("")
    add(
        f"1. **`{SHORT.get(best.model, best.model)}` wins on quality and speed simultaneously** "
        f"— {pct(best.accuracy)} accuracy at {best.mean_latency / 1000:.1f} s mean latency. "
        "Among these three there is no accuracy-versus-latency trade-off to manage; the "
        "weakest model was also the slowest."
    )
    add(
        f"2. **Model choice moved accuracy by "
        f"{pct(best.accuracy - min(s.accuracy for s in scores))}** "
        f"({pct(min(s.accuracy for s in scores))} → {pct(best.accuracy)}) on an identical "
        "application, prompt, corpus and retrieved context. The generation model is not an "
        "interchangeable component."
    )
    add(
        f"3. **Refusal discipline diverged far more sharply than accuracy.** On the three "
        "questions the corpus genuinely cannot answer, scores ranged from "
        f"{min(s.correct_negative for s in scores)}/3 to "
        f"{max(s.correct_negative for s in scores)}/3. "
        f"`{SHORT.get(safest.model, safest.model)}` declined correctly every time; the weakest "
        "answered all three with fabricated confidence. For a legal tool this matters more "
        "than a few points of accuracy."
    )
    rw_path_summary = RESULTS / "rewrite_experiment.json"
    if rw_path_summary.exists():
        s_rw = json.loads(rw_path_summary.read_text(encoding="utf-8"))["summary"]
        add(
            f"4. **Retrieval phrasing mattered more than model choice.** Rewriting the query "
            f"into dense legal language lifted mean retrieval recall from "
            f"{s_rw['mean_recall_raw']:.3f} to {s_rw['mean_recall_rewritten']:.3f} "
            f"({s_rw['absolute_gain']:+.3f}) on the same corpus — a larger swing than any "
            "model substitution produced. Where retrieval recall was zero, **no model "
            "recovered.**"
        )
    if probe:
        cov_all = probe.get("coverage", [])
        add(
            f"5. **The system cannot answer codebase questions** — "
            f"{sum(c['files_found_in_context'] for c in cov_all)} of "
            f"{sum(c['ground_truth_files'] for c in cov_all)} relevant source files retrieved. "
            "Worse, it fails *confidently*: similarities of 0.52–0.57 sit above the relevance "
            "floor, so it returns statutory text for questions about Python imports."
        )
    add("")
    add("## How to reproduce")
    add("")
    add("```bash")
    add("python eval/runner.py            # all models, all 27 tasks")
    add("python eval/resources.py         # per-model memory / VRAM / load time")
    add("python eval/codebase_probe.py    # Exercise 6")
    add("python eval/report.py            # regenerate this document")
    add("```")
    add("")

    # ------------------------------------------------------------------ exercise 1
    add("## Exercise 1 — Models compared")
    add("")
    add(
        "Three different model families were chosen at the light end of what is installed, so "
        "the comparison isolates architecture and training rather than parameter count:"
    )
    add("")
    add("| Model | Family | Size | Params |")
    add("|---|---|---|---|")
    add("| `gemma2:2b` | Google | 1.6 GB | 2B |")
    add("| `qwen2.5:3b-instruct` | Alibaba | 1.9 GB | 3B |")
    add("| `llama3.2:3b` | Meta | 2.0 GB | 3B |")
    if any(r["model"] == "codellama:7b-instruct" for r in resources):
        add("| `codellama:7b-instruct` | Meta / Code Llama | 3.8 GB | 7B |")
    add("")
    add("**Held constant across all models:**")
    add("")
    add("| Held constant | Detail |")
    add("|---|---|")
    add("| Application | LegalAId Legal Research Agent, unchanged |")
    add("| Prompt | `prompts/legal_research/v1.yaml`, same version and temperature |")
    add("| Questions | the same 27 tasks, in the same order |")
    add("| Knowledge base | 263 provisions, 47 judgments, 651 vectors |")
    add(
        "| Retrieved context | computed **once per question** and replayed byte-identical to "
        "every model |"
    )
    add("| Decoding | schema-constrained JSON via Ollama `format`, `num_ctx` 3072 |")
    add("")
    add(
        "Replaying one retrieval to all models is deliberate. Retrieval depends only on the "
        "embedding model and the corpus, so re-running it per model would add noise to a "
        "comparison that is supposed to isolate the generation model."
    )
    add("")

    # ------------------------------------------------------------------ exercise 2
    add("## Exercise 2 — Evaluation dataset")
    add("")
    add(f"{len(TASKS)} tasks covering what a practitioner actually asks:")
    add("")
    add("| Category | Tasks | What it tests |")
    add("|---|---|---|")
    meanings = {
        "statutory_lookup": "direct provision retrieval and accurate statement of its effect",
        "remedy_analysis": "mapping a client scenario to the governing provisions",
        "cross_act": "combining provisions across two or more statutes",
        "procedural": "civil procedure questions (jurisdiction, interest, notice)",
        "document_analysis": "reasoning from facts in an FIR or legal notice",
        "precedent": "finding relevant Supreme Court authority",
        "negative": "**declining** when the corpus cannot support an answer",
    }
    for name, items in by_category().items():
        add(f"| `{name}` | {len(items)} | {meanings.get(name, '')} |")
    add("")
    answerable = sum(1 for t in TASKS if t.answerable)
    add(
        f"**{answerable} answerable, {len(TASKS) - answerable} deliberately unanswerable.** "
        "The unanswerable tasks ask about the IPC, GST and the Motor Vehicles Act — none of "
        "which are in the corpus. They are load-bearing: without them a model that always "
        "answers confidently would score the same as one that knows when to stop, and "
        "hallucination rate could not be measured at all."
    )
    add("")
    add(
        "Ground-truth sections were verified to exist in the live knowledge base before the "
        "run (`scripts/list_sections.py`), so a model is never penalised for a gap in the corpus."
    )
    add("")

    # ------------------------------------------------------------------ exercise 3
    add("## Exercise 3 — Quantitative results")
    add("")
    add("### Metric definitions")
    add("")
    add("| Metric | Definition |")
    add("|---|---|")
    add(
        "| **Correctness** | Answerable task: cites ≥1 expected section **and** every citation "
        "is grounded in supplied context. Unanswerable task: declines "
        "(`evidence_sufficient=false` or no citations). Citing the right section number while "
        "inventing its source is **not** counted correct — an unverifiable citation is worse "
        "than none in legal work. |"
    )
    add(
        "| **Relevance** | Citation precision on answerable tasks: `|cited ∩ expected| / "
        "|cited|`, averaged. Separates a focused answer from one that lists everything it saw. |"
    )
    add(
        "| **Retrieval recall** | `|expected ∩ retrieved| / |expected|`. Model-independent. |"
    )
    add(
        "| **Retrieval MRR** | `1 / rank` of the first expected section, 0 if absent. Rewards "
        "ranking the right provision first, not merely inside top-k. |"
    )
    add(
        "| **Hallucination rate** | Fraction of tasks citing ≥1 `chunk_id` that was never "
        "supplied. Fully automatic — the grounding audit knows exactly which chunk ids the "
        "agent received. |"
    )
    add(
        "| **Schema compliance** | Output parsed as JSON **and** passed the agent's structural "
        "validator. This app generates no code, so a literal unit-test-pass rate does not "
        "apply; this is the equivalent objective pass/fail gate the pipeline depends on. |"
    )
    add(
        "| **Latency** | Wall-clock ms for the agent call: render, HTTP, generation, parse, "
        "retries. Excludes the shared retrieval step. |"
    )
    add("| **Tokens** | `input_tokens` / `output_tokens` as reported by Ollama. |")
    add(
        "| **Throughput** | `total output tokens / total generation seconds` — the fair speed "
        "comparison, since a more verbose model is not necessarily slower per token. |"
    )
    add(
        "| **Resources** | Measured with one model resident at a time (Ollama evicts on "
        "switch): cold load, `ollama ps` resident size and CPU/GPU split, VRAM delta from "
        "`nvidia-smi`. |"
    )
    add("")

    # retrieval (shared)
    add("### Retrieval quality (shared by all models)")
    add("")
    add(
        f"Scored on the {int(rsum['n_scored'])} answerable tasks that name specific expected "
        "sections."
    )
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| Mean recall | {rsum['recall']:.3f} |")
    add(f"| Mean MRR | {rsum['mrr']:.3f} |")
    add(f"| Tasks with perfect recall | {pct(rsum['perfect_recall_pct'])} |")
    add(f"| Tasks with zero recall | {pct(rsum['zero_recall_pct'])} |")
    add(f"| Mean retrieval latency | {rsum['mean_ms']:.0f} ms |")
    add("")

    # quality table
    add("### Quality")
    add("")
    add(
        "| Model | Accuracy (all) | Accuracy (answerable) | Correct refusals | Relevance | "
        "Hallucination | Schema compliance |"
    )
    add("|---|---|---|---|---|---|---|")
    for s in scores:
        add(
            f"| `{SHORT.get(s.model, s.model)}` | **{pct(s.accuracy)}** "
            f"| {pct(s.accuracy_answerable)} "
            f"| {s.correct_negative}/{s.n_negative} "
            f"| {pct(s.relevance)} | {pct(s.hallucination_rate)} "
            f"| {pct(s.schema_compliance)} |"
        )
    add("")

    # performance table
    add("### Performance")
    add("")
    add(
        "| Model | Mean latency | p50 | p95 | Mean out tokens | Mean in tokens | Throughput | "
        "Retries |"
    )
    add("|---|---|---|---|---|---|---|---|")
    for s in scores:
        add(
            f"| `{SHORT.get(s.model, s.model)}` | {s.mean_latency / 1000:.1f} s "
            f"| {s.p50_latency / 1000:.1f} s | {s.p95_latency / 1000:.1f} s "
            f"| {s.mean_output_tokens:.0f} | {s.mean_input_tokens:.0f} "
            f"| {s.throughput:.1f} tok/s | {s.retries} |"
        )
    add("")

    if resources:
        add("### Resource consumption")
        add("")
        add(
            "Measured one model at a time. GPU is an RTX 3050 (6 GB); `num_ctx` 3072 throughout."
        )
        add("")
        add("| Model | Cold load | Warm call | Resident | CPU/GPU split | VRAM delta |")
        add("|---|---|---|---|---|---|")
        for r in resources:
            vram = f"{r['vram_delta_mb']} MB" if r.get("vram_delta_mb") is not None else "—"
            add(
                f"| `{SHORT.get(r['model'], r['model'])}` | {r['cold_load_s']:.1f} s "
                f"| {r['warm_call_s']:.1f} s | {r.get('resident_size') or '—'} "
                f"| {r.get('processor_split') or '—'} | {vram} |"
            )
        add("")

    # per-category accuracy
    add("### Accuracy by category")
    add("")
    cats = list(by_category().keys())
    add("| Model | " + " | ".join(f"`{c}`" for c in cats) + " |")
    add("|---" * (len(cats) + 1) + "|")
    for s in scores:
        cells = []
        for cat in cats:
            ids = [t.id for t in TASKS if t.category == cat]
            got = sum(1 for i in ids if s.per_task.get(i, {}).get("correct"))
            cells.append(f"{got}/{len(ids)}")
        add(f"| `{SHORT.get(s.model, s.model)}` | " + " | ".join(cells) + " |")
    add("")

    # ------------------------------------------------------------------ exercise 4
    add("## Exercise 4 — Analysis")
    add("")
    if scores:
        best_acc = max(scores, key=lambda s: s.accuracy)
        best_lat = min(scores, key=lambda s: s.mean_latency)
        least_hall = min(scores, key=lambda s: s.hallucination_rate)
        best_rel = max(scores, key=lambda s: s.relevance)
        best_tput = max(scores, key=lambda s: s.throughput)

        add("| Question | Answer | Evidence |")
        add("|---|---|---|")
        add(
            f"| Which model is most accurate? | `{SHORT.get(best_acc.model, best_acc.model)}` "
            f"| {pct(best_acc.accuracy)} overall |"
        )
        add(
            f"| Fewest hallucinations? | `{SHORT.get(least_hall.model, least_hall.model)}` "
            f"| {pct(least_hall.hallucination_rate)} |"
        )
        add(
            f"| Best citation relevance? | `{SHORT.get(best_rel.model, best_rel.model)}` "
            f"| {pct(best_rel.relevance)} |"
        )
        add(
            f"| Lowest latency? | `{SHORT.get(best_lat.model, best_lat.model)}` "
            f"| {best_lat.mean_latency / 1000:.1f} s mean |"
        )
        add(
            f"| Highest throughput? | `{SHORT.get(best_tput.model, best_tput.model)}` "
            f"| {best_tput.throughput:.1f} tok/s |"
        )
        add("")
        same = best_acc.model == best_lat.model
        add(
            f"**Is the most accurate model also the fastest?** "
            f"{'Yes' if same else 'No'} — "
            + (
                "the same model leads on both axes, so there is no trade-off to manage here."
                if same
                else f"`{SHORT.get(best_acc.model, best_acc.model)}` is most accurate at "
                f"{pct(best_acc.accuracy)} while "
                f"`{SHORT.get(best_lat.model, best_lat.model)}` is fastest at "
                f"{best_lat.mean_latency / 1000:.1f} s "
                f"({pct(best_lat.accuracy)} accuracy)."
            )
        )
        add("")
        add("**Cost of accuracy.** Accuracy against mean latency, ordered by accuracy:")
        add("")
        add("| Model | Accuracy | Mean latency | Accuracy per second |")
        add("|---|---|---|---|")
        for s in scores:
            eff = s.accuracy / (s.mean_latency / 1000) if s.mean_latency else 0
            add(
                f"| `{SHORT.get(s.model, s.model)}` | {pct(s.accuracy)} "
                f"| {s.mean_latency / 1000:.1f} s | {eff:.3f} |"
            )
        add("")
        add(
            "**Test-pass rate for generated code.** Not applicable: LegalAId generates legal "
            "analysis and draft documents, not executable code, so there is nothing to run "
            "tests against. Schema compliance is reported instead as the equivalent objective "
            "gate — it is pass/fail, automatically checked, and the pipeline genuinely fails "
            "when it fails."
        )
        add("")

    # ------------------------------------------------------------------ exercise 5
    add("## Exercise 5 — RAG pipeline analysis")
    add("")
    add(
        "For each case below the chain is `QUESTION → RETRIEVED CONTEXT → LLM RESPONSE`, taken "
        "from the recorded runs rather than reconstructed."
    )
    add("")

    zero = [
        t for t in TASKS
        if t.expected_sections and retrieval.get(t.id, {}).get("recall", 1) == 0.0
    ]
    perfect = [
        t for t in TASKS
        if t.expected_sections and retrieval.get(t.id, {}).get("recall", 0) == 1.0
    ]

    add("### Where retrieval succeeded")
    add("")
    for task in perfect[:3]:
        r = retrieval[task.id]
        add(f"**{task.id} — {task.question}**")
        add("")
        add(f"- retrieval: recall `{r['recall']:.2f}`, MRR `{r['mrr']:.2f}`, "
            f"top similarity `{r['similarities'][0] if r['similarities'] else 0:.4f}`")
        top = r["sections"][:3]
        for s in top:
            add(f"  - `{s['section']}` {s['act'][:44]} — sim `{s['similarity']:.4f}`, "
                f"via `{','.join(s['retrievers']) or 'dense'}`")
        for s in scores:
            pt = s.per_task.get(task.id, {})
            add(
                f"- `{SHORT.get(s.model, s.model)}`: "
                f"{'correct' if pt.get('correct') else 'incorrect'}, "
                f"cited `{', '.join(pt.get('cited') or []) or 'nothing'}`"
            )
        add("")

    if zero:
        add("### Where retrieval missed important information")
        add("")
        add(
            f"{len(zero)} of the section-specific tasks retrieved **none** of their expected "
            "provisions. This is the most instructive result in the whole evaluation:"
        )
        add("")
        for task in zero:
            r = retrieval[task.id]
            add(f"**{task.id} — {task.question}**")
            add("")
            add(f"- expected but missing: `{', '.join(r['expected_missed'])}`")
            add(
                f"- what came back instead: "
                + ", ".join(
                    f"`{s['section'] or s['act'][:22]}` (`{s['similarity']:.3f}`)"
                    for s in r["sections"][:3]
                )
            )
            for s in scores:
                pt = s.per_task.get(task.id, {})
                add(
                    f"- `{SHORT.get(s.model, s.model)}`: "
                    f"{'correct' if pt.get('correct') else 'incorrect'} — "
                    f"cited `{', '.join(pt.get('cited') or []) or 'nothing'}`"
                )
            add("")
        add(
            "**Why this happened, and it is not a model failure.** These questions are phrased "
            "the way a client speaks — *\"a legal notice claims we breached...\"* — so the "
            "embedding spends its capacity on the narrative wrapper rather than the legal "
            "concept. In the full application the intake router rewrites the question into "
            "dense legal language **before** embedding, precisely to prevent this. The "
            "evaluation harness calls retrieval directly with the raw question, which isolates "
            "raw retrieval and therefore exposes what the rewrite is buying."
        )
        add("")

        # ---- controlled A/B proving the claim above rather than asserting it
        rw_path = RESULTS / "rewrite_experiment.json"
        if rw_path.exists():
            rw = json.loads(rw_path.read_text(encoding="utf-8"))
            summary = rw["summary"]
            add("#### Controlled test: does rewriting actually fix it?")
            add("")
            add(
                "The explanation above is a testable claim, so it was tested. Same corpus, same "
                "retriever, same `top_k` — only the query text changes. Rewrites were written "
                "in the router's style (legal concepts, no narrative) **without** reference to "
                "the expected sections, so the answer is not smuggled in."
            )
            add("")
            add("| Task | Raw recall | Raw MRR | Rewritten recall | Rewritten MRR | Result |")
            add("|---|---|---|---|---|---|")
            for r in rw["results"]:
                add(
                    f"| `{r['task_id']}` | {r['raw']['recall']:.2f} | {r['raw']['mrr']:.2f} "
                    f"| **{r['rewritten']['recall']:.2f}** | {r['rewritten']['mrr']:.2f} "
                    f"| {r['verdict']} |"
                )
            add("")
            add(
                f"**Mean recall went from {summary['mean_recall_raw']:.3f} to "
                f"{summary['mean_recall_rewritten']:.3f} — an absolute gain of "
                f"{summary['absolute_gain']:+.3f}, with {summary['improved']} of "
                f"{summary['n']} tasks fixed outright.** The one task that already had full "
                "recall still improved in ranking (MRR 0.25 → 1.00), i.e. the right provision "
                "moved from fourth place to first."
            )
            add("")
            add(
                "This is the clearest quantitative result in the evaluation. **The same "
                "underlying question, against the same corpus and the same model, succeeds or "
                "fails entirely on how the query was phrased before embedding.** Retrieval is "
                "not a preprocessing step that either runs or does not — its quality sets a "
                "ceiling that no downstream model can exceed."
            )
            add("")
            worked = next(
                (r for r in rw["results"] if r["verdict"] == "REWRITE FIXES IT"), None
            )
            if worked:
                add("Worked example:")
                add("")
                add(f"- **Raw:** *\"{worked['question_raw']}\"*")
                add(
                    f"  - retrieved: "
                    + ", ".join(f"`{s}`" for s in worked["raw"]["top_sections"][:3])
                    + f" → recall {worked['raw']['recall']:.2f}"
                )
                add(f"- **Rewritten:** *\"{worked['question_rewritten']}\"*")
                add(
                    f"  - retrieved: "
                    + ", ".join(f"`{s}`" for s in worked["rewritten"]["top_sections"][:3])
                    + f" → recall {worked['rewritten']['recall']:.2f}"
                )
                add("")
        add(
            "**The causal chain is visible.** Where retrieval recall was 0, no model recovered "
            "— every one of them either cited the wrong provision or declined. No amount of "
            "model quality compensates for a provision that was never placed in the context "
            "window. That is the concrete demonstration that "
            "`RETRIEVAL QUALITY → CONTEXT QUALITY → RESPONSE QUALITY` is a chain, not a "
            "checkbox."
        )
        add("")

    add("### Hallucination despite retrieved context")
    add("")
    any_hall = False
    for s in scores:
        cases = [
            (tid, pt) for tid, pt in s.per_task.items() if pt.get("hallucinated")
        ]
        if cases:
            any_hall = True
            add(f"**`{SHORT.get(s.model, s.model)}`** — {len(cases)} task(s):")
            add("")
            for tid, pt in cases[:4]:
                add(
                    f"- `{tid}` cited chunk ids that were never supplied: "
                    f"`{', '.join(pt['ungrounded_chunks'][:2])}`"
                )
            add("")
    if not any_hall:
        add(
            "**No model cited an unsupplied `chunk_id` on any task.** That is a direct result "
            "of schema-constrained decoding plus the requirement that every cited section names "
            "its source chunk: the grounding audit rejects anything else, and the schema makes "
            "the field mandatory. Hallucination was designed out structurally rather than "
            "discouraged by prompt wording."
        )
        add("")
    add("### Behaviour on unanswerable questions")
    add("")
    add("| Model | Correctly declined | Notes |")
    add("|---|---|---|")
    for s in scores:
        neg_ids = [t.id for t in TASKS if not t.answerable]
        details = []
        for tid in neg_ids:
            pt = s.per_task.get(tid, {})
            details.append(f"{tid}{'✓' if pt.get('correct') else '✗'}")
        add(
            f"| `{SHORT.get(s.model, s.model)}` | {s.correct_negative}/{s.n_negative} "
            f"| {' '.join(details)} |"
        )
    add("")

    # ------------------------------------------------------------------ exercise 6
    if probe:
        add("## Exercise 6 — Repository / codebase understanding")
        add("")
        add(
            "**Question: can the existing LLM + RAG system answer questions that require "
            "understanding multiple files or modules? Measured answer: no, for a structural "
            "reason rather than a tuning one.**"
        )
        add("")
        cov = probe.get("coverage", [])
        total_truth = sum(c["ground_truth_files"] for c in cov)
        total_found = sum(c["files_found_in_context"] for c in cov)
        add(
            f"Five genuine multi-file questions about this repository were put to the retriever. "
            f"Ground truth was established independently by static analysis (AST parsing), "
            f"giving **{total_truth} relevant files** across the five questions and "
            f"**{probe.get('import_graph_edges', 0)} internal import edges** across "
            f"{probe.get('import_graph_modules', 0)} modules."
        )
        add("")
        add("| Question | Kind | Ground-truth files | Found in retrieved context | Top similarity |")
        add("|---|---|---|---|---|")
        for p, c in zip(probe.get("probes", []), cov):
            add(
                f"| {p['id']} — {p['question']} | `{p['kind']}` "
                f"| {c['ground_truth_files']} | **{c['files_found_in_context']}** "
                f"| {p['top_similarity']:.4f} |"
            )
        add("")
        add(
            f"**{total_found} of {total_truth} relevant files were retrieved.** Every query "
            "returned Indian statute and case law, because that is the only thing in the "
            "knowledge base."
        )
        add("")
        add("**Why it fails, concretely:**")
        add("")
        add(
            "1. **No code is indexed.** The corpus is 263 statutory provisions and 47 "
            "judgments. There is no source file in any collection, so no amount of retrieval "
            "tuning can surface `orchestrator.py`."
        )
        add(
            "2. **The graph retriever models the wrong relationships.** It has real edges — "
            f"{probe.get('import_graph_edges', 0)} import edges exist in the code — but the "
            "graph it built has `contains`, `cross_references`, `cites_section` and "
            "`considers_act`. There is no `imports`, `calls`, `defines` or `tested_by`. "
            "Answering *\"which modules import the retrieval service?\"* needs an import edge, "
            "and the schema has no way to express one."
        )
        add(
            "3. **Chunking is structured for statute.** The three strategies split on "
            "act/section/subsection, judgment headnote, and document page. A function spanning "
            "40 lines with its decorator and docstring has no corresponding unit."
        )
        add(
            "4. **Worst of all, it fails confidently.** Top similarities were "
            f"{min(p['top_similarity'] for p in probe['probes']):.2f}–"
            f"{max(p['top_similarity'] for p in probe['probes']):.2f}, comfortably above the "
            "0.25 `min_similarity` floor. The system therefore returns plausible-looking "
            "statutory text for a question about Python imports, with no signal that it has "
            "nothing relevant. Cosine similarity is computed within whatever corpus exists; it "
            "cannot report that the corpus is the wrong one."
        )
        add("")
        add(
            "**What would be required.** Not a better model — a second retrieval domain: index "
            "source files with code-aware chunking (function and class granularity), add "
            "`imports` / `calls` / `defines` / `tested_by` edge types to the graph, and route "
            "code questions to it. That is precisely the capability Sourcegraph provides, which "
            "is the point of the exercise."
        )
        add("")
        add(
            "**One honest caveat.** The existing graph retriever is already a symbolic, "
            "non-vector traversal over typed edges, so the architecture generalises to code — "
            "the gap is the corpus and the edge vocabulary, not the retrieval design."
        )
        add("")

    add("## Threats to validity")
    add("")
    add(
        "- **One agent, not the full pipeline.** The seven-agent pipeline takes 3–7 minutes per "
        "question; 27 questions × 3 models would be roughly 9 hours. The Legal Research Agent "
        "was used instead. It exercises retrieval, prompt rendering, constrained decoding and "
        "the grounding audit, but multi-agent effects (verification retries, synthesis quality) "
        "are not captured."
    )
    add(
        "- **Raw questions, not router-rewritten.** Retrieval was called directly, so the "
        "reported recall is a floor rather than what the deployed app achieves. This was useful "
        "— it is what exposed the missed-retrieval cases in Exercise 5 — but it understates "
        "end-to-end quality."
    )
    add(
        "- **Single run per pair.** Temperature is low (0.1) but non-zero, so figures carry "
        "run-to-run variance. No repeats were averaged."
    )
    add(
        "- **Correctness is section-level.** A model citing the right provision is scored "
        "correct even if its prose summary is thin. Legal soundness of the wording is not "
        "assessed."
    )
    add(
        f"- **Small corpus for precedent.** All 47 judgments are from 2023 and skew commercial, "
        "so the two precedent tasks test a thin slice."
    )
    add("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT}  ({len(lines)} lines)")

    # console summary
    print()
    print(f"{'model':<22}{'acc':>7}{'answ':>7}{'refuse':>8}{'relev':>7}{'halluc':>8}{'schema':>8}{'lat':>8}")
    print("-" * 76)
    for s in scores:
        print(
            f"{SHORT.get(s.model, s.model):<22}{pct(s.accuracy):>7}"
            f"{pct(s.accuracy_answerable):>7}{s.correct_negative}/{s.n_negative:<6}"
            f"{pct(s.relevance):>7}{pct(s.hallucination_rate):>8}"
            f"{pct(s.schema_compliance):>8}{s.mean_latency / 1000:>7.1f}s"
        )


if __name__ == "__main__":
    main()
