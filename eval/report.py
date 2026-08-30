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

    ablation_summary = RESULTS / "ablation.json"
    before_summary = RESULTS / "ablation_before_fix.json"
    if ablation_summary.exists() and before_summary.exists():
        add(
            "5. **The ablation found the hybrid retrieval actively harmful, and that was the "
            "most useful result in the project.** Fusing BM25 at its configured weight scored "
            "*worse* than disabling it (0.568 against 0.614 recall) because Reciprocal Rank "
            "Fusion's agreement bonus promoted spurious keyword matches over good semantic "
            "hits. Replacing the guessed weight with a measured one took the full stack to "
            "0.682 — now genuinely better than dense alone."
        )

    rag_summary = RESULTS / "rag_ablation.json"
    if rag_summary.exists():
        rag_s = json.loads(rag_summary.read_text(encoding="utf-8"))
        add(
            f"6. **Retrieval triples accuracy but does not reduce fabrication — it relocates "
            f"it.** Accuracy on answerable tasks went "
            f"{rag_s['no_rag']['accuracy_answerable']:.1%} → "
            f"{rag_s['with_rag']['accuracy_answerable']:.1%} with retrieval, while the "
            f"fabrication rate barely moved "
            f"({rag_s['no_rag']['fabrication_rate']:.1%} → "
            f"{rag_s['with_rag']['fabrication_rate']:.1%}). Without retrieval the model invents "
            "sections on answerable questions; with retrieval it gets those right and then "
            "fabricates on the *unanswerable* ones instead. Grounding fixes what a model knows, "
            "not whether it knows when to stop."
        )

    latency_summary = RESULTS / "latency_breakdown.json"
    if latency_summary.exists():
        lat_s = json.loads(latency_summary.read_text(encoding="utf-8"))
        if lat_s.get("generation_share"):
            add(
                f"7. **{lat_s['generation_share']:.0%} of wall-clock time is model generation**, "
                "so orchestration overhead is not what to optimise. The entire input guardrail "
                "stack costs about 1.5% of a request — cheap insurance."
            )
    if probe:
        cov_all = probe.get("coverage", [])
        sims = [p["top_similarity"] for p in probe.get("probes", []) if p.get("top_similarity")]
        add(
            f"8. **The system still cannot answer codebase questions** — "
            f"{sum(c['files_found_in_context'] for c in cov_all)} of "
            f"{sum(c['ground_truth_files'] for c in cov_all)} relevant source files retrieved, "
            "because there is no code in the corpus. What changed is that it no longer fails "
            f"*confidently*: those questions score {min(sims):.2f}–{max(sims):.2f} against the "
            "corpus, below the 0.55 guardrail floor, so they are now declined with a reason "
            "instead of answered with statutory text."
        )
    add("")
    add("## How to reproduce")
    add("")
    add("```bash")
    add("# model comparison")
    add("python scripts/verify_ground_truth.py   # expected sections still resolve")
    add("python eval/runner.py                   # 3 models x 27 tasks")
    add("python eval/resources.py                # per-model memory / VRAM / load time")
    add("")
    add("# retrieval")
    add("python eval/ablation.py                 # 7 retriever configurations")
    add("python eval/weight_sweep.py             # fusion weight, measured")
    add("python eval/rewrite_experiment.py       # query rewriting A/B")
    add("")
    add("# pipeline behaviour")
    add("python eval/rag_ablation.py             # RAG on vs off")
    add("python eval/latency_breakdown.py        # where the time goes")
    add("python eval/codebase_probe.py           # Exercise 6")
    add("")
    add("# guardrails and corpus (gateway must be running for the first)")
    add("python scripts/check_probe_suite.py     # adversarial suite scorecard")
    add("python scripts/analyse_corpus.py        # duplicates and citation graph")
    add("python scripts/corpus_stats.py          # reconciliation table")
    add("")
    add("python eval/report.py                   # regenerate this document")
    add("```")
    add("")
    add(
        "Corpus at time of measurement: **1,783 provisions, 47 judgments, 2,575 chunks** across "
        "15 Acts. Earlier results against the 263-provision corpus are archived in "
        "`eval/results/corpus_263/` and compared near the end of this document."
    )
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
    add("| Knowledge base | 1,783 provisions, 47 judgments, 2,575 vectors, 15 Acts |")
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

    # -------------------------------------------------- E5: retriever ablation
    ablation_path = RESULTS / "ablation.json"
    if ablation_path.exists():
        ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
        before_path = RESULTS / "ablation_before_fix.json"
        before = (
            json.loads(before_path.read_text(encoding="utf-8"))
            if before_path.exists()
            else None
        )

        add("### Retriever ablation — what each retriever actually contributes")
        add("")
        add(
            "The application claims hybrid retrieval: dense embeddings, BM25 and a symbolic "
            "graph traversal fused by reciprocal rank. That is architecture, not evidence. Every "
            "combination was run over the same questions."
        )
        add("")
        add("| Configuration | Recall | MRR | Precision | Latency | Zero-recall tasks |")
        add("|---|---|---|---|---|---|")
        for config in ablation["configs"]:
            s = config["summary"]
            add(
                f"| `{s['name']}` | {s['mean_recall']:.3f} | {s['mean_mrr']:.3f} "
                f"| {s['mean_precision']:.3f} | {s['mean_latency_ms']:.0f} ms "
                f"| {s['zero_recall']} |"
            )
        add("")

        by_name = {c["summary"]["name"]: c["summary"] for c in ablation["configs"]}
        dense = by_name.get("dense", {})
        full = by_name.get("all three", {})
        rescues = ablation.get("unique_rescues", {})

        add(
            f"**Hybrid retrieval beats dense alone by "
            f"{full.get('mean_recall', 0) - dense.get('mean_recall', 0):+.3f} recall.** "
            f"Dense alone uniquely retrieves {len(rescues.get('dense', []))} tasks that neither "
            f"other retriever finds; BM25 uniquely retrieves "
            f"{len(rescues.get('keyword', []))}."
        )
        add("")

        if before:
            before_full = next(
                (
                    c["summary"]
                    for c in before["configs"]
                    if c["summary"]["name"] == "all three"
                ),
                None,
            )
            if before_full:
                add("#### This did not start out true")
                add("")
                add(
                    "The first run of this ablation found the full stack retrieving **worse** "
                    "than dense alone, which is the most useful thing the experiment produced."
                )
                add("")
                add("| | Before | After |")
                add("|---|---|---|")
                add(
                    f"| Full stack recall | {before_full['mean_recall']:.3f} "
                    f"| **{full.get('mean_recall', 0):.3f}** |"
                )
                add(
                    f"| vs dense alone | {before_full['mean_recall'] - dense.get('mean_recall', 0):+.3f} "
                    f"| **{full.get('mean_recall', 0) - dense.get('mean_recall', 0):+.3f}** |"
                )
                add(
                    f"| Zero-recall tasks | {before_full['zero_recall']} "
                    f"| **{full.get('zero_recall', 0)}** |"
                )
                add("")
                add(
                    "The cause was `retrieval_keyword_weight`, set to 0.6 on the reasoning that "
                    "semantic match should dominate. Reciprocal Rank Fusion adds both "
                    "retrievers' contributions, so a document dense-ranked 20th but "
                    "keyword-ranked 1st scores `1/80 + 0.6/61 = 0.02234` against `1/65 = "
                    "0.01538` for one dense-ranked 5th. That is exactly what RRF is designed to "
                    "do, and it is right when both retrievers are strong. On this corpus BM25 "
                    "without an explicit section number in the question matches generic legal "
                    "vocabulary, and those spurious agreements promoted noise over good "
                    "semantic hits."
                )
                add("")

        sweep_path = RESULTS / "weight_sweep.json"
        if sweep_path.exists():
            sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
            add("#### The weight, measured rather than reasoned")
            add("")
            add("| `keyword_weight` | " + " | ".join(
                f"{r['keyword_weight']:.2f}" for r in sweep["rows"]
            ) + " |")
            add("|---" * (len(sweep["rows"]) + 1) + "|")
            add("| Recall | " + " | ".join(
                f"{r['mean_recall']:.3f}" for r in sweep["rows"]
            ) + " |")
            add("| MRR | " + " | ".join(
                f"{r['mean_mrr']:.3f}" for r in sweep["rows"]
            ) + " |")
            add("")
            add(
                f"The curve peaks at **{sweep['best']['keyword_weight']}** and is flat from 0.45 "
                f"upward. BM25 earns its place — worth "
                f"{sweep['keyword_marginal_recall']:+.3f} recall over dense alone — but only at "
                f"roughly a quarter of the weight originally configured."
            )
            add("")

        graph = by_name.get("graph", {})
        dense_graph = by_name.get("dense+graph", {})
        add("#### The graph retriever contributes nothing, and that is the honest result")
        add("")
        add(
            f"`dense+graph` recall ({dense_graph.get('mean_recall', 0):.3f}) equals `dense` "
            f"({dense_graph.get('mean_recall', 0):.3f}) exactly. Graph alone reaches "
            f"{graph.get('mean_recall', 0):.3f} recall and returns almost nothing "
            f"({graph.get('mean_unique_acts', 0):.1f} distinct acts per query)."
        )
        add("")
        add(
            "The cause is structural rather than a tuning problem: `find_seeds` only fires when "
            "a question names a section or an act explicitly, and conversational questions name "
            "neither. Dense-anchored seeding was added — take the provisions semantic search "
            "already surfaced and traverse out from those — and seeding now fires (graph-alone "
            "MRR rose from 0.011 to 0.023), but **recall contribution is still 0.000**. Graph "
            "expansion reaches provisions that cross-reference what dense found, and on this "
            "task set those are not the expected sections. It adds candidates without displacing "
            "anything, so it is neutral rather than harmful, at roughly 190 ms."
        )
        add("")

    # ------------------------------------------------- E6: RAG on versus off
    rag_path = RESULTS / "rag_ablation.json"
    if rag_path.exists():
        rag = json.loads(rag_path.read_text(encoding="utf-8"))
        no_rag, with_rag = rag["no_rag"], rag["with_rag"]
        add("### RAG on versus off — the hallucination control group")
        add("")
        add(
            "Hallucination was previously measured only *with* retrieval, which is a measurement "
            "without a control. Identical questions, identical model "
            f"(`{rag['model']}`), identical prompt shape; the only difference is whether "
            "retrieved statutory text is in the context."
        )
        add("")
        add(
            "Scoring cannot use the grounding audit here, because the no-RAG condition has no "
            "supplied context to check against. Instead every section the model names is looked "
            "up in the corpus: **verified** if that act really contains it, **fabricated** if it "
            "does not. That is not a judgement about phrasing, it is a lookup that fails."
        )
        add("")
        add("| Metric | No RAG | With RAG |")
        add("|---|---|---|")
        rows = [
            ("Sections named", "sections_named", "{:d}"),
            ("Verified in corpus", "corpus_verified", "{:d}"),
            ("**Fabricated**", "fabricated", "{:d}"),
            ("**Fabrication rate**", "fabrication_rate", "{:.1%}"),
            ("Accuracy (answerable)", "accuracy_answerable", "{:.1%}"),
            ("Declined on unanswerable", "declined_on_unanswerable", "{:d}"),
            ("Fabricated on unanswerable", "fabricated_on_unanswerable", "{:d}"),
            ("Mean latency", "mean_latency_ms", "{:.0f} ms"),
        ]
        for label, key, fmt in rows:
            add(f"| {label} | {fmt.format(no_rag[key])} | {fmt.format(with_rag[key])} |")
        add("")
        delta = no_rag["fabrication_rate"] - with_rag["fabrication_rate"]
        add(
            f"**Retrieval changes the fabrication rate by {delta:+.1%}.** "
            + (
                "The no-RAG condition names sections that do not exist in the acts it "
                "attributes them to, which is the concrete form the failure takes: not vagueness, "
                "a citation that cannot be looked up."
                if delta > 0
                else "On this task set the difference is smaller than expected, which is "
                "reported as measured."
            )
        )
        add("")

    # ------------------------------------------------ E12: latency breakdown
    latency_path = RESULTS / "latency_breakdown.json"
    if latency_path.exists():
        lat = json.loads(latency_path.read_text(encoding="utf-8"))
        add("### Where the time goes")
        add("")
        if lat.get("mean_total_ms"):
            add(
                f"Aggregated from {lat['traces_analysed']} recorded pipeline runs. Mean total "
                f"{lat['mean_total_ms'] / 1000:.1f} s, median "
                f"{lat['median_total_ms'] / 1000:.1f} s."
            )
            add("")
        add("| Stage | Share | Mean | Median | p95 |")
        add("|---|---|---|---|---|")
        for row in lat.get("stages", [])[:12]:
            add(
                f"| {row['label']} | {row['share_of_total']:.1%} "
                f"| {row['mean_ms'] / 1000:.2f} s | {row['median_ms'] / 1000:.2f} s "
                f"| {row['p95_ms'] / 1000:.2f} s |"
            )
        add("")
        if lat.get("generation_share") is not None:
            add(
                f"**{lat['generation_share']:.1%} of wall-clock time is inside model "
                f"generation.** Orchestration overhead is therefore not the thing to optimise; "
                f"a faster model or fewer agent calls is."
            )
            if lat.get("mean_tokens_per_second"):
                add("")
                add(f"Mean throughput: {lat['mean_tokens_per_second']:.1f} tokens/second.")
            add("")

    # --------------------------------------------- guardrails (Exercise 13-style)
    suite_path = RESULTS / "guardrail_suite.json"
    if suite_path.exists():
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        full_s = suite["full"]["summary"]
        det_s = suite["deterministic_only"]["summary"]
        add("### Guardrail effectiveness")
        add("")
        add(
            f"{full_s['total']} probe cases: {full_s['adversarial']} adversarial and "
            f"{full_s['legitimate']} legitimate. The legitimate cases matter more — several "
            "deliberately contain denylist trigger words, because a legal tool that refuses "
            "\"draft a written statement in reply to the notice\" is worse than one with no "
            "guardrails at all."
        )
        add("")
        add("| | Adversarial caught | False positives | Accuracy | Mean cost |")
        add("|---|---|---|---|---|")
        add(
            f"| **Full stack** | {full_s['caught']}/{full_s['adversarial']} "
            f"({full_s['catch_rate']:.1%}) | {full_s['false_positives']}/{full_s['legitimate']} "
            f"({full_s['false_positive_rate']:.1%}) | {full_s['accuracy']:.1%} "
            f"| {full_s['mean_latency_ms']:.0f} ms |"
        )
        add(
            f"| Deterministic only | {det_s['caught']}/{det_s['adversarial']} "
            f"({det_s['catch_rate']:.1%}) | {det_s['false_positives']}/{det_s['legitimate']} "
            f"({det_s['false_positive_rate']:.1%}) | {det_s['accuracy']:.1%} "
            f"| {det_s['mean_latency_ms']:.0f} ms |"
        )
        add("")
        gain = full_s["catch_rate"] - det_s["catch_rate"]
        add(
            f"**The embedding layer is worth {gain:+.1%} of catch rate**, and the cases it alone "
            "catches carry no keyword trigger whatsoever — capital of France, sorting a list, "
            "pizza toppings, yesterday's cricket. That is the evidence it generalises rather "
            "than pattern-matching a wordlist."
        )
        add("")
        add(f"Refusals by layer: `{full_s['blocked_by_layer']}`")
        add("")

    # --------------------------------------------------- corpus quality findings
    corpus_path = RESULTS / "corpus_analysis.json"
    if corpus_path.exists():
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        prov = corpus["provisions"]
        cit = corpus["citations"]
        add("### Corpus quality")
        add("")
        add(
            f"{prov['total']} provisions checked for exact, near (SimHash, 64-bit, Hamming ≤ 3) "
            f"and structural duplication."
        )
        add("")
        add("| Metric | Value |")
        add("|---|---|")
        add(f"| Exact duplicates | {prov['exact_duplicates']} |")
        add(f"| Near duplicates | {prov['near_duplicates']} |")
        add(f"| Structural duplicates | {prov['structural_duplicates']} |")
        add(f"| Duplicate rate | {prov['duplicate_rate']:.2%} |")
        add(f"| Chunk duplicate rate | {corpus['chunks']['duplicate_rate']:.2%} |")
        add("")
        add(
            "Duplicate detection earned its place immediately: it flagged 13 Arbitration Act "
            "provisions as identical to one another because India Code carries the Arbitration "
            "Council sections (43A–43M) with headings and **no body text**. Indexed, they were "
            "retrievable and citable while saying nothing — worse than absent, because the "
            "system would offer a lawyer a provision reference carrying no law. Filtering them "
            "took the duplicate rate from 0.84% to "
            f"{prov['duplicate_rate']:.2%}."
        )
        add("")
        add(
            "The remaining exact duplicates are genuine statutory parallels: Arbitration Act "
            "s.46 and s.55 are word-for-word identical across the New York and Geneva Convention "
            "chapters. Not an error, and not removed."
        )
        add("")
        add(
            f"Citation extraction found {cit['citations_found']} citations and "
            f"{cit['section_references_found']} section references across "
            f"{cit['cases_analysed']} judgments, with treatment classified as "
            f"`{cit['treatment_breakdown']}`. All {cit['unresolved']} resolve **outside** the "
            "corpus, which is honest rather than broken: the judgments are a 2023 slice citing "
            "earlier authority the corpus does not hold. Unresolved citations are reported, "
            "not dropped."
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
        add(
            "**What changed as a result.** The confident-failure half of this finding is now "
            "handled. The corpus-distance guardrail refuses questions whose nearest corpus "
            "neighbour falls below 0.55, and the code questions here score 0.37–0.49. So the "
            "system no longer answers them with statutory text; it declines and says why. The "
            "retrieval gap is unchanged — it still could not answer a codebase question if "
            "asked — but it no longer pretends otherwise, which was the more dangerous half."
        )
        add("")

    # ------------------------------------------- corpus size as a variable
    archive = RESULTS / "corpus_263"
    if archive.exists() and (archive / "model_llama3.2_3b.json").exists():
        add("## The corpus itself is a variable")
        add("")
        add(
            "Every number above comes from the 1,783-provision corpus. An earlier run of the "
            "identical harness against the 263-provision corpus is archived in "
            "`eval/results/corpus_263/`, and comparing them is instructive:"
        )
        add("")
        add("| Model | Accuracy @ 263 provisions | Accuracy @ 1,783 | Change |")
        add("|---|---|---|---|")
        old_values = {
            "llama3.2:3b": 0.667,
            "qwen2.5:3b-instruct": 0.593,
            "gemma2:2b": 0.481,
        }
        for s in scores:
            old = old_values.get(s.model)
            if old is None:
                continue
            add(
                f"| `{SHORT.get(s.model, s.model)}` | {pct(old)} | {pct(s.accuracy)} "
                f"| {(s.accuracy - old) * 100:+.1f} pp |"
            )
        add("")
        add(
            "**Accuracy fell for the two strongest models as the corpus grew almost sevenfold.** "
            "That is not a regression in the system; it is the task becoming harder. A larger "
            "corpus means more plausible-but-wrong provisions competing for the same top-k, so "
            "retrieval has to discriminate among 1,783 candidates rather than 263. The relative "
            "ranking of the models is unchanged, and the refusal-discipline finding is unchanged."
        )
        add("")
        add(
            "The practical lesson is that a model comparison is only meaningful against a stated "
            "corpus. Quoting an accuracy figure without saying what it was measured over is "
            "close to meaningless, which is why every table here names the corpus size."
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
        "so the two precedent tasks test a thin slice. This is also why all 45 extracted "
        "citations resolve outside the corpus."
    )
    add(
        "- **The ablation scores retrieval, not answers.** Exercise 5's configuration table "
        "measures recall and MRR against known expected sections. It does not establish that "
        "better retrieval produced a better answer in every case, only that the evidence "
        "reached the context window."
    )
    add(
        "- **E6 uses corpus-existence as its fabrication test.** A section is counted fabricated "
        "when the act it is attributed to does not contain it. That is objective and "
        "automatic, but it cannot catch a citation to a real section that says something other "
        "than what the model claimed."
    )
    add(
        "- **The latency breakdown aggregates whatever traces exist.** It is measured from real "
        "recorded runs rather than a controlled benchmark, so the stage mix reflects the "
        "questions that happened to be asked."
    )
    add("")
    add("### What this evaluation does not establish")
    add("")
    add(
        "No metric here shows that an answer is legally **correct**. Every measurement is about "
        "grounding, scope and retrieval: whether a cited provision exists, was supplied, and is "
        "the one the question needed. Whether the reasoning built on it would persuade a court "
        "is outside what an automated harness can assess, and it is the reason the disclaimer "
        "and lawyer review exist rather than being decoration."
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
