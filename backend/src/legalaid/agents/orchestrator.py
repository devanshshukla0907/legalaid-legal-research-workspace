"""Orchestrator: plans the work, isolates each agent's context, and runs the refinement loop.

Execution plan
--------------
    retrieval (hybrid: dense + BM25 + graph)
        |
    Document Understanding        (only when the matter has documents)
        |
        +---------------+          <-- these two are independent, so they run CONCURRENTLY
        v               v
    Legal Research   Similar Cases
        |               |
        +-------+-------+
                |
        Verification (critic)  --revise--> re-run the failing agent with guidance   [LOOP]
                |
            Drafting
                |
        Verification (critic)  --revise--> re-run drafting with guidance            [LOOP]
                |
          Final assembly

Why concurrency: the two research agents share no data, and each takes roughly 30-60 s on this
hardware at ~12 tok/s. Running them together roughly halves the wall time for the slowest part
of the pipeline.

Why the loop: a small model cites things it was not given. Rather than accept that or hide it,
each agent's output is audited and, if it fails, the agent is re-run with concrete corrective
guidance. Every iteration is recorded, so the UI shows the loop actually happening - including
when a retry fails to fix the problem.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from legalaid.agents import document_understanding as du
from legalaid.agents import legal_research as lr
from legalaid.agents import similar_cases as sc
from legalaid.agents.base import AgentContext, AgentResult, ContextItem
from legalaid.agents.document_understanding import DocumentUnderstandingAgent
from legalaid.agents.drafting import DraftingAgent
from legalaid.agents.intake_router import (
    IntakeRouterAgent,
    RoutingDecision,
    decision_from,
    fallback_decision,
)
from legalaid.agents.legal_research import LegalResearchAgent
from legalaid.agents.similar_cases import SimilarCaseAgent
from legalaid.agents.synthesis import SynthesisAgent, fallback_answer
from legalaid.agents.verification import AuditReport, VerificationAgent, audit
from legalaid.config import get_settings
from legalaid.core.guardrails import (
    GuardrailDecision,
    OutputVerdict,
    record_llm_scope,
    screen_input,
    screen_output,
)
from legalaid.core.logging import get_logger
from legalaid.core.tracing import TraceRecorder, new_trace_id
from legalaid.domain.contracts import RetrievalRequest, RetrievedChunk
from legalaid.domain.enums import (
    AGENT_LABELS,
    AgentKey,
    Collection,
    DraftLength,
    DraftTone,
    DraftType,
    RetrievalMode,
    StageKey,
)
from legalaid.services.retrieval.service import search

logger = get_logger(__name__)

SERVICE = "application-service"


def _merge_upstream(*contexts: AgentContext | None) -> list[ContextItem]:
    """Every distinct evidence item the research agents actually used.

    Drafting and Synthesis reason from findings rather than raw chunks, but they still need the
    provenance chain - and the grounding audit needs each item's real source label to verify a
    cited section or case against. Passing ids alone loses that, which is what made the audit
    silently pass anything those two agents wrote.
    """
    merged: dict[str, ContextItem] = {}
    for context in contexts:
        if context is None:
            continue
        for item in context.included:
            # A case label is the more useful one to keep if the same chunk appears twice.
            if item.chunk_id not in merged or item.kind == "case":
                merged[item.chunk_id] = item
    return [merged[key] for key in sorted(merged)]


@dataclass(slots=True)
class LoopIteration:
    """One pass of the refinement loop, recorded for the UI."""

    agent: str
    iteration: int
    audit: dict[str, Any]
    llm_review: dict[str, Any] | None
    action: str
    guidance: str | None


@dataclass(slots=True)
class OrchestrationResult:
    trace_id: str
    question: str
    retrieval: Any
    facts: dict[str, Any]
    legal: dict[str, Any]
    cases: dict[str, Any]
    draft: dict[str, Any]
    # The lawyer-facing answer written by the synthesis agent.
    answer: dict[str, Any] = field(default_factory=dict)
    routing: RoutingDecision | None = None
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    loop_iterations: list[LoopIteration] = field(default_factory=list)
    plan: list[dict[str, Any]] = field(default_factory=list)
    total_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    # Guardrail records, so the UI can show which layer allowed or refused the request and
    # how well grounded the answer turned out to be.
    guardrail: GuardrailDecision | None = None
    output_guardrail: OutputVerdict | None = None


class Orchestrator:
    """Central controller. Decides what runs, with what context, in what order."""

    def __init__(
        self, *, max_loop_iterations: int | None = None, model: str | None = None
    ) -> None:
        settings = get_settings()
        self.max_loops = (
            max_loop_iterations
            if max_loop_iterations is not None
            else settings.orchestrator_max_loop_iterations
        )
        # A single model override flows to every agent, so the UI can run the same question
        # through different LLMs and compare.
        self.model = model
        self.router = IntakeRouterAgent(model=model)
        self.legal_agent = LegalResearchAgent(model=model)
        self.case_agent = SimilarCaseAgent(model=model)
        self.draft_agent = DraftingAgent(model=model)
        self.doc_agent = DocumentUnderstandingAgent(model=model)
        self.synthesis_agent = SynthesisAgent(model=model)
        self.critic = VerificationAgent(model=model)

    # ------------------------------------------------------------------ public
    async def run(
        self,
        *,
        question: str,
        matter_id: str | None = None,
        matter_title: str = "",
        draft_type: DraftType | None = None,
        tone: DraftTone = DraftTone.FORMAL,
        length: DraftLength = DraftLength.STANDARD,
        instructions: str = "",
        enable_drafting: bool | None = None,
        enable_verification: bool = True,
        pasted_text: str = "",
        document_summaries: list[str] | None = None,
    ) -> OrchestrationResult:
        started = time.perf_counter()
        recorder = TraceRecorder(
            new_trace_id(),
            question=question,
            mode="multi_agent",
            matter_id=matter_id,
        )
        recorder.open()
        trace_id = recorder.trace_id

        result = OrchestrationResult(
            trace_id=trace_id,
            question=question,
            retrieval=None,
            facts={},
            legal={},
            cases={},
            draft={},
        )

        full_input = "\n\n".join(p for p in (question, pasted_text) if p.strip())

        with recorder.stage(
            StageKey.USER_INPUT, service=SERVICE, input={"question": question}
        ) as handle:
            handle.set_output(
                question=question,
                pasted_chars=len(pasted_text),
                matter_id=matter_id,
                model=self.model or get_settings().llm_model,
            )

        # -------------------------------------------------- input guardrails
        # Deterministic layers run before the router, so an out-of-scope request never costs
        # an LLM call, and a router failure can no longer wave everything through.
        settings = get_settings()
        guard = GuardrailDecision(allowed=True)
        if settings.guardrails_enabled:
            with recorder.stage(
                StageKey.INPUT_GUARDRAIL,
                service=SERVICE,
                input={"chars": len(full_input)},
            ) as handle:
                guard = await screen_input(full_input)
                handle.set_output(**guard.to_dict())
            result.guardrail = guard
            if not guard.allowed:
                return self._refuse(
                    result,
                    recorder,
                    started,
                    reason=guard.message,
                    blocked_by=guard.blocked_by.value if guard.blocked_by else "guardrail",
                )
            for warning in guard.warnings:
                result.warnings.append(warning.reason)

        # ------------------------------------------- agent 0: intake routing
        # This runs BEFORE retrieval on purpose: it decides which agents are needed and
        # rewrites the input into the text that will actually be embedded.
        decision = await self._route(
            full_input, document_summaries, trace_id, recorder, result
        )
        result.routing = decision

        # Layer 6: the router's scope verdict, recorded alongside the deterministic layers so
        # the trace shows which layer actually made the call.
        if settings.guardrails_enabled:
            guard = record_llm_scope(
                guard,
                is_legal_matter=decision.is_legal_matter,
                reasoning=decision.reasoning,
            )
            result.guardrail = guard

        if not decision.is_legal_matter:
            # Refusing to invent a legal question is the correct behaviour here.
            return self._refuse(
                result,
                recorder,
                started,
                reason=(
                    "I could not identify a legal issue in what you sent, so no statutory or "
                    f"case-law research was carried out.\n\nThe router's reading of your input: "
                    f"{decision.reasoning}\n\nIf this is a legal matter, try naming the dispute, "
                    "the relevant conduct and the remedy you are considering, or attach the "
                    "document."
                ),
                blocked_by="llm_scope",
                routing=decision,
            )

        # Explicit arguments override the router; otherwise the router decides.
        do_draft = decision.needs_drafting if enable_drafting is None else enable_drafting
        chosen_draft_type = draft_type or decision.draft_type

        # ---------------------------------------------------------- retrieval
        mode = self._mode_for(decision)
        with recorder.stage(
            StageKey.CONTEXT_CONSTRUCTION,
            service="retrieval-service",
            input={
                "original_input": full_input[:600],
                "search_query": decision.search_query,
                "mode": mode.value,
                "query_was_rewritten": decision.search_query.strip() != full_input.strip(),
            },
        ) as handle:
            retrieval = await search(
                RetrievalRequest(
                    trace_id=trace_id,
                    # The rewritten query is what gets embedded, not the raw input.
                    question=decision.search_query,
                    mode=mode,
                    matter_id=matter_id,
                )
            )
            handle.set_output(
                retrieved=len(retrieval.chunks),
                context_tokens=retrieval.context_tokens,
                collections={
                    t.collection.value: t.returned for t in retrieval.searches if t.returned
                },
            )
        result.retrieval = retrieval
        chunks = retrieval.chunks

        plan = self._plan(chunks, decision=decision, enable_verification=enable_verification)
        result.plan = plan
        with recorder.stage(StageKey.ORCHESTRATION_PLAN, service=SERVICE) as handle:
            handle.set_output(
                steps=plan,
                retrieved_chunks=len(chunks),
                agents_requested=decision.agents_requested,
            )

        # ------------------------------------------ agent 4: document understanding
        doc_chunks = [c for c in chunks if c.collection is Collection.USER_DOCUMENTS]
        if doc_chunks and decision.needs_document_facts:
            context = du.prepare(
                self.doc_agent,
                question=question,
                chunks=chunks,
                matter_title=matter_title,
            )
            facts_result = await self._run_with_loop(
                self.doc_agent, context, trace_id, recorder,
                StageKey.DOCUMENT_UNDERSTANDING, result, enable_verification,
            )
            result.agent_results["document_understanding"] = facts_result
            result.facts = facts_result.output
        else:
            recorder.skip_stage(
                StageKey.DOCUMENT_UNDERSTANDING,
                service=SERVICE,
                reason=(
                    "no matter documents available"
                    if not doc_chunks
                    else "router determined document facts were not required"
                ),
            )

        # -------------------- agents 1 and 2 run CONCURRENTLY (no dependency)
        tasks: list[Any] = []
        keys: list[str] = []
        legal_context = case_context = None

        if decision.needs_law:
            legal_context = lr.prepare(
                self.legal_agent,
                question=decision.search_query,
                chunks=chunks,
                matter_summary=str(result.facts.get("factual_summary") or ""),
            )
            tasks.append(
                self._run_with_loop(
                    self.legal_agent, legal_context, trace_id, recorder,
                    StageKey.LEGAL_RESEARCH_AGENT, result, enable_verification,
                )
            )
            keys.append("legal_research")
        else:
            recorder.skip_stage(
                StageKey.LEGAL_RESEARCH_AGENT,
                service="agent-service",
                reason="router determined statutory research was not required",
            )

        if decision.needs_cases:
            case_context = sc.prepare(
                self.case_agent,
                question=decision.search_query,
                chunks=chunks,
                matter_facts=str(result.facts.get("factual_summary") or ""),
                legal_issues=list(result.facts.get("potential_issues") or [])
                or decision.legal_concepts,
            )
            tasks.append(
                self._run_with_loop(
                    self.case_agent, case_context, trace_id, recorder,
                    StageKey.SIMILAR_CASE_AGENT, result, enable_verification,
                )
            )
            keys.append("similar_cases")
        else:
            recorder.skip_stage(
                StageKey.SIMILAR_CASE_AGENT,
                service="agent-service",
                reason="router determined precedent search was not required",
            )

        if tasks:
            # Concurrent: these two agents share no data, and each costs 25-60 s.
            outcomes = await asyncio.gather(*tasks)
            for key, outcome in zip(keys, outcomes, strict=True):
                result.agent_results[key] = outcome
                if key == "legal_research":
                    result.legal = outcome.output
                else:
                    result.cases = outcome.output

        # ------------------------------------------------ agent 3: drafting (dependent)
        if do_draft:
            upstream_items = _merge_upstream(legal_context, case_context)
            draft_context = self.draft_agent.build_context(
                question=question,
                verified_facts=result.facts,
                legal_findings=result.legal,
                case_findings=result.cases,
                draft_type=chosen_draft_type,
                tone=tone,
                length=length,
                instructions=instructions,
                upstream_items=upstream_items,
            )
            draft_result = await self._run_with_loop(
                self.draft_agent, draft_context, trace_id, recorder,
                StageKey.DRAFTING_AGENT, result, enable_verification,
            )
            result.agent_results["drafting"] = draft_result
            result.draft = draft_result.output
        else:
            recorder.skip_stage(
                StageKey.DRAFTING_AGENT, service=SERVICE, reason="drafting not requested"
            )

        # ------------------------------------------ agent 6: synthesis (the actual answer)
        # Always runs. Structured findings are not an answer, and the lawyer asked a question.
        synthesis_context = self.synthesis_agent.build_context(
            question=question or full_input,
            facts=result.facts,
            legal_findings=result.legal,
            case_findings=result.cases,
            draft=result.draft,
            upstream_items=_merge_upstream(legal_context, case_context),
        )
        synthesis_result = await self._run_with_loop(
            self.synthesis_agent, synthesis_context, trace_id, recorder,
            StageKey.SYNTHESIS_AGENT, result, enable_verification,
        )
        result.agent_results["synthesis"] = synthesis_result
        if synthesis_result.parse_ok and synthesis_result.output.get("answer_markdown"):
            result.answer = synthesis_result.output

            # ---------------------------------------------- output guardrail
            # The last thing before the answer reaches the lawyer: score how much of it rests
            # on retrieved evidence, and refuse to present a confident answer that rests on
            # nothing at all.
            if get_settings().guardrails_enabled:
                last_audit = next(
                    (
                        i.audit
                        for i in reversed(result.loop_iterations)
                        if i.agent == AGENT_LABELS.get(AgentKey.SYNTHESIS, "synthesis")
                    ),
                    None,
                )
                with recorder.stage(StageKey.OUTPUT_GUARDRAIL, service=SERVICE) as handle:
                    verdict = screen_output(
                        answer=result.answer,
                        audit_report=last_audit,
                        allowed_chunk_ids=synthesis_context.allowed_chunk_ids,
                    )
                    handle.set_output(**verdict.to_dict())
                result.output_guardrail = verdict
                if not verdict.allowed:
                    # Downgrade rather than discard: the reasoning may still be useful, but it
                    # must not be presented as confident, sourced advice.
                    result.answer["confidence"] = "low"
                    result.answer["groundedness_warning"] = verdict.message
                    result.warnings.append(verdict.message)
        else:
            # The lawyer must never see an empty result, so assemble a plain summary from the
            # structured findings and mark it clearly as not model-written.
            result.answer = fallback_answer(
                question=question or full_input,
                legal=result.legal,
                cases=result.cases,
            )

        result.total_ms = (time.perf_counter() - started) * 1000
        for key, agent_result in result.agent_results.items():
            if not agent_result.parse_ok:
                result.warnings.append(
                    f"{AGENT_LABELS.get(agent_result.agent_key, key)} did not return valid "
                    f"JSON after {agent_result.attempts} attempts ({agent_result.parse_error})"
                )

        with recorder.stage(StageKey.FINAL_RESPONSE, service=SERVICE) as handle:
            handle.set_output(
                total_ms=result.total_ms,
                agents_run=list(result.agent_results),
                loop_iterations=len(result.loop_iterations),
                warnings=result.warnings,
            )
        recorder.close(
            status="success" if not result.warnings else "partial",
            answer={
                "routing": decision.to_dict(),
                "synthesis": result.answer,
                "facts": result.facts,
                "legal": result.legal,
                "cases": result.cases,
                "draft": result.draft,
            },
            summary={
                "total_ms": result.total_ms,
                "plan": result.plan,
                "routing": decision.to_dict(),
                "warnings": result.warnings,
                "loop_iterations": [
                    {
                        "agent": i.agent,
                        "iteration": i.iteration,
                        "action": i.action,
                        "problems": i.audit.get("problem_count"),
                    }
                    for i in result.loop_iterations
                ],
            },
        )

        logger.info(
            "orchestration_complete",
            trace_id=trace_id,
            total_ms=round(result.total_ms, 1),
            agents=len(result.agent_results),
            loops=len(result.loop_iterations),
        )
        return result

    # ----------------------------------------------------------------- internals
    async def _route(
        self,
        user_input: str,
        document_summaries: list[str] | None,
        trace_id: str,
        recorder: TraceRecorder,
        result: OrchestrationResult,
    ) -> RoutingDecision:
        """Classify the request and rewrite it into a retrieval query."""
        context = self.router.build_context(
            user_input=user_input, document_summaries=document_summaries
        )
        with recorder.stage(
            StageKey.INTAKE_CLASSIFICATION,
            service="agent-service",
            input={"input_chars": len(user_input), "documents": len(document_summaries or [])},
        ) as handle:
            router_result = await self.router.run(
                trace_id=trace_id, context=context, max_attempts=2
            )
            if router_result.parse_ok and not self.router.validate(
                router_result.output, context
            ):
                decision = decision_from(router_result.output, user_input=user_input)
            else:
                # Never let a router failure stall the pipeline. Scope is taken from the
                # deterministic guardrail rather than assumed: those layers already ran and
                # allowed this request, which is a more trustworthy signal than a generation
                # that just failed to parse.
                decision = fallback_decision(
                    user_input,
                    has_documents=bool(document_summaries),
                    is_legal_matter=(
                        result.guardrail.allowed if result.guardrail is not None else True
                    ),
                )
            handle.set_output(
                task=decision.task,
                input_kind=decision.input_kind,
                document_type=decision.document_type,
                agents_requested=decision.agents_requested,
                is_legal_matter=decision.is_legal_matter,
                used_fallback=decision.used_fallback,
                reasoning=decision.reasoning,
            )
        result.agent_results["intake_router"] = router_result

        # The rewrite is its own stage so the UI can show original vs embedded text.
        with recorder.stage(
            StageKey.QUERY_REWRITE,
            service="agent-service",
            input={"original": user_input[:1000]},
        ) as handle:
            handle.set_output(
                original=user_input[:1000],
                search_query=decision.search_query,
                legal_concepts=decision.legal_concepts,
                rewritten=decision.search_query.strip() != user_input.strip(),
                original_chars=len(user_input),
                query_chars=len(decision.search_query),
            )
        return decision

    @staticmethod
    def _mode_for(decision: RoutingDecision) -> RetrievalMode:
        """Search only the collections the request actually needs.

        `needs_document_facts` has to widen the mode, not just request an agent: LAW_SEARCH and
        CASE_SEARCH do not include USER_DOCUMENTS, so a narrow mode used to leave `doc_chunks`
        empty and the Document Understanding stage was skipped with the misleading reason "no
        matter documents available" even though the router had explicitly asked for it.
        COMBINED is the only mode that reaches user documents.
        """
        if decision.needs_document_facts:
            return RetrievalMode.COMBINED_LEGAL_RESEARCH
        if decision.needs_law and decision.needs_cases:
            return RetrievalMode.COMBINED_LEGAL_RESEARCH
        if decision.needs_cases:
            return RetrievalMode.CASE_SEARCH
        if decision.needs_law:
            return RetrievalMode.LAW_SEARCH
        return RetrievalMode.COMBINED_LEGAL_RESEARCH

    def _plan(
        self,
        chunks: list[RetrievedChunk],
        *,
        decision: RoutingDecision,
        enable_verification: bool,
    ) -> list[dict[str, Any]]:
        """The execution plan, recorded so the Pipeline page can render it."""
        steps: list[dict[str, Any]] = [
            {
                "step": "intake_router",
                "reason": f"classified as {decision.task}",
                "parallel": False,
            },
            {
                "step": "query_rewrite",
                "reason": "focus the text that gets embedded",
                "parallel": False,
            },
            {
                "step": "retrieval",
                "mode": "hybrid (dense + keyword + graph)",
                "parallel": False,
            },
        ]
        if decision.needs_document_facts:
            steps.append(
                {
                    "step": "document_understanding",
                    "reason": "input includes a document",
                    "parallel": False,
                }
            )
        concurrent = [
            name
            for name, wanted in (
                ("legal_research", decision.needs_law),
                ("similar_cases", decision.needs_cases),
            )
            if wanted
        ]
        if concurrent:
            steps.append(
                {
                    "step": " + ".join(concurrent),
                    "reason": (
                        "independent of each other"
                        if len(concurrent) > 1
                        else "requested by router"
                    ),
                    "parallel": len(concurrent) > 1,
                }
            )
        if enable_verification:
            steps.append({"step": "verification", "reason": "audit grounding", "parallel": False})
        if decision.needs_drafting:
            steps.append(
                {"step": "drafting", "reason": "depends on research output", "parallel": False}
            )
        steps.append(
            {
                "step": "synthesis",
                "reason": "write the answer to the advocate",
                "parallel": False,
            }
        )
        return steps

    async def _run_with_loop(
        self,
        agent: Any,
        context: AgentContext,
        trace_id: str,
        recorder: TraceRecorder,
        stage: StageKey,
        result: OrchestrationResult,
        enable_verification: bool,
    ) -> AgentResult:
        """Run an agent, audit it, and retry with guidance while it fails. Loop engineering."""
        agent_name = AGENT_LABELS.get(agent.agent_key, agent.agent_key.value)
        guidance: str | None = None
        agent_result: AgentResult | None = None

        for iteration in range(1, self.max_loops + 1):
            with recorder.stage(
                stage,
                service="agent-service",
                input={
                    "agent": agent_name,
                    "attempt": iteration,
                    "included_chunks": len(context.included),
                    "excluded_chunks": len(context.excluded),
                    "prompt": agent.prompt.label,
                    "correction_applied": bool(guidance),
                },
            ) as handle:
                agent_result = await agent.run(
                    trace_id=trace_id, context=context, retry_note=guidance
                )
                handle.set_output(
                    status=agent_result.status,
                    model=agent_result.model,
                    input_tokens=agent_result.input_tokens,
                    output_tokens=agent_result.output_tokens,
                    latency_ms=agent_result.latency_ms,
                    parse_ok=agent_result.parse_ok,
                )

            structural = agent.validate(agent_result.output, context) if agent_result.parse_ok else [
                f"output was not valid JSON: {agent_result.parse_error}"
            ]
            report = audit(
                agent_name=agent_name,
                output=agent_result.output,
                context=context,
                structural_problems=structural,
                prose_fields=getattr(agent, "prose_fields", ()),
            )

            # Recorded on every pass, not only on failure. A guardrail that is invisible when
            # it succeeds cannot be shown to be working, and "no refinement loop appeared"
            # reads identically to "no audit ran".
            with recorder.stage(
                StageKey.CITATION_VALIDATION,
                service=SERVICE,
                input={
                    "agent": agent_name,
                    "iteration": iteration,
                    "allowed_chunk_ids": len(context.allowed_chunk_ids),
                    "prose_fields_scanned": list(getattr(agent, "prose_fields", ())),
                },
            ) as audit_handle:
                audit_handle.set_output(
                    passed=report.passed,
                    problem_count=report.problem_count,
                    audit=report.to_dict(),
                    enforced=enable_verification,
                )

            if report.passed:
                result.loop_iterations.append(
                    LoopIteration(
                        agent=agent_name,
                        iteration=iteration,
                        audit=report.to_dict(),
                        llm_review=None,
                        action="accepted",
                        guidance=None,
                    )
                )
                break

            if not enable_verification:
                # The audit still ran and still found problems; the caller asked us not to act
                # on them. Labelling that "accepted" would be a lie, so it is recorded as
                # explicitly unverified and the problems are kept visible.
                result.loop_iterations.append(
                    LoopIteration(
                        agent=agent_name,
                        iteration=iteration,
                        audit=report.to_dict(),
                        llm_review=None,
                        action="accepted_unverified",
                        guidance=report.guidance() or None,
                    )
                )
                result.warnings.append(
                    f"{agent_name}: verification was disabled for this run, so "
                    f"{report.problem_count} grounding problem(s) were left uncorrected."
                )
                break

            last_iteration = iteration >= self.max_loops
            action = "accepted_with_warnings" if last_iteration else "retry"

            # ---- LLM critic: the judgement calls a set operation cannot make.
            # Runs only when the deterministic layer has already found something, because it
            # costs a full generation and the deterministic result is authoritative on
            # citation membership anyway.
            llm_review = await self._llm_review(
                reviewed_agent=agent_name,
                reviewed_output=agent_result.output,
                reviewed_context=context,
                trace_id=trace_id,
                recorder=recorder,
            )

            result.loop_iterations.append(
                LoopIteration(
                    agent=agent_name,
                    iteration=iteration,
                    audit=report.to_dict(),
                    llm_review=llm_review,
                    action=action,
                    guidance=report.guidance() or None,
                )
            )
            with recorder.stage(
                StageKey.REFINEMENT_LOOP,
                service=SERVICE,
                input={"agent": agent_name, "iteration": iteration},
            ) as loop_handle:
                loop_handle.set_output(
                    agent=agent_name,
                    iteration=iteration,
                    problems=report.problem_count,
                    action=action,
                    audit=report.to_dict(),
                    guidance=report.guidance() or None,
                )
            logger.info(
                "refinement_loop",
                agent=agent_name,
                iteration=iteration,
                problems=report.problem_count,
                action=action,
            )
            if last_iteration:
                # Unsupported citations are stripped rather than presented as authority.
                self._strip_unsupported(agent_result.output, report)
                break

            # The deterministic guidance leads because it is exact. The critic's guidance is
            # appended for the things it can see and set membership cannot: overclaiming,
            # conclusions beyond the evidence.
            guidance = report.guidance()
            critic_note = (llm_review or {}).get("retry_guidance")
            if critic_note and str(critic_note).strip():
                guidance = f"{guidance} Reviewer also noted: {str(critic_note).strip()}"

        assert agent_result is not None
        return agent_result

    def _refuse(
        self,
        result: OrchestrationResult,
        recorder: TraceRecorder,
        started: float,
        *,
        reason: str,
        blocked_by: str,
        routing: RoutingDecision | None = None,
    ) -> OrchestrationResult:
        """Decline a request without running the pipeline.

        The user still gets a real reply explaining what happened and which check refused it -
        an empty result would leave them guessing, and naming the layer makes the guardrail
        demonstrable rather than mysterious.
        """
        recorder.skip_stage(
            StageKey.CONTEXT_CONSTRUCTION,
            service="retrieval-service",
            reason=f"refused by {blocked_by}",
        )
        result.warnings.append(f"Request refused by the {blocked_by} guardrail.")
        result.answer = {
            "direct_answer": "I cannot answer this with the material available.",
            "answer_markdown": reason,
            "key_points": [],
            "confidence": "low",
            "refused": True,
            "refused_by": blocked_by,
        }
        result.total_ms = (time.perf_counter() - started) * 1000
        payload: dict[str, Any] = {"synthesis": result.answer}
        summary: dict[str, Any] = {
            "refused_by": blocked_by,
            "total_ms": result.total_ms,
        }
        if routing is not None:
            payload["routing"] = routing.to_dict()
            summary["routing"] = routing.to_dict()
        if result.guardrail is not None:
            payload["guardrail"] = result.guardrail.to_dict()
            summary["guardrail"] = result.guardrail.to_dict()
        recorder.close(status="refused", answer=payload, summary=summary)
        logger.info("request_refused", blocked_by=blocked_by, trace_id=result.trace_id)
        return result

    async def _llm_review(
        self,
        *,
        reviewed_agent: str,
        reviewed_output: dict[str, Any],
        reviewed_context: AgentContext,
        trace_id: str,
        recorder: TraceRecorder,
    ) -> dict[str, Any] | None:
        """Second-tier review by the critic agent. Never allowed to break the pipeline."""
        try:
            critic_context = self.critic.build_context(
                question=reviewed_context.question,
                reviewed_agent=reviewed_agent,
                reviewed_output=reviewed_output,
                reviewed_context=reviewed_context,
            )
            with recorder.stage(
                StageKey.VERIFICATION_AGENT,
                service="agent-service",
                input={
                    "reviewed_agent": reviewed_agent,
                    "allowed_chunk_ids": len(reviewed_context.allowed_chunk_ids),
                },
            ) as handle:
                critic_result = await self.critic.run(
                    trace_id=trace_id, context=critic_context, max_attempts=1
                )
                review = critic_result.output if critic_result.parse_ok else None
                handle.set_output(
                    parse_ok=critic_result.parse_ok,
                    verdict=(review or {}).get("verdict"),
                    unsupported_claims=len((review or {}).get("unsupported_claims") or []),
                    overconfident=len((review or {}).get("overconfident_statements") or []),
                    latency_ms=critic_result.latency_ms,
                )
            return review
        except Exception as exc:  # noqa: BLE001
            # A critic failure must never cost the user their answer; the deterministic audit
            # has already done the load-bearing work.
            logger.warning(
                "llm_review_failed", agent=reviewed_agent, error=f"{type(exc).__name__}: {exc}"
            )
            return None

    @staticmethod
    def _strip_unsupported(output: dict[str, Any], report: AuditReport) -> None:
        """Remove ungrounded citations that survived the loop.

        Better to return fewer, verifiable authorities than to hand a lawyer a citation the
        system cannot substantiate.
        """
        bad_ids = set(report.unsupported_citations)
        bad_cases = {c.lower() for c in report.unsupported_cases}
        if not bad_ids and not bad_cases:
            return

        for key in ("sections", "cases"):
            entries = output.get(key)
            if not isinstance(entries, list):
                continue
            output[key] = [
                e
                for e in entries
                if not (
                    isinstance(e, dict)
                    and (
                        e.get("chunk_id") in bad_ids
                        or str(e.get("case_title") or "").lower() in bad_cases
                    )
                )
            ]
        notes = output.setdefault("uncertainties", [])
        if isinstance(notes, list):
            notes.append(
                "Some generated citations could not be verified against the retrieved "
                "evidence and were removed."
            )
