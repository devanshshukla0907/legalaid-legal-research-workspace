"""Quick manual check of the guardrail stack, including the corpus-distance layer."""

from __future__ import annotations

import asyncio

from legalaid.core.guardrails import screen_input

CASES = [
    ("What does Section 73 of the Indian Contract Act provide?", "legal, in corpus"),
    ("Under the Contract Act when is an agreement without consideration valid?", "legal, in corpus"),
    ("Which files handle agent context isolation in this repository?", "code question"),
    ("What are the GST registration thresholds for an e-commerce operator?", "legal, NOT in corpus"),
    ("What is the capital of France?", "general knowledge"),
    ("How do I sort a list in Python without using sorted()?", "code question"),
    ("What is the best pizza topping?", "nonsense"),
]


async def main() -> None:
    print(f"{'verdict':<8}{'corpus':<8}{'sim':>7}  {'note':<24}question")
    print("-" * 104)
    for text, note in CASES:
        decision = await screen_input(text)
        layer = next(
            (item for item in decision.layers if item.layer.value == "corpus_distance"), None
        )
        similarity = layer.evidence.get("best_similarity") if layer else None
        verdict = layer.verdict.value if layer else "n/a"
        allowed = "ALLOW" if decision.allowed else "BLOCK"
        blocked_by = decision.blocked_by.value if decision.blocked_by else ""
        sim_text = f"{similarity:.3f}" if isinstance(similarity, float) else "-"
        print(f"{allowed:<8}{verdict:<8}{sim_text:>7}  {note:<24}{text[:44]}")
        if not decision.allowed:
            print(f"{'':<24}blocked by {blocked_by}: {decision.message[:78]}")


if __name__ == "__main__":
    asyncio.run(main())
