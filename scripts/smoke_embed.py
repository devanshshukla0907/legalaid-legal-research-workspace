"""Verify the OllamaClient embedding path: batching, dimensions and semantic sanity."""

from __future__ import annotations

import asyncio
import math
import time

from legalaid.config import get_settings
from legalaid.services.llm.ollama_client import OllamaClient


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


async def main() -> None:
    settings = get_settings()
    client = OllamaClient()
    print("model      :", settings.embedding_model)
    print("batch size :", settings.embedding_batch_size)

    texts = [
        "Compensation for loss or damage caused by breach of contract.",
        "The party who suffers by a breach is entitled to receive compensation.",
        "Punishment for criminal conspiracy under the penal code.",
        "Procedure for admission of electronic records as evidence.",
    ]
    started = time.perf_counter()
    vectors = await client.embed(texts)
    elapsed = time.perf_counter() - started

    print(f"returned   : {len(vectors)} vectors of {len(vectors[0])} dims")
    print(f"latency    : {elapsed*1000:.0f} ms total, {elapsed/len(texts)*1000:.0f} ms/text")
    assert len(vectors) == len(texts)
    assert all(len(v) == settings.embedding_dimensions for v in vectors)

    # Semantic sanity: the two contract-compensation lines must be closer to each other
    # than either is to the criminal-conspiracy line. If this fails, retrieval cannot work.
    same_topic = cosine(vectors[0], vectors[1])
    cross_topic = cosine(vectors[0], vectors[2])
    print(f"\ncosine(contract, contract-ish) = {same_topic:.4f}")
    print(f"cosine(contract, conspiracy)   = {cross_topic:.4f}")
    assert same_topic > cross_topic, "embeddings are not semantically ordered"
    print("\nsemantic ordering OK")


if __name__ == "__main__":
    asyncio.run(main())
