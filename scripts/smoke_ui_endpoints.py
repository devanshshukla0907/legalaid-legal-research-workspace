"""Verify every endpoint the UI depends on returns usable data."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def check(path: str, probe) -> None:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=90) as response:
            payload = json.loads(response.read())
        print(f"  OK   {path:<44} {probe(payload)}")
    except urllib.error.HTTPError as exc:
        print(f"  FAIL {path:<44} HTTP {exc.code}: {exc.read()[:180].decode(errors='replace')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {path:<44} {type(exc).__name__}: {exc}")


def main() -> None:
    print("UI endpoint check")
    check("/api/system/health", lambda p: f"status={p['status']} vectors={p['retrievers']['dense']['vectors']}")
    check("/api/models", lambda p: f"{len(p['installed'])} models, default={p['default_llm_model']}")
    check("/api/metrics", lambda p: f"llm calls={p['llm']['calls']} traces={p['pipeline']['traces']}")
    check("/api/agents", lambda p: f"{len(p['agents'])} agents")
    check("/api/prompts", lambda p: f"{len(p['prompts'])} prompts")
    check("/api/stages", lambda p: f"{len(p['stages'])} stage kinds")
    check("/api/traces?limit=5", lambda p: f"{len(p['traces'])} traces")
    check("/api/chunks?limit=5", lambda p: f"total={p['total']}")
    check("/api/embeddings?limit=5", lambda p: f"total={p['total']} models={p['models']}")
    check("/api/knowledge", lambda p: f"{len(p['acts'])} acts, {len(p['cases'])} cases, {len(p['sources'])} sources")
    check("/api/knowledge/graph?limit=50", lambda p: f"{len(p['nodes'])} nodes, {len(p['edges'])} edges")
    check("/api/knowledge/provisions?act=Indian+Contract+Act%2C+1872", lambda p: f"{len(p['provisions'])} provisions")
    check("/api/matters", lambda p: f"{len(p['matters'])} matters")
    check("/api/documents", lambda p: f"{len(p['documents'])} documents")

    # Follow a chunk and a trace, which is what the inspector pages do.
    with urllib.request.urlopen(f"{BASE}/api/chunks?limit=1", timeout=60) as response:
        chunk_id = json.loads(response.read())["chunks"][0]["chunk_id"]
    check(f"/api/chunks/{chunk_id}", lambda p: f"{p['char_count']} chars, embedding={bool(p['embedding'])}, vector_dims={p['vector_dimensions']}")

    with urllib.request.urlopen(f"{BASE}/api/traces?limit=1", timeout=60) as response:
        traces = json.loads(response.read())["traces"]
    if traces:
        check(
            f"/api/traces/{traces[0]['trace_id']}",
            lambda p: f"{len(p['stages'])} stages, {len(p['agent_runs'])} agents, {len(p['model_runs'])} model runs",
        )

    with urllib.request.urlopen(f"{BASE}/api/documents", timeout=60) as response:
        documents = json.loads(response.read())["documents"]
    if documents:
        check(
            f"/api/documents/{documents[0]['id']}",
            lambda p: f"{len(p['pages'])} pages, {len(p['chunks'])} chunks",
        )


if __name__ == "__main__":
    main()
