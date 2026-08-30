"""Graph RAG — retrieval by traversing legal relationships, with no vectors involved.

This is a third, structurally different retriever. Dense search finds text that *means*
something similar; BM25 finds text containing the same *words*. Neither can answer "which
judgments interpret this section?", because that is a question about relationships, not
similarity.

Legal knowledge is naturally a graph:

    Act ──contains──> Section ──cited_by──> Judgment ──considers──> other Sections
                         │                      │
                         └──referenced_by───────┴──> Section (cross-references)

The graph is built from real evidence already captured during ingestion:
  * statute cross-references parsed out of provision text ("as provided in section 74")
  * the `legal_provisions` a judgment's headnote says it considered
  * section tokens appearing in judgment text

Traversal is purely symbolic - no embeddings are used - which makes it a genuine complement to
the vector path and lets the UI show a different kind of evidence: a citation chain rather than
a similarity score.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field

from sqlalchemy import select

from legalaid.core.logging import get_logger
from legalaid.db.base import session_scope
from legalaid.db.models import Case, Chunk, LegalProvision
from legalaid.domain.enums import Collection

logger = get_logger(__name__)

# "section 73", "s. 73", "sections 73 and 74", "Order VII Rule 11"
_SECTION_REF_RE = re.compile(
    r"\b(?:section|sections|sec\.?|s\.)\s*(\d+[A-Z]{0,2})(?:\s*(?:and|,|to)\s*(\d+[A-Z]{0,2}))?",
    re.I,
)
_ORDER_RULE_RE = re.compile(r"\bOrder\s+([IVXLCDM]+)\s+Rule\s+(\d+[A-Z]{0,2})", re.I)


@dataclass(slots=True)
class GraphNode:
    key: str
    kind: str  # "provision" | "case" | "act"
    label: str
    chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    evidence: str | None = None


@dataclass(slots=True)
class GraphPath:
    """How a node was reached, so the UI can show the citation chain."""

    nodes: list[str]
    relations: list[str]

    def describe(self, graph: LegalGraph) -> str:
        parts: list[str] = []
        for index, key in enumerate(self.nodes):
            node = graph.nodes.get(key)
            parts.append(node.label if node else key)
            if index < len(self.relations):
                parts.append(f" --{self.relations[index]}--> ")
        return "".join(parts)


@dataclass(slots=True)
class GraphResult:
    node: GraphNode
    hops: int
    path: GraphPath
    relation: str


class LegalGraph:
    """A small in-memory knowledge graph over provisions and judgments."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.out: dict[str, list[GraphEdge]] = defaultdict(list)
        self.incoming: dict[str, list[GraphEdge]] = defaultdict(list)
        self.built_for: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------ build
    def add_node(self, node: GraphNode) -> None:
        existing = self.nodes.get(node.key)
        if existing is None:
            self.nodes[node.key] = node
        else:
            existing.chunk_ids.extend(c for c in node.chunk_ids if c not in existing.chunk_ids)

    def add_edge(self, source: str, target: str, relation: str, evidence: str | None = None) -> None:
        if source == target or source not in self.nodes or target not in self.nodes:
            return
        edge = GraphEdge(source, target, relation, evidence)
        self.out[source].append(edge)
        self.incoming[target].append(edge)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.out.values())

    # --------------------------------------------------------------- traverse
    def neighbours(self, key: str, *, relations: set[str] | None = None) -> list[GraphEdge]:
        edges = list(self.out.get(key, [])) + [
            # Traverse citations in both directions: "what does this cite?" and
            # "what cites this?" are both legally meaningful questions.
            GraphEdge(e.target, e.source, f"{e.relation}_by", e.evidence)
            for e in self.incoming.get(key, [])
        ]
        if relations:
            edges = [e for e in edges if e.relation in relations]
        return edges

    def expand(
        self, seeds: list[str], *, max_hops: int = 2, limit: int = 25
    ) -> list[GraphResult]:
        """Breadth-first expansion from seed nodes, nearest first."""
        results: list[GraphResult] = []
        seen: set[str] = set(seeds)
        queue: deque[tuple[str, int, GraphPath]] = deque(
            (seed, 0, GraphPath([seed], [])) for seed in seeds if seed in self.nodes
        )

        while queue and len(results) < limit:
            key, hops, path = queue.popleft()
            if hops >= max_hops:
                continue
            for edge in self.neighbours(key):
                if edge.target in seen or edge.target not in self.nodes:
                    continue
                seen.add(edge.target)
                next_path = GraphPath(
                    [*path.nodes, edge.target], [*path.relations, edge.relation]
                )
                results.append(
                    GraphResult(
                        node=self.nodes[edge.target],
                        hops=hops + 1,
                        path=next_path,
                        relation=edge.relation,
                    )
                )
                queue.append((edge.target, hops + 1, next_path))
                if len(results) >= limit:
                    break
        return results

    def find_seeds(self, question: str) -> list[str]:
        """Locate graph entry points named explicitly in the question."""
        seeds: list[str] = []
        for match in _SECTION_REF_RE.finditer(question):
            for group in (match.group(1), match.group(2)):
                if not group:
                    continue
                number = group.upper()
                for key, node in self.nodes.items():
                    if node.kind == "provision" and node.metadata.get("section_number") == number:
                        if key not in seeds:
                            seeds.append(key)
        lowered = question.lower()
        for key, node in self.nodes.items():
            if node.kind != "act" or key in seeds:
                continue
            label = node.label.lower()
            # Match the short title as well as the full one. Act labels carry the year
            # ("Indian Contract Act, 1872") but nobody writing a question includes it, so an
            # exact-substring test on the full label almost never fired.
            short = label.split(",")[0].strip()
            if label in lowered or (len(short) > 8 and short in lowered):
                seeds.append(key)
        return seeds

    def seed_from_chunks(self, chunk_ids: list[str]) -> list[str]:
        """Graph entry points for chunks another retriever already found.

        The graph's own `find_seeds` can only fire when a question names a section or an act,
        which conversational questions do not - measured contribution to recall was exactly
        zero. Its real role is expansion from a known-good anchor rather than standalone
        retrieval: given the provisions dense search surfaced, traverse to what they reference.
        """
        wanted = set(chunk_ids)
        seeds: list[str] = []
        for key, node in self.nodes.items():
            if node.kind == "provision" and wanted.intersection(node.chunk_ids):
                seeds.append(key)
        return seeds


def _provision_key(act: str, section: str | None) -> str:
    return f"provision:{act}|{section or '-'}"


def _case_key(case_id: str) -> str:
    return f"case:{case_id}"


def _act_key(act: str) -> str:
    return f"act:{act}"


def build_graph(force: bool = False) -> LegalGraph:
    """Build the graph from provisions, cases and their chunks."""
    global _graph
    with _lock:
        with session_scope() as session:
            provisions = session.scalars(select(LegalProvision)).all()
            cases = session.scalars(select(Case)).all()
            chunks = session.scalars(select(Chunk)).all()

            signature = (len(provisions), len(cases))
            if _graph is not None and not force and _graph.built_for == signature:
                return _graph

            graph = LegalGraph()
            chunks_by_provision: dict[str, list[str]] = defaultdict(list)
            chunks_by_case: dict[str, list[str]] = defaultdict(list)
            for chunk in chunks:
                if chunk.provision_id:
                    chunks_by_provision[chunk.provision_id].append(chunk.id)
                if chunk.case_id:
                    chunks_by_case[chunk.case_id].append(chunk.id)

            # ---- act and provision nodes
            by_section: dict[tuple[str, str], str] = {}
            for provision in provisions:
                act_key = _act_key(provision.act_name)
                graph.add_node(GraphNode(key=act_key, kind="act", label=provision.act_name))

                key = _provision_key(provision.act_name, provision.section_number)
                graph.add_node(
                    GraphNode(
                        key=key,
                        kind="provision",
                        label=f"Section {provision.section_number}, {provision.act_name}",
                        chunk_ids=chunks_by_provision.get(provision.id, []),
                        metadata={
                            "act_name": provision.act_name,
                            "section_number": provision.section_number,
                            "heading": provision.section_heading,
                            "source_url": provision.source_url,
                            "provision_id": provision.id,
                        },
                    )
                )
                graph.add_edge(act_key, key, "contains")
                if provision.section_number:
                    by_section[(provision.act_name, provision.section_number.upper())] = key

            # ---- statute cross-references, evidenced by the provision text itself
            for provision in provisions:
                source_key = _provision_key(provision.act_name, provision.section_number)
                for match in _SECTION_REF_RE.finditer(provision.text or ""):
                    for group in (match.group(1), match.group(2)):
                        if not group:
                            continue
                        target = by_section.get((provision.act_name, group.upper()))
                        if target and target != source_key:
                            graph.add_edge(
                                source_key, target, "cross_references",
                                evidence=match.group(0),
                            )

            # ---- case nodes and the provisions they consider
            for case in cases:
                key = _case_key(case.id)
                graph.add_node(
                    GraphNode(
                        key=key,
                        kind="case",
                        label=case.case_title,
                        chunk_ids=chunks_by_case.get(case.id, []),
                        metadata={
                            "citation": case.citation,
                            "court": case.court,
                            "year": case.year,
                            "outcome": case.disposal_nature,
                            "source_url": case.source_url,
                            "case_db_id": case.id,
                        },
                    )
                )

                # Statutes named in the headnote are authoritative evidence of what the
                # judgment actually considered.
                for act_name in case.legal_provisions or []:
                    normalised = str(act_name).strip()
                    for candidate in {p.act_name for p in provisions}:
                        if _acts_match(normalised, candidate):
                            graph.add_edge(
                                key, _act_key(candidate), "considers_act", evidence=normalised
                            )

                # Section references in the headnote/issues link a case to a provision.
                haystack = f"{case.facts_summary or ''}\n{case.issues or ''}"
                for match in _SECTION_REF_RE.finditer(haystack):
                    for group in (match.group(1), match.group(2)):
                        if not group:
                            continue
                        for (act_name, section), provision_key in by_section.items():
                            if section == group.upper() and _case_mentions_act(case, act_name):
                                graph.add_edge(
                                    key, provision_key, "cites_section",
                                    evidence=match.group(0),
                                )

            graph.built_for = signature
            _graph = graph
            logger.info(
                "legal_graph_built",
                nodes=len(graph.nodes),
                edges=graph.edge_count,
                provisions=len(provisions),
                cases=len(cases),
            )
            return graph


def _acts_match(a: str, b: str) -> bool:
    """Loose act-name comparison; e-SCR writes act names inconsistently."""
    def norm(value: str) -> str:
        value = value.lower().replace(",", " ")
        value = re.sub(r"\b(the|indian|act|code|of|india)\b", " ", value)
        return " ".join(value.split())

    left, right = norm(a), norm(b)
    if not left or not right:
        return False
    return left in right or right in left


def _case_mentions_act(case: Case, act_name: str) -> bool:
    listed = " ".join(str(p) for p in (case.legal_provisions or []))
    return _acts_match(act_name, listed) if listed else False


_graph: LegalGraph | None = None
_lock = threading.Lock()


def get_graph() -> LegalGraph:
    return build_graph()


def reset_graph() -> None:
    global _graph
    with _lock:
        _graph = None


def graph_stats() -> dict[str, object]:
    graph = get_graph()
    kinds: dict[str, int] = defaultdict(int)
    for node in graph.nodes.values():
        kinds[node.kind] += 1
    relations: dict[str, int] = defaultdict(int)
    for edges in graph.out.values():
        for edge in edges:
            relations[edge.relation] += 1
    return {
        "nodes": len(graph.nodes),
        "edges": graph.edge_count,
        "nodes_by_kind": dict(kinds),
        "edges_by_relation": dict(relations),
    }


def chunk_ids_for(results: list[GraphResult], *, collection: Collection | None = None) -> list[str]:
    """Chunk ids reachable from a traversal, for assembling agent context."""
    ids: list[str] = []
    for result in results:
        if collection is Collection.CASE_LAW and result.node.kind != "case":
            continue
        if collection in {Collection.LEGAL_ACTS} and result.node.kind != "provision":
            continue
        for chunk_id in result.node.chunk_ids:
            if chunk_id not in ids:
                ids.append(chunk_id)
    return ids
