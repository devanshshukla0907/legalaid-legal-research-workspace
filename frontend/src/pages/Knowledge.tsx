/** Knowledge Base: acts, provisions, judgments, sources and the graph structure. */

import { useState } from "react";
import { api } from "../api";
import { Card, Empty, ErrorNote, Panel, Spinner, useAsync } from "../components";

export default function Knowledge() {
  const { data, error, loading } = useAsync(() => api.knowledge(), []);
  const [tab, setTab] = useState<"acts" | "cases" | "sources" | "graph">("acts");

  if (loading) return <Card><Spinner label="Loading knowledge base…" /></Card>;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;

  return (
    <>
      <div className="page-head">
        <h1>Knowledge Base</h1>
        <p>
          Collections are kept separate — statute, case law and your documents are never merged
          into one undifferentiated index, so retrieval always knows what it is searching.
        </p>
      </div>

      <div className="tabs">
        {(
          [
            ["acts", `Legislation (${data.acts.length})`],
            ["cases", `Case law (${data.cases.length})`],
            ["sources", `Sources (${data.sources.length})`],
            ["graph", "Graph structure"],
          ] as const
        ).map(([key, label]) => (
          <div
            key={key}
            className={`tab ${tab === key ? "active" : ""}`}
            onClick={() => setTab(key)}
          >
            {label}
          </div>
        ))}
      </div>

      {tab === "acts" && <ActsTab acts={data.acts} />}
      {tab === "cases" && <CasesTab cases={data.cases} />}
      {tab === "sources" && <SourcesTab sources={data.sources} />}
      {tab === "graph" && <GraphTab />}
    </>
  );
}

function ActsTab({ acts }: { acts: any[] }) {
  const [act, setAct] = useState<string | null>(null);
  return (
    <div className="split">
      <Card title="Acts" tight>
        <table>
          <tbody>
            {acts.map((entry) => (
              <tr
                key={entry.act_name}
                className="clickable"
                onClick={() => setAct(entry.act_name)}
                style={entry.act_name === act ? { background: "var(--bg-hover)" } : undefined}
              >
                <td>
                  <strong style={{ fontSize: 13 }}>{entry.act_name}</strong>
                  <div style={{ fontSize: 11, color: "var(--text-faint)" }}>
                    {entry.provisions} provisions · {entry.authority}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <div>{act ? <ProvisionList act={act} /> : <Empty>Select an act.</Empty>}</div>
    </div>
  );
}

function ProvisionList({ act }: { act: string }) {
  const { data, loading } = useAsync(() => api.provisions({ act }), [act]);
  if (loading) return <Card><Spinner /></Card>;
  return (
    <Card title={act} actions={<span className="badge neutral">{data?.provisions.length} sections</span>}>
      {(data?.provisions ?? []).map((provision: any) => (
        <Panel
          key={provision.id}
          summary={
            <>
              <span className="badge law">Section {provision.section_number}</span>
              <span style={{ fontSize: 13 }}>{provision.section_heading}</span>
              <span style={{ flex: 1 }} />
              <span className="mono" style={{ fontSize: 11, color: "var(--text-faint)" }}>
                {provision.chars} chars
              </span>
            </>
          }
        >
          <div className="chunk-text">{provision.text}</div>
          {provision.amendment_note && (
            <div style={{ marginTop: 8, fontSize: 12, color: "var(--text-dim)" }}>
              <strong>Amendment footnotes</strong>
              <div style={{ whiteSpace: "pre-wrap" }}>{provision.amendment_note}</div>
            </div>
          )}
          <div className="row" style={{ marginTop: 8 }}>
            <a className="btn small" href={provision.source_url} target="_blank" rel="noreferrer">
              Official source
            </a>
            <span style={{ fontSize: 11, color: "var(--text-faint)" }}>
              {provision.source_authority}
            </span>
          </div>
        </Panel>
      ))}
    </Card>
  );
}

function CasesTab({ cases }: { cases: any[] }) {
  return (
    <Card title="Curated judgments" actions={<span className="badge case">CC-BY-4.0</span>}>
      <table>
        <thead>
          <tr>
            <th>Case</th>
            <th>Citation</th>
            <th>Year</th>
            <th>Outcome</th>
            <th>Provisions considered</th>
          </tr>
        </thead>
        <tbody>
          {cases.map((entry) => (
            <tr key={entry.id}>
              <td>
                <strong style={{ fontSize: 13 }}>{entry.case_title}</strong>
                <div style={{ fontSize: 11, color: "var(--text-faint)" }}>
                  {entry.court}
                  {entry.judges?.length ? ` · ${entry.judges.join(", ")}` : ""}
                </div>
              </td>
              <td className="mono" style={{ fontSize: 11 }}>
                {entry.citation}
                {entry.neutral_citation && (
                  <div style={{ color: "var(--text-faint)" }}>{entry.neutral_citation}</div>
                )}
              </td>
              <td className="num">{entry.year}</td>
              <td style={{ fontSize: 12 }}>{entry.disposal_nature ?? "—"}</td>
              <td>
                {(entry.legal_provisions ?? []).slice(0, 3).map((p: string) => (
                  <span key={p} className="badge law" style={{ marginRight: 4, marginBottom: 3 }}>
                    {p}
                  </span>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function SourcesTab({ sources }: { sources: any[] }) {
  return (
    <>
      {sources.map((source) => (
        <Card key={source.key} title={source.name}>
          <dl className="kv">
            <dt>Authority</dt>
            <dd>{source.authority}</dd>
            <dt>Licence</dt>
            <dd>{source.licence ?? "—"}</dd>
            <dt>Attribution</dt>
            <dd>{source.attribution ?? "—"}</dd>
            <dt>Items</dt>
            <dd className="mono">{source.item_count}</dd>
            <dt>Retrieved</dt>
            <dd className="mono" style={{ fontSize: 11 }}>{source.retrieved_at ?? "—"}</dd>
            {source.source_url && (
              <>
                <dt>URL</dt>
                <dd>
                  <a href={source.source_url} target="_blank" rel="noreferrer">
                    {source.source_url}
                  </a>
                </dd>
              </>
            )}
            {source.notes && (
              <>
                <dt>Notes</dt>
                <dd style={{ color: "var(--text-dim)" }}>{source.notes}</dd>
              </>
            )}
          </dl>
        </Card>
      ))}
    </>
  );
}

function GraphTab() {
  const { data, loading } = useAsync(() => api.graph(200), []);
  if (loading) return <Card><Spinner label="Building graph…" /></Card>;
  if (!data) return null;

  const byKind: Record<string, any[]> = {};
  for (const node of data.nodes) {
    (byKind[node.kind] ??= []).push(node);
  }

  return (
    <>
      <Card title="Graph RAG — symbolic retrieval with no vectors">
        <p style={{ marginTop: 0, color: "var(--text-dim)", fontSize: 13 }}>
          This index answers questions similarity cannot, such as “which judgments interpret this
          section?”. Edges are built from real evidence captured during ingestion: statutory
          cross-references parsed from provision text, and the statutes a judgment's headnote says
          it considered.
        </p>
        <div className="grid cols-4">
          <div className="metric">
            <div className="label">Nodes</div>
            <div className="value">{data.stats.total_nodes}</div>
          </div>
          <div className="metric">
            <div className="label">Edges</div>
            <div className="value">{data.stats.total_edges}</div>
          </div>
          {Object.entries(byKind).map(([kind, nodes]) => (
            <div className="metric" key={kind}>
              <div className="label">{kind}</div>
              <div className="value">{nodes.length}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title={`Edges (${data.edges.length} shown)`}>
        <table>
          <thead>
            <tr>
              <th>From</th>
              <th>Relation</th>
              <th>To</th>
              <th>Evidence</th>
            </tr>
          </thead>
          <tbody>
            {data.edges.slice(0, 120).map((edge: any, index: number) => (
              <tr key={index}>
                <td style={{ fontSize: 12 }}>
                  {data.nodes.find((n: any) => n.key === edge.source)?.label ?? edge.source}
                </td>
                <td>
                  <span className="badge graph">{edge.relation}</span>
                </td>
                <td style={{ fontSize: 12 }}>
                  {data.nodes.find((n: any) => n.key === edge.target)?.label ?? edge.target}
                </td>
                <td className="mono" style={{ fontSize: 11, color: "var(--text-faint)" }}>
                  {edge.evidence ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </>
  );
}
