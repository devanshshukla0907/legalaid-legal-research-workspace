/** Chunk Explorer: full detail for any chunk, including its vector and how it has been used. */

import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, type ChunkDetail } from "../api";
import {
  Card,
  CollectionBadge,
  Empty,
  ErrorNote,
  Json,
  Panel,
  SimilarityBar,
  Spinner,
  VectorPreview,
  ms,
  useAsync,
} from "../components";

const COLLECTIONS = [
  "",
  "LEGAL_ACTS",
  "CASE_LAW",
  "USER_DOCUMENTS",
] as const;

export default function Chunks() {
  const { chunkId } = useParams();
  const navigate = useNavigate();
  const [collection, setCollection] = useState("");
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string | null>(chunkId ?? null);

  useEffect(() => {
    if (chunkId) setSelected(chunkId);
  }, [chunkId]);

  const list = useAsync(
    () => api.chunks({ collection: collection || undefined, q: search || undefined, limit: 100 }),
    [collection, search],
  );

  return (
    <>
      <div className="page-head">
        <h1>Chunk Explorer</h1>
        <p>
          Every chunk with its metadata, embedding, vector, retrieval history and full provenance
          back to an official source.
        </p>
      </div>

      <div className="split">
        <Card title="Chunks" tight>
          <div className="row" style={{ padding: "8px 0", gap: 6 }}>
            <select
              value={collection}
              onChange={(event) => setCollection(event.target.value)}
              style={{ width: "auto" }}
            >
              {COLLECTIONS.map((c) => (
                <option key={c} value={c}>
                  {c || "All collections"}
                </option>
              ))}
            </select>
            <input
              placeholder="Search text…"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && setSearch(query)}
            />
            <button className="small" onClick={() => setSearch(query)}>
              Find
            </button>
          </div>

          {list.loading ? (
            <Spinner />
          ) : list.error ? (
            <ErrorNote error={list.error} />
          ) : (
            <>
              <div style={{ fontSize: 11, color: "var(--text-faint)", padding: "0 0 6px" }}>
                {list.data?.total ?? 0} chunks
              </div>
              <div className="list-scroll">
                <table>
                  <tbody>
                    {list.data?.chunks.map((chunk: any) => (
                      <tr
                        key={chunk.chunk_id}
                        className="clickable"
                        onClick={() => {
                          setSelected(chunk.chunk_id);
                          navigate(`/chunks/${chunk.chunk_id}`);
                        }}
                        style={
                          chunk.chunk_id === selected
                            ? { background: "var(--bg-hover)" }
                            : undefined
                        }
                      >
                        <td>
                          <div className="row" style={{ gap: 5 }}>
                            <CollectionBadge collection={chunk.collection} />
                            <strong style={{ fontSize: 12 }}>
                              {chunk.section ?? chunk.heading ?? "—"}
                            </strong>
                          </div>
                          <div
                            style={{
                              fontSize: 11,
                              color: "var(--text-dim)",
                              marginTop: 2,
                              maxWidth: 320,
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {chunk.source_title}
                          </div>
                          <div style={{ fontSize: 10, color: "var(--text-faint)" }}>
                            {chunk.char_count} chars · {chunk.token_count} tokens ·{" "}
                            {chunk.strategy}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </Card>

        <div>
          {selected ? (
            <ChunkView chunkId={selected} />
          ) : (
            <Empty>Select a chunk to inspect it.</Empty>
          )}
        </div>
      </div>
    </>
  );
}

function ChunkView({ chunkId }: { chunkId: string }) {
  const { data, error, loading } = useAsync<ChunkDetail>(() => api.chunk(chunkId), [chunkId]);

  if (loading) return <Card><Spinner label="Loading chunk…" /></Card>;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;

  const provision = data.provenance.provision as any;
  const legalCase = data.provenance.case as any;
  const document = data.provenance.document as any;

  return (
    <>
      <Card
        title={
          <span className="row" style={{ gap: 8 }}>
            <CollectionBadge collection={data.collection} />
            {data.section ?? data.heading ?? "Chunk"}
          </span>
        }
        actions={<span className="mono" style={{ fontSize: 11 }}>{data.chunk_id}</span>}
      >
        <div
          className={`chunk-text ${data.collection === "CASE_LAW" ? "case" : data.collection === "USER_DOCUMENTS" ? "doc" : ""}`}
          style={{ maxHeight: 420 }}
        >
          {data.text}
        </div>
      </Card>

      <div className="grid cols-2">
        <Card title="Chunk metadata">
          <dl className="kv">
            <dt>Source title</dt>
            <dd>{data.source_title}</dd>
            <dt>Section</dt>
            <dd>{data.section ?? "—"}</dd>
            <dt>Chapter</dt>
            <dd>{data.chapter ?? "—"}</dd>
            <dt>Heading</dt>
            <dd>{data.heading ?? "—"}</dd>
            <dt>Page</dt>
            <dd>{data.page ?? "—"}</dd>
            <dt>Source type</dt>
            <dd>{data.source_type}</dd>
            <dt>Chunk strategy</dt>
            <dd className="mono">{data.strategy}</dd>
            <dt>Characters</dt>
            <dd className="mono">{data.char_count}</dd>
            <dt>Tokens</dt>
            <dd className="mono">{data.token_count}</dd>
            <dt>Ordinal</dt>
            <dd className="mono">{data.ordinal}</dd>
            <dt>Version</dt>
            <dd className="mono">{data.version}</dd>
            <dt>Created</dt>
            <dd className="mono" style={{ fontSize: 11 }}>{data.created_at ?? "—"}</dd>
          </dl>
        </Card>

        <Card title="Embedding & vector">
          {data.embedding ? (
            <>
              <dl className="kv">
                <dt>Model</dt>
                <dd className="mono">{data.embedding.model}</dd>
                <dt>Provider</dt>
                <dd className="mono">{data.embedding.provider}</dd>
                <dt>Dimensions</dt>
                <dd className="mono">{data.embedding.dimensions}</dd>
                <dt>Vector ID</dt>
                <dd className="mono">{data.embedding.vector_id}</dd>
                <dt>L2 norm</dt>
                <dd className="mono">{data.embedding.norm.toFixed(6)}</dd>
                <dt>Embed latency</dt>
                <dd className="mono">{ms(data.embedding.latency_ms)}</dd>
                <dt>Input tokens</dt>
                <dd className="mono">{data.embedding.input_tokens ?? "—"}</dd>
              </dl>
              <div style={{ marginTop: 10 }}>
                <div style={{ fontSize: 11, color: "var(--text-faint)", marginBottom: 5 }}>
                  Vector preview
                </div>
                <VectorPreview values={data.vector?.slice(0, 48) ?? data.embedding.preview} />
              </div>
              {data.vector && (
                <Panel summary={<span>Raw vector (first 64 of {data.vector_dimensions})</span>}>
                  <pre>[{data.vector.map((v) => v.toFixed(6)).join(", ")}]</pre>
                </Panel>
              )}
            </>
          ) : (
            <Empty>This chunk has not been embedded.</Empty>
          )}
        </Card>
      </div>

      <Card title="Provenance — where this came from">
        {provision && (
          <dl className="kv">
            <dt>Act</dt>
            <dd>{provision.act_name}</dd>
            <dt>Section</dt>
            <dd>
              {provision.section_number} — {provision.section_heading}
            </dd>
            <dt>Authority</dt>
            <dd>{provision.source_authority}</dd>
            <dt>Official source</dt>
            <dd>
              <a href={provision.source_url} target="_blank" rel="noreferrer">
                {provision.source_url}
              </a>
            </dd>
            <dt>Retrieved at</dt>
            <dd className="mono" style={{ fontSize: 11 }}>{provision.retrieved_at}</dd>
            {provision.amendment_note && (
              <>
                <dt>Amendment footnotes</dt>
                <dd style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>
                  {provision.amendment_note}
                </dd>
              </>
            )}
          </dl>
        )}
        {legalCase && (
          <dl className="kv">
            <dt>Case</dt>
            <dd>{legalCase.case_title}</dd>
            <dt>Citation</dt>
            <dd className="mono">{legalCase.citation}</dd>
            <dt>Neutral citation</dt>
            <dd className="mono">{legalCase.neutral_citation ?? "—"}</dd>
            <dt>Court</dt>
            <dd>{legalCase.court}</dd>
            <dt>Decided</dt>
            <dd>{legalCase.decision_date ?? "—"}</dd>
            <dt>Bench</dt>
            <dd>{(legalCase.judges ?? []).join(", ") || "—"}</dd>
            <dt>Outcome</dt>
            <dd>{legalCase.disposal_nature ?? "—"}</dd>
            <dt>Licence</dt>
            <dd>{legalCase.licence ?? "—"}</dd>
            <dt>Source</dt>
            <dd>
              <a href={legalCase.source_url} target="_blank" rel="noreferrer">
                official PDF
              </a>
            </dd>
          </dl>
        )}
        {document && (
          <dl className="kv">
            <dt>Document</dt>
            <dd>{document.filename}</dd>
            <dt>Classified as</dt>
            <dd>{document.doc_class}</dd>
            <dt>Pages</dt>
            <dd className="mono">{document.page_count}</dd>
          </dl>
        )}
        {!provision && !legalCase && !document && <Empty>No linked source record.</Empty>}
      </Card>

      <div className="grid cols-2">
        <Card title={`Retrieval history (${data.retrieved_count})`}>
          {data.retrievals.length === 0 ? (
            <Empty>This chunk has never been retrieved.</Empty>
          ) : (
            <table>
              <thead>
                <tr>
                  <th className="num">Rank</th>
                  <th>Similarity</th>
                  <th>Why</th>
                </tr>
              </thead>
              <tbody>
                {data.retrievals.map((r, i) => (
                  <tr key={i}>
                    <td className="num">{r.rank}</td>
                    <td>
                      <SimilarityBar value={r.similarity} />
                    </td>
                    <td style={{ fontSize: 11, color: "var(--text-dim)" }}>
                      {r.match_reasons.slice(0, 2).join(" · ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>

        <Card title="Used by agents">
          {data.used_by_agents.length === 0 ? (
            <Empty>No agent has used this chunk yet.</Empty>
          ) : (
            data.used_by_agents.map((use, i) => (
              <div key={i} className="row between" style={{ padding: "5px 0", fontSize: 12 }}>
                <span>{use.agent}</span>
                <a className="mono" style={{ fontSize: 11 }} href={`/pipeline/${use.trace_id}`}>
                  {use.trace_id}
                </a>
              </div>
            ))
          )}
        </Card>
      </div>

      <Card title="Raw metadata">
        <Json value={data.metadata} />
      </Card>
    </>
  );
}
