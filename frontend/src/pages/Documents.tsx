/** Documents: upload or paste, then inspect extracted pages and the chunks derived from them. */

import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Card, Empty, ErrorNote, Panel, Spinner, StatusBadge, useAsync } from "../components";

export default function Documents() {
  const [refresh, setRefresh] = useState(0);
  const { data, error, loading } = useAsync(() => api.documents(), [refresh]);
  const [selected, setSelected] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function upload(file: File) {
    setBusy(true);
    setNote(null);
    try {
      const response = await api.upload(file);
      setNote(
        `${response.filename}: ${response.pages} page(s), ${response.chunks} chunks, ${response.vectors_indexed} vectors — classified as ${response.doc_class}`,
      );
      setRefresh((n) => n + 1);
    } catch (err) {
      setNote((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function paste() {
    setBusy(true);
    setNote(null);
    try {
      const response = await api.paste({ text, filename: "pasted-document.txt" });
      setNote(
        `Ingested: ${response.chunks} chunks, ${response.vectors_indexed} vectors — classified as ${response.doc_class}`,
      );
      setText("");
      setRefresh((n) => n + 1);
    } catch (err) {
      setNote((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <h1>Documents</h1>
        <p>
          Upload a PDF, DOCX or TXT, or paste text directly. Page-level provenance is preserved so
          every extracted fact can be traced back to the page it came from.
        </p>
      </div>

      <Card title="Add a document">
        <div className="row" style={{ marginBottom: 10 }}>
          <button onClick={() => fileRef.current?.click()} disabled={busy}>
            Choose file
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.txt,.docx,.md"
            style={{ display: "none" }}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) upload(file);
              event.target.value = "";
            }}
          />
          <span style={{ fontSize: 12, color: "var(--text-faint)" }}>
            PDF · DOCX · TXT · MD, up to 25 MB
          </span>
        </div>
        <textarea
          placeholder="…or paste an FIR, legal notice, agreement or client statement here"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <div className="row" style={{ marginTop: 8 }}>
          <button className="primary" onClick={paste} disabled={busy || text.trim().length < 20}>
            {busy ? "Processing…" : "Ingest pasted text"}
          </button>
          {busy && <Spinner />}
        </div>
        {note && (
          <div className="disclaimer" style={{ marginTop: 10, marginBottom: 0 }}>
            {note}
          </div>
        )}
      </Card>

      {error && <ErrorNote error={error} />}

      <div className="split">
        <Card title="Documents" tight>
          {loading ? (
            <Spinner />
          ) : !data?.documents.length ? (
            <Empty>No documents yet.</Empty>
          ) : (
            <table>
              <tbody>
                {data.documents.map((doc: any) => (
                  <tr
                    key={doc.id}
                    className="clickable"
                    onClick={() => setSelected(doc.id)}
                    style={doc.id === selected ? { background: "var(--bg-hover)" } : undefined}
                  >
                    <td>
                      <div className="row" style={{ gap: 6 }}>
                        <StatusBadge status={doc.status} />
                        <strong style={{ fontSize: 13 }}>{doc.filename}</strong>
                      </div>
                      <div style={{ fontSize: 11, color: "var(--text-faint)", marginTop: 2 }}>
                        {doc.doc_class} · {doc.page_count}p · {doc.chunks} chunks ·{" "}
                        {(doc.size_bytes / 1024).toFixed(0)} KB
                        {doc.is_synthetic && " · synthetic"}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>

        <div>
          {selected ? <DocumentView id={selected} /> : <Empty>Select a document.</Empty>}
        </div>
      </div>
    </>
  );
}

function DocumentView({ id }: { id: string }) {
  const { data, loading } = useAsync(() => api.document(id), [id]);
  if (loading) return <Card><Spinner /></Card>;
  if (!data) return null;

  return (
    <>
      <Card
        title={data.filename}
        actions={
          <span className="row" style={{ gap: 6 }}>
            <span className="badge doc">{data.doc_class}</span>
            <StatusBadge status={data.status} />
          </span>
        }
      >
        <dl className="kv">
          <dt>Extraction method</dt>
          <dd className="mono">{data.extraction_method}</dd>
          <dt>Pages</dt>
          <dd className="mono">{data.page_count}</dd>
          <dt>Chunks</dt>
          <dd className="mono">{data.chunks.length}</dd>
          <dt>SHA-256</dt>
          <dd className="mono" style={{ fontSize: 11 }}>{data.sha256}</dd>
        </dl>
        {data.error && <ErrorNote error={data.error} />}
      </Card>

      <Card title="Extracted text by page">
        {data.pages.map((page: any) => (
          <Panel
            key={page.page_number}
            summary={
              <>
                <span className="badge neutral">Page {page.page_number}</span>
                <span className="mono" style={{ fontSize: 11, color: "var(--text-faint)" }}>
                  {page.char_count} chars
                </span>
              </>
            }
          >
            <div className="chunk-text doc">{page.text}</div>
          </Panel>
        ))}
      </Card>

      <Card title={`Chunks derived from this document (${data.chunks.length})`}>
        <table>
          <thead>
            <tr>
              <th className="num">#</th>
              <th className="num">Page</th>
              <th>Heading</th>
              <th className="num">Chars</th>
              <th className="num">Tokens</th>
              <th>Preview</th>
            </tr>
          </thead>
          <tbody>
            {data.chunks.map((chunk: any) => (
              <tr key={chunk.chunk_id}>
                <td className="num">{chunk.ordinal}</td>
                <td className="num">{chunk.page ?? "—"}</td>
                <td style={{ fontSize: 12 }}>{chunk.heading ?? "—"}</td>
                <td className="num">{chunk.char_count}</td>
                <td className="num">{chunk.token_count}</td>
                <td style={{ fontSize: 11, color: "var(--text-dim)" }}>
                  <Link to={`/chunks/${chunk.chunk_id}`}>{chunk.preview.slice(0, 70)}…</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </>
  );
}
