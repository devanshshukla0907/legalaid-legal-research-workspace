/** Embeddings: the embedding model, dimensions, and per-vector detail. */

import { Link } from "react-router-dom";
import { api } from "../api";
import {
  Card,
  CollectionBadge,
  Empty,
  ErrorNote,
  Metric,
  Spinner,
  VectorPreview,
  ms,
  useAsync,
} from "../components";

export default function Embeddings() {
  const list = useAsync(() => api.embeddings(120), []);
  const models = useAsync(() => api.models(), []);

  if (list.loading) return <Card><Spinner label="Loading embeddings…" /></Card>;
  if (list.error) return <ErrorNote error={list.error} />;

  return (
    <>
      <div className="page-head">
        <h1>Embeddings</h1>
        <p>
          Every chunk's vector, the model that produced it and how long it took. Vectors are
          L2-normalised, so inner product is exactly cosine similarity.
        </p>
      </div>

      <div className="grid cols-4">
        <Metric label="Total embeddings" value={list.data?.total ?? 0} />
        <Metric
          label="Model"
          value={<span style={{ fontSize: 14 }}>{models.data?.embedding_model ?? "—"}</span>}
          sub={models.data?.provider}
        />
        <Metric label="Dimensions" value={models.data?.embedding_dimensions ?? "—"} />
        <Metric
          label="Distinct models used"
          value={list.data?.models.length ?? 0}
          sub={list.data?.models.join(", ")}
        />
      </div>

      <Card title="Vectors">
        {!list.data?.embeddings.length ? (
          <Empty>Nothing embedded yet.</Empty>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Collection</th>
                <th>Vector ID</th>
                <th>Preview</th>
                <th className="num">Dims</th>
                <th className="num">Norm</th>
                <th className="num">Tokens</th>
                <th className="num">Latency</th>
                <th>Chunk</th>
              </tr>
            </thead>
            <tbody>
              {list.data.embeddings.map((embedding: any) => (
                <tr key={embedding.id}>
                  <td>
                    <CollectionBadge collection={embedding.collection} />
                  </td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    {embedding.vector_id}
                  </td>
                  <td style={{ width: 130 }}>
                    <VectorPreview values={embedding.preview} />
                  </td>
                  <td className="num">{embedding.dimensions}</td>
                  <td className="num">{embedding.norm?.toFixed(4)}</td>
                  <td className="num">{embedding.input_tokens ?? "—"}</td>
                  <td className="num">{ms(embedding.latency_ms)}</td>
                  <td>
                    <Link className="mono" style={{ fontSize: 10 }} to={`/chunks/${embedding.chunk_id}`}>
                      inspect
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </>
  );
}
