/**
 * Typed client for the LegalAId gateway.
 *
 * The types mirror the recorded observability data rather than a simplified view of it: the UI's
 * whole purpose is to expose chunks, embeddings, vectors, agent contexts, prompts and model runs,
 * so flattening any of that here would defeat the point.
 */

const BASE = import.meta.env.VITE_API_BASE ?? "";

/**
 * API key. The gateway authenticates every route except /api/system/health.
 *
 * Resolution order: an explicit build-time key, then one saved in localStorage. In local
 * development the Vite proxy injects the key from data/.dev-api-key, so neither is needed.
 */
const API_KEY_STORAGE = "legalaid.apiKey";

export function getApiKey(): string {
  return (
    (import.meta.env.VITE_API_KEY as string | undefined) ??
    localStorage.getItem(API_KEY_STORAGE) ??
    ""
  );
}

export function setApiKey(key: string): void {
  if (key.trim()) localStorage.setItem(API_KEY_STORAGE, key.trim());
  else localStorage.removeItem(API_KEY_STORAGE);
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly traceId?: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const key = getApiKey();
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...(key ? { "X-API-Key": key } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    let traceId: string | undefined;
    try {
      const payload = await response.json();
      message = payload?.error?.message ?? payload?.detail ?? message;
      traceId = payload?.error?.trace_id;
    } catch {
      /* non-JSON error body; keep the status text */
    }
    if (response.status === 401) {
      message = `${message} — set an API key (see data/.dev-api-key for the local key).`;
    }
    throw new ApiError(message, response.status, traceId);
  }
  return response.json() as Promise<T>;
}

const get = <T,>(path: string) => request<T>(path);
const post = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

// --------------------------------------------------------------------------- types

export interface Capability {
  key: string;
  label: string;
  phase: number;
  exercise: string;
  description: string;
  pipeline_stages: string[];
  enabled: boolean;
}

export interface Health {
  status: string;
  environment: string;
  active_phase: number;
  transport: string;
  ollama: {
    reachable: boolean;
    base_url: string;
    llm_model: string;
    llm_model_available: boolean;
    embedding_model: string;
    embedding_model_available: boolean;
    error: string | null;
  };
  retrievers: {
    dense: { vectors: number; by_collection: Record<string, number>; dimensions: number };
    keyword: { documents: number; vocabulary: number };
    graph: {
      nodes: number;
      edges: number;
      nodes_by_kind: Record<string, number>;
      edges_by_relation: Record<string, number>;
    };
  };
  counts: Record<string, number>;
  capabilities: Capability[];
}

export interface ModelInfo {
  name: string;
  size_bytes: number;
  parameter_size: string | null;
  quantization: string | null;
  family: string | null;
  usable_for_generation: boolean;
}

export interface ModelsResponse {
  provider: string;
  default_llm_model: string;
  embedding_model: string;
  embedding_dimensions: number;
  num_ctx: number;
  temperature: number;
  reachable: boolean;
  installed: ModelInfo[];
}

export interface RetrievedChunk {
  chunk_id: string;
  collection: string;
  rank: number;
  similarity: number;
  source_title: string;
  section: string | null;
  chapter: string | null;
  page: number | null;
  heading: string | null;
  text: string;
  source_url: string | null;
  source_type: string;
  token_count: number;
  match_reasons: string[];
  matched_terms: string[];
  metadata: Record<string, unknown>;
}

export interface CollectionSearchTrace {
  collection: string;
  retrieval_run_id: string;
  candidate_count: number;
  filtered_count: number;
  top_k: number;
  returned: number;
  min_similarity: number;
  filters: Record<string, unknown>;
  embedding_ms: number;
  search_ms: number;
  rank_ms: number;
  results: RetrievedChunk[];
}

export interface RetrievalResponse {
  trace_id: string | null;
  mode: string;
  question: string;
  query_embedding: {
    provider: string;
    model: string;
    dimensions: number;
    latency_ms: number;
    input_chars: number;
    input_tokens: number | null;
    preview: number[];
  } | null;
  searches: CollectionSearchTrace[];
  chunks: RetrievedChunk[];
  total_ms: number;
  context_chars: number;
  context_tokens: number;
}

export interface Routing {
  input_kind: string;
  document_type: string;
  task: string;
  needs_law: boolean;
  needs_cases: boolean;
  needs_drafting: boolean;
  needs_document_facts: boolean;
  is_legal_matter: boolean;
  original_input: string;
  search_query: string;
  legal_concepts: string[];
  suggested_draft_type: string;
  agents_requested: string[];
  reasoning: string;
  used_fallback: boolean;
}

export interface ContextManifestItem {
  chunk_id: string;
  kind: string;
  label: string;
  reason: string;
  similarity: number | null;
  token_count?: number;
}

export interface AgentSummary {
  agent_run_id: string;
  agent_key: string;
  label: string;
  model: string;
  prompt: string;
  input_tokens: number | null;
  output_tokens: number | null;
  latency_ms: number;
  attempts: number;
  status: string;
  parse_ok: boolean;
  parse_error: string | null;
  output: Record<string, unknown>;
  system_prompt: string;
  user_prompt: string;
  raw_text: string;
  context: {
    agent: string;
    question: string;
    included: ContextManifestItem[];
    excluded: ContextManifestItem[];
    included_count: number;
    excluded_count: number;
    included_tokens: number;
  };
}

export interface LoopIteration {
  agent: string;
  iteration: number;
  action: string;
  audit: Record<string, unknown>;
  guidance: string | null;
}

/** A tagged statement in the answer, so retrieved source and model inference stay separable. */
export interface KeyPoint {
  kind: "FACT" | "LAW" | "CASE_PRECEDENT" | "AI_INFERENCE" | "UNCERTAINTY";
  point: string;
  authority?: string;
}

/** The written reply produced by the synthesis agent. */
export interface SynthesisAnswer {
  direct_answer?: string;
  answer_markdown?: string;
  key_points?: KeyPoint[];
  what_is_missing?: string[];
  suggested_next_steps?: string[];
  confidence?: "high" | "moderate" | "low";
}

export interface ResearchResult {
  trace_id: string;
  question: string;
  total_ms: number;
  answer: SynthesisAnswer | null;
  routing: Routing | null;
  plan: { step: string; reason?: string; mode?: string; parallel: boolean }[];
  facts: Record<string, unknown>;
  legal: Record<string, unknown>;
  cases: Record<string, unknown>;
  draft: Record<string, unknown>;
  warnings: string[];
  retrieval: RetrievalResponse | null;
  loop_iterations: LoopIteration[];
  agents: Record<string, AgentSummary>;
}

export interface TraceStage {
  id: string;
  ordinal: number;
  stage_key: string;
  label: string;
  service: string;
  transport: string;
  status: string;
  duration_ms: number | null;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  error: string | null;
}

export interface TraceDetail {
  trace_id: string;
  question: string;
  mode: string;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  total_ms: number | null;
  answer: Record<string, unknown> | null;
  summary: Record<string, unknown> | null;
  error: string | null;
  stages: TraceStage[];
  agent_runs: {
    id: string;
    agent_key: string;
    label: string;
    status: string;
    duration_ms: number | null;
    prompt_name: string | null;
    prompt_version: string | null;
    included_chunk_ids: string[];
    excluded_chunk_ids: string[];
    context_manifest: Record<string, unknown>;
    output: Record<string, unknown>;
    error: string | null;
  }[];
  model_runs: {
    id: string;
    agent_key: string;
    provider: string;
    model: string;
    temperature: number | null;
    max_tokens: number | null;
    input_tokens: number | null;
    output_tokens: number | null;
    total_duration_ms: number | null;
    tokens_per_second: number | null;
    prompt_name: string | null;
    prompt_version: string | null;
    system_prompt: string | null;
    user_prompt: string | null;
    raw_response: string | null;
    status: string;
  }[];
  retrievals: {
    id: string;
    collection: string;
    mode: string;
    query_text: string;
    embedding_model: string;
    dimensions: number;
    query_vector_preview: number[];
    candidate_count: number;
    filtered_count: number;
    returned_count: number;
    top_k: number;
    min_similarity: number;
    embedding_ms: number;
    search_ms: number;
    rank_ms: number;
    results: {
      chunk_id: string;
      rank: number;
      similarity: number;
      match_reasons: string[];
      matched_terms: string[];
    }[];
  }[];
}

export interface ChunkDetail {
  chunk_id: string;
  collection: string;
  source_type: string;
  source_title: string;
  section: string | null;
  chapter: string | null;
  heading: string | null;
  page: number | null;
  text: string;
  char_count: number;
  token_count: number;
  ordinal: number;
  strategy: string;
  source_url: string | null;
  metadata: Record<string, unknown>;
  version: string;
  created_at: string | null;
  embedding: {
    id: string;
    vector_id: string;
    model: string;
    provider: string;
    dimensions: number;
    norm: number;
    latency_ms: number;
    input_tokens: number | null;
    preview: number[];
  } | null;
  vector: number[] | null;
  vector_dimensions: number | null;
  retrieved_count: number;
  retrievals: {
    retrieval_run_id: string;
    rank: number;
    similarity: number;
    match_reasons: string[];
  }[];
  used_by_agents: { agent: string; trace_id: string; run_id: string }[];
  provenance: {
    provision: Record<string, unknown> | null;
    case: Record<string, unknown> | null;
    document: Record<string, unknown> | null;
  };
}

export interface AgentDescriptor {
  key: string;
  label: string;
  order: number;
  responsibility: string;
  sees: string;
  withheld: string;
  prompt_name: string;
  prompt_version?: string;
  prompt_description?: string;
  temperature?: number | null;
  max_tokens?: number | null;
  runs: number;
  avg_latency_ms: number | null;
  success_rate: number | null;
  recent: {
    id: string;
    trace_id: string;
    status: string;
    duration_ms: number | null;
    prompt_version: string | null;
    included: number;
    excluded: number;
  }[];
}

export interface CompareResult {
  question: string;
  model: string;
  without_rag: {
    answer: string;
    input_tokens: number | null;
    output_tokens: number | null;
    latency_ms: number;
    sources: unknown[];
  };
  with_rag: {
    answer: string;
    input_tokens: number | null;
    output_tokens: number | null;
    latency_ms: number;
    sources: {
      chunk_id: string;
      section: string | null;
      source_title: string;
      similarity: number;
      source_url: string | null;
    }[];
  };
  retrieved_context: string;
}

// --------------------------------------------------------------------------- guardrails

export interface GuardrailLayerResult {
  layer: string;
  verdict: "allow" | "block" | "warn";
  reason: string;
  evidence: Record<string, any>;
}

export interface GuardrailDecision {
  allowed: boolean;
  blocked_by: string | null;
  message: string;
  layers: GuardrailLayerResult[];
  layers_run: number;
}

export interface OutputGuardrail {
  allowed: boolean;
  groundedness: number;
  total_citations: number;
  grounded_citations: number;
  problems: string[];
  message: string;
}

export interface GuardrailConfig {
  enabled: boolean;
  layers: { layer: string; kind: string; checks: string }[];
  corpus_similarity_floor: number;
  floor_basis: string;
  output_guardrail: string;
  auth_enabled: boolean;
  rate_limit: string;
  llm_max_concurrency: number;
}

export interface GuardrailProbeCase {
  id: string;
  text: string;
  category: string;
  note: string;
  expected_blocked_by: string | null;
  allowed: boolean;
  blocked_by: string | null;
  message: string;
  correct: boolean;
  expected_layer_matched: boolean;
  latency_ms: number;
  layers: GuardrailLayerResult[];
}

export interface GuardrailProbeResult {
  summary: {
    total: number;
    adversarial: number;
    legitimate: number;
    caught: number;
    catch_rate: number;
    caught_by_expected_layer: number;
    false_positives: number;
    false_positive_rate: number;
    accuracy: number;
    blocked_by_layer: Record<string, number>;
    mean_latency_ms: number;
    max_latency_ms: number;
    corpus_layer_enabled: boolean;
  };
  cases: GuardrailProbeCase[];
}

// --------------------------------------------------------------------------- endpoints

export const api = {
  health: () => get<Health>("/api/system/health"),
  models: () => get<ModelsResponse>("/api/models"),
  metrics: () => get<Record<string, any>>("/api/metrics"),
  agents: () => get<{ agents: AgentDescriptor[] }>("/api/agents"),
  prompts: () => get<{ prompts: any[] }>("/api/prompts"),
  stages: () => get<{ stages: { key: string; label: string }[] }>("/api/stages"),

  research: (body: {
    question: string;
    pasted_text?: string;
    matter_id?: string | null;
    model?: string | null;
    enable_drafting?: boolean | null;
    enable_verification?: boolean;
    instructions?: string;
  }) => post<ResearchResult>("/api/research/query", body),

  compare: (body: { question: string; model?: string | null; max_tokens?: number }) =>
    post<CompareResult>("/api/research/compare", body),

  retrieve: (body: {
    question: string;
    mode?: string;
    top_k?: number | null;
    matter_id?: string | null;
    enable_keyword?: boolean | null;
    enable_graph?: boolean | null;
    enable_dense?: boolean | null;
    min_similarity?: number | null;
  }) => post<RetrievalResponse>("/api/retrieval/search", body),

  guardrailConfig: () => get<GuardrailConfig>("/api/guardrails/config"),
  guardrailProbe: (checkCorpus = true) =>
    post<GuardrailProbeResult>(
      `/api/guardrails/probe?check_corpus=${checkCorpus}`,
      {},
    ),

  traces: (limit = 30) => get<{ traces: any[] }>(`/api/traces?limit=${limit}`),
  trace: (id: string) => get<TraceDetail>(`/api/traces/${id}`),

  chunks: (params: { collection?: string; q?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.collection) search.set("collection", params.collection);
    if (params.q) search.set("q", params.q);
    search.set("limit", String(params.limit ?? 50));
    search.set("offset", String(params.offset ?? 0));
    return get<{ total: number; chunks: any[] }>(`/api/chunks?${search}`);
  },
  chunk: (id: string) => get<ChunkDetail>(`/api/chunks/${id}`),
  embeddings: (limit = 50) =>
    get<{ total: number; models: string[]; embeddings: any[] }>(`/api/embeddings?limit=${limit}`),

  knowledge: () => get<any>("/api/knowledge"),
  provisions: (params: { act?: string; q?: string } = {}) => {
    const search = new URLSearchParams();
    if (params.act) search.set("act", params.act);
    if (params.q) search.set("q", params.q);
    return get<{ provisions: any[] }>(`/api/knowledge/provisions?${search}`);
  },
  graph: (limit = 160) => get<any>(`/api/knowledge/graph?limit=${limit}`),

  matters: () => get<{ matters: any[] }>("/api/matters"),
  documents: (matterId?: string) =>
    get<{ documents: any[] }>(`/api/documents${matterId ? `?matter_id=${matterId}` : ""}`),
  document: (id: string) => get<any>(`/api/documents/${id}`),

  paste: (body: { text: string; filename?: string; matter_id?: string | null }) =>
    post<any>("/api/documents/paste", body),

  upload: async (file: File, matterId?: string | null) => {
    const form = new FormData();
    form.append("file", file);
    if (matterId) form.append("matter_id", matterId);
    return request<any>("/api/documents/upload", { method: "POST", body: form });
  },

  /** Live pipeline events for a trace. Returns an unsubscribe function. */
  streamTrace: (traceId: string, onEvent: (event: any) => void) => {
    const source = new EventSource(`${BASE}/api/research/stream?trace_id=${traceId}`);
    source.onmessage = (message) => {
      try {
        onEvent(JSON.parse(message.data));
      } catch {
        /* keepalive comment lines are not JSON */
      }
    };
    source.onerror = () => source.close();
    return () => source.close();
  },
};
