/** Model selector. Lists installed Ollama models and excludes embedding-only models. */

import { api, type ModelsResponse } from "./api";
import { useAsync } from "./components";

export function ModelPicker({
  value,
  onChange,
}: {
  value: string | null;
  onChange: (model: string | null) => void;
}) {
  const { data } = useAsync<ModelsResponse>(() => api.models(), []);
  const usable = (data?.installed ?? []).filter((m) => m.usable_for_generation);

  return (
    <label className="row" style={{ gap: 6 }} title="Generation model used by every agent">
      <span style={{ fontSize: 11, color: "var(--text-faint)", textTransform: "uppercase" }}>
        LLM
      </span>
      <select
        style={{ width: "auto", minWidth: 190, padding: "5px 8px", fontSize: 12 }}
        value={value ?? ""}
        onChange={(event) => onChange(event.target.value || null)}
      >
        <option value="">
          {data ? `Default (${data.default_llm_model})` : "Default"}
        </option>
        {usable.map((m) => (
          <option key={m.name} value={m.name}>
            {m.name}
            {m.parameter_size ? ` · ${m.parameter_size}` : ""}
          </option>
        ))}
      </select>
    </label>
  );
}
