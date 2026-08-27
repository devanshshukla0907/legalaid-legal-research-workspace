"""Measure per-model resource cost: load time, RSS, VRAM and GPU/CPU split.

Why this is a separate script rather than part of the runner: Ollama keeps one model resident at
a time and evicts on switch, so per-call sampling during a mixed run would attribute one model's
memory to another. Here each model is loaded alone, measured, then explicitly unloaded.

Sources of truth:
  * `ollama ps` reports the resident size and the CPU/GPU split Ollama itself decided.
  * The ollama process RSS is read from the OS for host-side memory.
  * `nvidia-smi` gives actual VRAM used, if an NVIDIA GPU is present.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

OLLAMA = Path.home() / "AppData/Local/Programs/Ollama/ollama.exe"
OUT = Path(__file__).parent / "results" / "resources.json"

MODELS = ["gemma2:2b", "qwen2.5:3b-instruct", "llama3.2:3b", "codellama:7b-instruct"]

PROMPT = "Reply with the single word: ok"


def sh(args: list[str], timeout: int = 180) -> str:
    try:
        out = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return f"ERR {type(exc).__name__}: {exc}"


def nvidia_used_mb() -> int | None:
    raw = sh(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    match = re.search(r"(\d+)", raw)
    return int(match.group(1)) if match else None


def ollama_rss_mb() -> float | None:
    raw = sh(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-Process -Name 'ollama*' -ErrorAction SilentlyContinue "
            "| Measure-Object WorkingSet64 -Sum).Sum",
        ],
        timeout=60,
    )
    match = re.search(r"(\d{6,})", raw)
    return round(int(match.group(1)) / 1024 / 1024, 1) if match else None


def parse_ps() -> dict[str, str]:
    """Parse `ollama ps` for SIZE and PROCESSOR of the resident model."""
    raw = sh([str(OLLAMA), "ps"], timeout=60)
    for line in raw.splitlines()[1:]:
        if not line.strip():
            continue
        # NAME  ID  SIZE  PROCESSOR  CONTEXT  UNTIL
        size = re.search(r"(\d+\.?\d*\s*[GM]B)", line)
        processor = re.search(r"(\d+%\s*/\s*\d+%\s*CPU/GPU|\d+%\s*GPU|\d+%\s*CPU)", line)
        return {
            "raw": line.strip(),
            "size": size.group(1) if size else "",
            "processor": processor.group(1) if processor else "",
        }
    return {"raw": "", "size": "", "processor": ""}


def measure(model: str) -> dict:
    print(f"\n=== {model} ===")
    sh([str(OLLAMA), "stop", model], timeout=120)
    time.sleep(3)
    baseline_vram = nvidia_used_mb()
    baseline_rss = ollama_rss_mb()
    print(f"  baseline: vram={baseline_vram} MB  ollama_rss={baseline_rss} MB")

    # Cold load: first generation includes reading weights from disk.
    started = time.perf_counter()
    body = json.dumps(
        {
            "model": model,
            "prompt": PROMPT,
            "stream": False,
            "options": {"num_predict": 8, "num_ctx": 3072},
        }
    )
    sh(
        [
            "curl.exe",
            "-s",
            "-X",
            "POST",
            "http://localhost:11434/api/generate",
            "-H",
            "Content-Type: application/json",
            "-d",
            body,
        ],
        timeout=900,
    )
    cold_s = time.perf_counter() - started

    time.sleep(2)
    loaded_vram = nvidia_used_mb()
    loaded_rss = ollama_rss_mb()
    ps = parse_ps()

    # Warm call: model already resident, so this is generation cost only.
    started = time.perf_counter()
    sh(
        [
            "curl.exe",
            "-s",
            "-X",
            "POST",
            "http://localhost:11434/api/generate",
            "-H",
            "Content-Type: application/json",
            "-d",
            body,
        ],
        timeout=600,
    )
    warm_s = time.perf_counter() - started

    record = {
        "model": model,
        "cold_load_s": round(cold_s, 2),
        "warm_call_s": round(warm_s, 2),
        "resident_size": ps["size"],
        "processor_split": ps["processor"],
        "vram_baseline_mb": baseline_vram,
        "vram_loaded_mb": loaded_vram,
        "vram_delta_mb": (loaded_vram - baseline_vram)
        if (loaded_vram is not None and baseline_vram is not None)
        else None,
        "ollama_rss_baseline_mb": baseline_rss,
        "ollama_rss_loaded_mb": loaded_rss,
        "ollama_rss_delta_mb": round(loaded_rss - baseline_rss, 1)
        if (loaded_rss is not None and baseline_rss is not None)
        else None,
        "ps_raw": ps["raw"],
    }
    print(
        f"  cold={record['cold_load_s']}s warm={record['warm_call_s']}s "
        f"size={record['resident_size']} split={record['processor_split']} "
        f"vram_delta={record['vram_delta_mb']} MB"
    )
    sh([str(OLLAMA), "stop", model], timeout=120)
    return record


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = [measure(m) for m in MODELS]
    OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\n-> {OUT}")

    print(f"\n{'model':<28}{'cold':>7}{'warm':>7}{'resident':>11}{'split':>18}{'vram':>9}")
    print("-" * 82)
    for r in records:
        print(
            f"{r['model']:<28}{r['cold_load_s']:>6.1f}s{r['warm_call_s']:>6.1f}s"
            f"{r['resident_size']:>11}{r['processor_split']:>18}"
            f"{(str(r['vram_delta_mb']) + ' MB') if r['vram_delta_mb'] is not None else '-':>9}"
        )


if __name__ == "__main__":
    main()
