import json
import os


def resolve_trace_path(path: str) -> str:
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Trace not found: {path}")


def load_trace_jsonl(path: str) -> list:
    resolved = resolve_trace_path(path)
    steps = []
    with open(resolved, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            steps.append(json.loads(line))
    return steps
