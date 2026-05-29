import json
from pathlib import Path
from typing import Any

from lerobot.common.artifacts.validator import validate_artifact


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL row {line_number} in {path}: {exc}") from exc
    return rows


def append_run_record(registry_path: str | Path, record: dict[str, Any]) -> None:
    registry_path = Path(registry_path)
    run_id = record.get("run_id")
    if not run_id:
        raise ValueError("run registry record requires `run_id`.")

    existing = _read_jsonl(registry_path)
    if any(row.get("run_id") == run_id for row in existing):
        raise ValueError(f"duplicate run_id in registry: {run_id}")

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def record_artifact_validation(
    registry_path: str | Path,
    *,
    run_id: str,
    artifact_path: str | Path,
    experiment_name: str,
    profile: str = "benchmark",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = validate_artifact(artifact_path, profile=profile)
    record = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "path": str(result.path),
        "artifact_kind": result.artifact_kind,
        "validation_status": "passed" if result.ok else "rejected",
        "validation_errors": result.errors,
        "validation_warnings": result.warnings,
    }
    if extra:
        record.update(extra)
    append_run_record(registry_path, record)
    return record
