import json

import pytest

from lerobot.common.artifacts.registry import append_run_record, record_artifact_validation


def test_append_run_record_rejects_duplicate_run_id(tmp_path):
    registry = tmp_path / "runs.jsonl"

    append_run_record(registry, {"run_id": "run-1", "status": "completed"})
    with pytest.raises(ValueError, match="duplicate run_id"):
        append_run_record(registry, {"run_id": "run-1", "status": "completed"})


def test_record_artifact_validation_accepts_rejected_runs(tmp_path):
    artifact = tmp_path / "bad_eval"
    artifact.mkdir()
    (artifact / "eval_info.json").write_text("{bad")
    registry = tmp_path / "runs.jsonl"

    record = record_artifact_validation(
        registry,
        run_id="bad-run",
        artifact_path=artifact,
        experiment_name="polyppo-smoke",
        profile="smoke",
    )

    assert record["validation_status"] == "rejected"
    rows = [json.loads(line) for line in registry.read_text().splitlines()]
    assert rows == [record]
