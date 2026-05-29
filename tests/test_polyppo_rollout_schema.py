import json
from pathlib import Path

from lerobot.common.artifacts.validator import validate_artifact
from lerobot.common.polyppo.rollout import collect_polyppo_rollouts
from lerobot.common.polyppo.storage import load_rollout_artifact


def _write_mock_rollout_config(tmp_path: Path) -> Path:
    output_dir = tmp_path / "rollout"
    config = tmp_path / "rollout.yaml"
    config.write_text(
        f"""
run:
  id: test_rollout
  output_dir: {output_dir}
  seed: 123
  device: cpu
  gpu_id: 0
  validate_profile: smoke
policy:
  path: null
  temperature: 0.7
rollout:
  mock: true
  n_prefixes: 4
  attempts_per_prefix: 3
  prefix_steps: 2
  continuation_horizon: 4
  n_obs_steps: 2
  action_dim: 2
  action_chunk_size: 2
  n_code_layers: 2
  n_codes: 8
polyppo:
  diversity_kind: code
  lambda_div: 0.1
"""
    )
    return config


def test_mock_rollout_collection_writes_valid_phase2_artifact(tmp_path):
    config = _write_mock_rollout_config(tmp_path)

    payload = collect_polyppo_rollouts(config)
    loaded_payload, tensors = load_rollout_artifact(payload["output_path"])
    result = validate_artifact(payload["output_path"], profile="smoke")

    assert result.ok, result.errors
    assert payload["validation_status"] == "passed"
    assert loaded_payload["n_sets"] == 4
    assert loaded_payload["n_attempts"] == 3
    assert len(loaded_payload["prefix_env_state_hash"]) == 4
    assert tensors["code_ids"].shape[:3] == (4, 3, 4)
    assert tensors["old_log_probs"].shape == (4, 3, 4)
    assert tensors["values"].shape == (4, 3, 4)
    assert tensors["advantages"].shape == (4, 3, 4)
    assert (Path(payload["output_path"]) / "seed_manifest.json").exists()


def test_rollout_validator_rejects_single_attempt_sets(tmp_path):
    config = _write_mock_rollout_config(tmp_path)
    payload = collect_polyppo_rollouts(config)
    artifact_path = Path(payload["output_path"])
    rollout_json = artifact_path / "ppo_rollout.json"
    data = json.loads(rollout_json.read_text())
    data["n_attempts"] = 1
    rollout_json.write_text(json.dumps(data))

    result = validate_artifact(artifact_path, profile="smoke")

    assert not result.ok
    assert any("n_attempts must be >1" in error for error in result.errors)
