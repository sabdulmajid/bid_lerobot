import json

from lerobot.common.artifacts.validator import validate_artifact


def _write_eval_artifact(path, *, sampler="coherence", n_episodes=20, diagnostics=None, checkpoint=None):
    path.mkdir(parents=True)
    if checkpoint is None:
        checkpoint = {
            "main_policy_path": "/tmp/main",
            "reference_policy_path": None,
            "main_policy_hashes": [{"path": "/tmp/main/model.safetensors", "sha256": "main"}],
            "reference_policy_hashes": [],
        }
    payload = {
        "per_episode": [
            {
                "episode_ix": ix,
                "seed": 1000 + ix,
                "sum_reward": 1.0,
                "max_reward": 1.0,
                "success": True,
                "num_steps": 10,
            }
            for ix in range(n_episodes)
        ],
        "aggregated": {
            "pc_success": 100.0,
            "avg_sum_reward": 1.0,
            "pass_at_k": {"pass@1": 1.0},
            "sampler_diagnostics": diagnostics or {},
        },
        "run_metadata": {
            "sampler": sampler,
            "n_episodes": n_episodes,
            "checkpoint": checkpoint,
        },
    }
    (path / "eval_info.json").write_text(json.dumps(payload))
    (path / "seed_manifest.json").write_text(json.dumps([1000 + ix for ix in range(n_episodes)]))


def test_valid_eval_artifact_passes_benchmark_profile(tmp_path):
    artifact = tmp_path / "eval"
    _write_eval_artifact(artifact)

    result = validate_artifact(artifact, profile="benchmark")

    assert result.ok
    assert result.errors == []


def test_truncated_eval_json_is_rejected(tmp_path):
    artifact = tmp_path / "eval"
    artifact.mkdir()
    (artifact / "eval_info.json").write_text('{"per_episode": [')

    result = validate_artifact(artifact, profile="smoke")

    assert not result.ok
    assert any("invalid JSON" in error for error in result.errors)


def test_benchmark_profile_rejects_one_episode_smoke(tmp_path):
    artifact = tmp_path / "eval"
    _write_eval_artifact(artifact, n_episodes=1)

    result = validate_artifact(artifact, profile="benchmark")

    assert not result.ok
    assert any("requires at least 20 episodes" in error for error in result.errors)


def test_bidirectional_eval_rejects_missing_or_identical_reference_hashes(tmp_path):
    missing_ref = tmp_path / "missing_ref"
    _write_eval_artifact(missing_ref, sampler="bidirectional")
    identical_ref = tmp_path / "identical_ref"
    _write_eval_artifact(
        identical_ref,
        sampler="bidirectional",
        checkpoint={
            "main_policy_path": "/tmp/main",
            "reference_policy_path": "/tmp/ref",
            "main_policy_hashes": [{"path": "/tmp/main/model.safetensors", "sha256": "same"}],
            "reference_policy_hashes": [{"path": "/tmp/ref/model.safetensors", "sha256": "same"}],
        },
    )

    missing = validate_artifact(missing_ref)
    identical = validate_artifact(identical_ref)

    assert not missing.ok
    assert any("requires reference policy hashes" in error for error in missing.errors)
    assert not identical.ok
    assert any("identical main and reference" in error for error in identical.errors)


def test_poly_bid_requires_full_sampler_diagnostics(tmp_path):
    artifact = tmp_path / "poly"
    _write_eval_artifact(
        artifact,
        sampler="poly_bid",
        checkpoint={
            "main_policy_path": "/tmp/main",
            "reference_policy_path": "/tmp/ref",
            "main_policy_hashes": [{"path": "/tmp/main/model.safetensors", "sha256": "main"}],
            "reference_policy_hashes": [{"path": "/tmp/ref/model.safetensors", "sha256": "ref"}],
        },
    )

    result = validate_artifact(artifact)

    assert not result.ok
    assert any("poly_bid missing sampler diagnostics" in error for error in result.errors)


def test_ppo_rollout_requires_logprob_value_and_set_ids(tmp_path):
    artifact = tmp_path / "rollout"
    artifact.mkdir()
    (artifact / "ppo_rollout.json").write_text(json.dumps({"artifact_kind": "ppo_rollout"}))

    result = validate_artifact(artifact)

    assert not result.ok
    assert any("ppo rollout missing keys" in error for error in result.errors)
