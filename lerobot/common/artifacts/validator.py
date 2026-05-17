import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONTRASTIVE_SAMPLERS = {"contrastive", "bidirectional", "bidirectional latent", "poly_bid"}
POLY_BID_DIAGNOSTIC_KEYS = {
    "selected_candidate_index",
    "candidate_indices",
    "bid_quality_scores",
    "diversity_scores",
    "objective_scores",
}
PPO_ROLLOUT_KEYS = {
    "n_sets",
    "n_attempts",
    "set_id",
    "attempt_id",
    "old_log_prob",
    "value",
    "return",
    "advantage",
}


@dataclass
class ValidationResult:
    path: Path
    artifact_kind: str
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def reject(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)


def _load_json(path: Path, result: ValidationResult) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        result.reject(f"invalid JSON at {path}: {type(exc).__name__}: {exc}")
        return None


def _iter_numeric(path: str, value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_numeric(f"{path}.{key}" if path else str(key), item)
    elif isinstance(value, list):
        for ix, item in enumerate(value):
            yield from _iter_numeric(f"{path}[{ix}]", item)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, int | float):
        yield path, float(value)


def _validate_finite_metrics(result: ValidationResult, payload: dict[str, Any]) -> None:
    for path, value in _iter_numeric("", payload):
        if not math.isfinite(value):
            result.reject(f"non-finite numeric metric at {path}: {value}")


def _hash_values(entries: list[dict[str, Any]] | None) -> set[str]:
    if not entries:
        return set()
    return {str(entry.get("sha256")) for entry in entries if entry.get("sha256")}


def _validate_eval_artifact(path: Path, profile: str) -> ValidationResult:
    result = ValidationResult(path=path, artifact_kind="eval")
    eval_path = path / "eval_info.json"
    seed_path = path / "seed_manifest.json"

    if not eval_path.exists():
        result.reject("missing eval_info.json")
        return result

    payload = _load_json(eval_path, result)
    if payload is None:
        return result
    if not isinstance(payload, dict):
        result.reject("eval_info.json must contain a JSON object")
        return result

    for key in ("per_episode", "aggregated", "run_metadata"):
        if key not in payload:
            result.reject(f"missing eval_info.json key: {key}")
    if not result.ok:
        return result

    per_episode = payload["per_episode"]
    aggregated = payload["aggregated"]
    metadata = payload["run_metadata"]
    if not isinstance(per_episode, list):
        result.reject("per_episode must be a list")
        return result

    n_episodes = int(metadata.get("n_episodes", len(per_episode)))
    min_episodes = {"smoke": 1, "benchmark": 20, "final": 500}.get(profile, 20)
    if len(per_episode) != n_episodes:
        result.reject(f"per_episode length {len(per_episode)} does not match n_episodes {n_episodes}")
    if n_episodes < min_episodes:
        result.reject(f"profile '{profile}' requires at least {min_episodes} episodes, got {n_episodes}")

    required_episode_keys = {"episode_ix", "seed", "sum_reward", "max_reward", "success", "num_steps"}
    for ix, episode in enumerate(per_episode):
        missing = required_episode_keys.difference(episode)
        if missing:
            result.reject(f"episode {ix} missing keys: {sorted(missing)}")

    if seed_path.exists():
        seed_manifest = _load_json(seed_path, result)
        if isinstance(seed_manifest, list):
            if len(seed_manifest) < n_episodes:
                result.reject(f"seed_manifest length {len(seed_manifest)} is shorter than n_episodes {n_episodes}")
            seeds = [episode.get("seed") for episode in per_episode]
            if len(set(seeds)) != len(seeds):
                result.reject("per_episode seeds contain duplicates")
        else:
            result.reject("seed_manifest.json must contain a list")
    else:
        result.reject("missing seed_manifest.json")

    sampler = metadata.get("sampler")
    checkpoint = metadata.get("checkpoint", {})
    main_hashes = _hash_values(checkpoint.get("main_policy_hashes"))
    reference_hashes = _hash_values(checkpoint.get("reference_policy_hashes"))
    if checkpoint.get("main_policy_path") and not main_hashes:
        result.reject("main checkpoint path is set but main_policy_hashes is empty")
    if sampler in CONTRASTIVE_SAMPLERS:
        if not checkpoint.get("reference_policy_path") or not reference_hashes:
            result.reject(f"sampler '{sampler}' requires reference policy hashes")
        if main_hashes and reference_hashes and main_hashes == reference_hashes:
            result.reject(f"sampler '{sampler}' has identical main and reference checkpoint hashes")

    if sampler == "poly_bid":
        diagnostics = aggregated.get("sampler_diagnostics", {})
        missing = POLY_BID_DIAGNOSTIC_KEYS.difference(diagnostics)
        if missing:
            result.reject(f"poly_bid missing sampler diagnostics: {sorted(missing)}")

    _validate_finite_metrics(result, aggregated)
    return result


def _validate_ppo_rollout_artifact(path: Path, profile: str) -> ValidationResult:
    del profile
    result = ValidationResult(path=path, artifact_kind="ppo_rollout")
    rollout_path = path / "ppo_rollout.json"
    if not rollout_path.exists():
        result.reject("missing ppo_rollout.json")
        return result

    payload = _load_json(rollout_path, result)
    if payload is None:
        return result
    if not isinstance(payload, dict):
        result.reject("ppo_rollout.json must contain a JSON object")
        return result

    missing = PPO_ROLLOUT_KEYS.difference(payload)
    if missing:
        result.reject(f"ppo rollout missing keys: {sorted(missing)}")
    if payload.get("artifact_kind") == "polyppo_rollout":
        poly_missing = {"diversity_score", "poly_return", "poly_lambda"}.difference(payload)
        if poly_missing:
            result.reject(f"polyppo rollout missing keys: {sorted(poly_missing)}")
    _validate_finite_metrics(result, payload)
    return result


def validate_artifact(path: str | Path, profile: str = "benchmark") -> ValidationResult:
    path = Path(path)
    if (path / "eval_info.json").exists():
        return _validate_eval_artifact(path, profile)
    if (path / "ppo_rollout.json").exists():
        return _validate_ppo_rollout_artifact(path, profile)
    result = ValidationResult(path=path, artifact_kind="unknown", ok=False)
    result.errors.append("unknown artifact kind: expected eval_info.json or ppo_rollout.json")
    return result
