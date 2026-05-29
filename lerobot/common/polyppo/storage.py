#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from lerobot.common.artifacts.validator import validate_artifact
from lerobot.common.polyppo.metrics import tensor_shape_summary
from lerobot.common.polyppo.utils import json_ready, write_json


def save_rollout_artifact(
    output_dir: str | Path,
    payload: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    *,
    validate_profile: str = "smoke",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "rollout_tensors.pt"
    torch.save({key: value.detach().cpu() for key, value in tensors.items()}, tensor_path)

    payload = dict(payload)
    payload["tensor_path"] = tensor_path.name
    payload["tensor_shapes"] = tensor_shape_summary(tensors)
    write_json(output_dir / "ppo_rollout.json", payload)

    seeds = payload.get("seed_manifest", [])
    write_json(output_dir / "seed_manifest.json", seeds)

    result = validate_artifact(output_dir, profile=validate_profile)
    payload["validation_status"] = "passed" if result.ok else "rejected"
    payload["validation_errors"] = result.errors
    write_json(output_dir / "ppo_rollout.json", payload)
    return payload


def load_rollout_artifact(path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    path = Path(path)
    if path.is_file():
        artifact_dir = path.parent
        payload_path = path
    else:
        artifact_dir = path
        payload_path = artifact_dir / "ppo_rollout.json"
    payload = json.loads(payload_path.read_text())
    tensor_path = artifact_dir / payload["tensor_path"]
    tensors = torch.load(tensor_path, map_location="cpu")
    return payload, tensors


def write_train_artifact(output_dir: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "polyppo_train_info.json", json_ready(payload))
    return payload
