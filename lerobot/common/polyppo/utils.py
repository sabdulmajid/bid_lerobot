#!/usr/bin/env python

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    return OmegaConf.load(path)


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def now_run_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def command_line() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def git_metadata(cwd: str | Path | None = None) -> dict[str, Any]:
    cwd = str(cwd or Path.cwd())

    def _run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        status = _run(["git", "status", "--porcelain"])
        branch = _run(["git", "branch", "--show-current"])
        return {"git_sha": sha, "git_dirty": bool(status), "git_branch": branch}
    except Exception:
        return {"git_sha": None, "git_dirty": None, "git_branch": None}


def select_device(device: str | None = None, gpu_id: int | None = None) -> torch.device:
    if device:
        requested = torch.device(device)
        if requested.type == "cuda" and requested.index is None and gpu_id is not None:
            requested = torch.device(f"cuda:{gpu_id}")
    elif gpu_id is not None:
        requested = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    else:
        requested = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def current_gpu_id(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return device.index if device.index is not None else torch.cuda.current_device()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hashes(path: str | Path | None) -> list[dict[str, str]]:
    if not path:
        return []
    root = Path(path).expanduser()
    if not root.exists():
        return []
    if root.is_file():
        return [{"path": str(root), "sha256": file_sha256(root)}]

    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix in weight_suffixes]
    if not files:
        files = [p for p in root.rglob("config.yaml") if p.is_file()]
    return [{"path": str(p), "sha256": file_sha256(p)} for p in sorted(files)]


def stable_hash(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return json_ready(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))


def env_override(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    return value if value not in (None, "") else default
