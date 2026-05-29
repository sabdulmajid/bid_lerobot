#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StressConfig:
    name: str = "standard"
    action_noise_std: float = 0.0
    observation_noise_std: float = 0.0


def apply_action_noise(action: np.ndarray, cfg: StressConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.action_noise_std <= 0:
        return action
    return action + rng.normal(0.0, cfg.action_noise_std, size=action.shape)


def apply_observation_noise(observation: dict, cfg: StressConfig, rng: np.random.Generator) -> dict:
    if cfg.observation_noise_std <= 0:
        return observation
    out = dict(observation)
    if "agent_pos" in out:
        out["agent_pos"] = out["agent_pos"] + rng.normal(0.0, cfg.observation_noise_std, size=out["agent_pos"].shape)
    return out
