#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exact state snapshot/restore helpers for gym-pusht environments."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


BODY_FIELDS = (
    "position",
    "velocity",
    "force",
    "angle",
    "angular_velocity",
    "torque",
)
BODY_VECTOR_FIELDS = {"position", "velocity", "force"}


def snapshot_pusht_state(env: Any) -> dict[str, Any]:
    """Capture the mutable simulator state needed to exactly resume PushT.

    The snapshot includes Gymnasium wrapper bookkeeping, PushTEnv fields, its
    random generator state, and the Pymunk body state for the agent and block.
    """
    pusht_env = _unwrap(env)
    return {
        "wrappers": _snapshot_wrappers(env),
        "goal_pose": _copy_array_like(pusht_env.goal_pose),
        "n_contact_points": copy.deepcopy(pusht_env.n_contact_points),
        "_last_action": _copy_array_like(getattr(pusht_env, "_last_action", None)),
        "teleop": copy.deepcopy(getattr(pusht_env, "teleop", None)),
        "np_random_bit_generator_state": copy.deepcopy(pusht_env.np_random.bit_generator.state),
        "agent": _snapshot_body(pusht_env.agent),
        "block": _snapshot_body(pusht_env.block),
    }


def restore_pusht_state(env: Any, state: dict[str, Any]) -> None:
    """Restore a PushT environment from ``snapshot_pusht_state`` output."""
    pusht_env = _unwrap(env)

    _restore_wrappers(env, state.get("wrappers", {}))

    pusht_env.goal_pose = _copy_array_like(state["goal_pose"])
    pusht_env.n_contact_points = copy.deepcopy(state["n_contact_points"])
    pusht_env._last_action = _copy_array_like(state["_last_action"])
    pusht_env.teleop = copy.deepcopy(state["teleop"])
    pusht_env.np_random.bit_generator.state = copy.deepcopy(state["np_random_bit_generator_state"])

    _restore_body(pusht_env.agent, state["agent"])
    _restore_body(pusht_env.block, state["block"], angle_before_position=True)

    space = getattr(pusht_env, "space", None)
    if space is not None:
        space.reindex_shapes_for_body(pusht_env.agent)
        space.reindex_shapes_for_body(pusht_env.block)


def get_pusht_state(env: Any) -> dict[str, Any]:
    """Alias for ``snapshot_pusht_state``."""
    return snapshot_pusht_state(env)


def set_pusht_state(env: Any, state: dict[str, Any]) -> None:
    """Alias for ``restore_pusht_state``."""
    restore_pusht_state(env, state)


def _unwrap(env: Any) -> Any:
    if hasattr(env, "unwrapped"):
        return env.unwrapped

    current = env
    seen_ids = set()
    while hasattr(current, "env") and id(current) not in seen_ids:
        seen_ids.add(id(current))
        current = current.env
    return current


def _iter_env_chain(env: Any):
    current = env
    seen_ids = set()
    while current is not None and id(current) not in seen_ids:
        seen_ids.add(id(current))
        yield current
        current = getattr(current, "env", None)


def _snapshot_wrappers(env: Any) -> dict[str, Any]:
    snapshot = {}
    for wrapper in _iter_env_chain(env):
        wrapper_name = type(wrapper).__name__
        if wrapper_name == "TimeLimit" and hasattr(wrapper, "_elapsed_steps"):
            snapshot["TimeLimit._elapsed_steps"] = copy.deepcopy(wrapper._elapsed_steps)
        elif wrapper_name == "OrderEnforcing" and hasattr(wrapper, "_has_reset"):
            snapshot["OrderEnforcing._has_reset"] = copy.deepcopy(wrapper._has_reset)
    return snapshot


def _restore_wrappers(env: Any, state: dict[str, Any]) -> None:
    for wrapper in _iter_env_chain(env):
        wrapper_name = type(wrapper).__name__
        if wrapper_name == "TimeLimit" and "TimeLimit._elapsed_steps" in state:
            wrapper._elapsed_steps = copy.deepcopy(state["TimeLimit._elapsed_steps"])
        elif wrapper_name == "OrderEnforcing" and "OrderEnforcing._has_reset" in state:
            wrapper._has_reset = copy.deepcopy(state["OrderEnforcing._has_reset"])


def _snapshot_body(body: Any) -> dict[str, Any]:
    return {field: _copy_array_like(getattr(body, field)) for field in BODY_FIELDS}


def _restore_body(body: Any, state: dict[str, Any], *, angle_before_position: bool = False) -> None:
    fields = list(BODY_FIELDS)
    if angle_before_position:
        fields.remove("angle")
        fields.insert(0, "angle")

    for field in fields:
        setattr(body, field, _body_value_for_restore(field, state[field]))


def _copy_array_like(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return copy.deepcopy(value)
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        array = np.array(value, copy=True)
    except (TypeError, ValueError):
        return copy.deepcopy(value)
    if array.ndim == 0:
        return array.item()
    return array


def _body_value_for_restore(field: str, value: Any) -> Any:
    value = _copy_array_like(value)
    if field in BODY_VECTOR_FIELDS and isinstance(value, np.ndarray):
        return value.tolist()
    return value
