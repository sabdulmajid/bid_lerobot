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
import numpy as np
import pytest

from lerobot.common.envs.pusht_state import restore_pusht_state, snapshot_pusht_state


class FakeSpace:
    def __init__(self):
        self.reindexed_bodies = []

    def reindex_shapes_for_body(self, body):
        self.reindexed_bodies.append(body.name)


class FakeBody:
    def __init__(self, name):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "set_log", [])
        self.position = np.array([1.0, 2.0])
        self.velocity = np.array([3.0, 4.0])
        self.force = np.array([5.0, 6.0])
        self.angle = 0.1
        self.angular_velocity = 0.2
        self.torque = 0.3
        self.set_log.clear()

    def __setattr__(self, name, value):
        if name in {"position", "velocity", "force", "angle", "angular_velocity", "torque"}:
            self.set_log.append(name)
        object.__setattr__(self, name, value)


class FakePushTEnv:
    def __init__(self):
        self.goal_pose = np.array([256.0, 256.0, np.pi / 4])
        self.n_contact_points = 7
        self._last_action = np.array([10.0, 20.0])
        self.teleop = True
        self.np_random = np.random.default_rng(123)
        self.agent = FakeBody("agent")
        self.block = FakeBody("block")
        self.space = FakeSpace()

    @property
    def unwrapped(self):
        return self


class OrderEnforcing:
    def __init__(self, env):
        self.env = env
        self._has_reset = True

    @property
    def unwrapped(self):
        return self.env.unwrapped


class TimeLimit:
    def __init__(self, env):
        self.env = env
        self._elapsed_steps = 12

    @property
    def unwrapped(self):
        return self.env.unwrapped


def test_snapshot_restore_fake_pusht_state_and_wrappers():
    unwrapped = FakePushTEnv()
    env = TimeLimit(OrderEnforcing(unwrapped))
    state = snapshot_pusht_state(env)
    expected_next_random = unwrapped.np_random.integers(0, 1000)

    env._elapsed_steps = 99
    env.env._has_reset = False
    unwrapped.goal_pose[:] = -1
    unwrapped.n_contact_points = 0
    unwrapped._last_action[:] = -2
    unwrapped.teleop = False
    unwrapped.agent.position = np.array([100.0, 200.0])
    unwrapped.agent.velocity = np.array([300.0, 400.0])
    unwrapped.agent.force = np.array([500.0, 600.0])
    unwrapped.agent.angle = 1.1
    unwrapped.agent.angular_velocity = 1.2
    unwrapped.agent.torque = 1.3
    unwrapped.block.position = np.array([700.0, 800.0])
    unwrapped.block.velocity = np.array([900.0, 1000.0])
    unwrapped.block.force = np.array([1100.0, 1200.0])
    unwrapped.block.angle = 2.1
    unwrapped.block.angular_velocity = 2.2
    unwrapped.block.torque = 2.3
    unwrapped.np_random.integers(0, 1000)

    unwrapped.agent.set_log.clear()
    unwrapped.block.set_log.clear()
    restore_pusht_state(env, state)

    assert env._elapsed_steps == 12
    assert env.env._has_reset is True
    np.testing.assert_array_equal(unwrapped.goal_pose, [256.0, 256.0, np.pi / 4])
    assert unwrapped.n_contact_points == 7
    np.testing.assert_array_equal(unwrapped._last_action, [10.0, 20.0])
    assert unwrapped.teleop is True
    np.testing.assert_array_equal(unwrapped.agent.position, [1.0, 2.0])
    np.testing.assert_array_equal(unwrapped.agent.velocity, [3.0, 4.0])
    np.testing.assert_array_equal(unwrapped.agent.force, [5.0, 6.0])
    assert unwrapped.agent.angle == 0.1
    assert unwrapped.agent.angular_velocity == 0.2
    assert unwrapped.agent.torque == 0.3
    np.testing.assert_array_equal(unwrapped.block.position, [1.0, 2.0])
    np.testing.assert_array_equal(unwrapped.block.velocity, [3.0, 4.0])
    np.testing.assert_array_equal(unwrapped.block.force, [5.0, 6.0])
    assert unwrapped.block.angle == 0.1
    assert unwrapped.block.angular_velocity == 0.2
    assert unwrapped.block.torque == 0.3
    assert unwrapped.np_random.integers(0, 1000) == expected_next_random
    assert unwrapped.block.set_log.index("angle") < unwrapped.block.set_log.index("position")
    assert unwrapped.space.reindexed_bodies == ["agent", "block"]


def test_snapshot_is_independent_from_subsequent_mutation():
    unwrapped = FakePushTEnv()
    state = snapshot_pusht_state(unwrapped)

    unwrapped.goal_pose[:] = 0
    unwrapped._last_action[:] = 0
    unwrapped.agent.position[:] = 0

    np.testing.assert_array_equal(state["goal_pose"], [256.0, 256.0, np.pi / 4])
    np.testing.assert_array_equal(state["_last_action"], [10.0, 20.0])
    np.testing.assert_array_equal(state["agent"]["position"], [1.0, 2.0])


def test_real_pusht_restore_replays_next_step_exactly_when_available():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("gym_pusht")
    pytest.importorskip("pymunk")

    env = gym.make("gym_pusht/PushT-v0", disable_env_checker=True)
    try:
        env.reset(seed=0)
        env.step(np.array([128.0, 256.0], dtype=np.float32))
        state = snapshot_pusht_state(env)
        action = np.array([300.0, 120.0], dtype=np.float32)

        expected = env.step(action)
        env.step(np.array([10.0, 500.0], dtype=np.float32))
        restore_pusht_state(env, state)
        actual = env.step(action)

        _assert_step_equal(actual, expected)
    finally:
        env.close()


def _assert_step_equal(actual, expected):
    actual_obs, actual_reward, actual_terminated, actual_truncated, actual_info = actual
    expected_obs, expected_reward, expected_terminated, expected_truncated, expected_info = expected

    _assert_obs_equal(actual_obs, expected_obs)
    assert actual_reward == expected_reward
    assert actual_terminated == expected_terminated
    assert actual_truncated == expected_truncated
    for key in ("is_success", "coverage"):
        assert actual_info[key] == expected_info[key]
    for key in ("pos_agent", "vel_agent", "block_pose", "goal_pose"):
        np.testing.assert_allclose(actual_info[key], expected_info[key])


def _assert_obs_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
    else:
        np.testing.assert_array_equal(actual, expected)
