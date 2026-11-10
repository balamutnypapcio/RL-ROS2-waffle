import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from tb3_nav.nav_env import NavEnv
from tb3_nav.train_ppo import STAGES


@pytest.mark.parametrize("stage", range(len(STAGES)))
def test_check_env_every_stage(stage):
    check_env(NavEnv(**STAGES[stage]))


def test_reset_observation_in_bounds():
    env = NavEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["is_success"] is False


def test_set_difficulty_used_from_next_reset():
    env = NavEnv(**STAGES[0])
    env.set_difficulty(**STAGES[-1])
    assert env.modes == ("random", "rooms", "maze")
    assert env.max_steps == STAGES[-1]["max_steps"]
    env.reset(seed=0)   # must not fail with the new settings


def test_reaching_goal_gives_bonus_and_ends_episode():
    env = NavEnv()
    env.reset(seed=1)   # seed 1: goal away from walls
    q = env.placer.robot_qpos
    env.data.qpos[q:q + 2] = env.goal_xy   # teleport the robot onto the goal
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    env.prev_goal_dist = env._goal_dist()

    _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated and not truncated
    assert info["is_success"]
    assert reward > 10
