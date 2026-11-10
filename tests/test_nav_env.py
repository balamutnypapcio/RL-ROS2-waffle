import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from tb3_nav.nav_env import NavEnv
from tb3_nav.train_ppo import STAGES
from tb3_nav.layout_generator import WALL
from tb3_nav.make_scene import cell_center
from tb3_nav.nav_env import TIME_PENALTY



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

def test_touching_a_block_is_a_collision():
    env = NavEnv(n_obstacles=10)
    env.reset(seed=0)
    row, col = np.argwhere(env.layout.grid == WALL)[0]   # first wall cell of the map
    x, y = cell_center(int(row), int(col), env.placer.grid_size, env.placer.cell_size)
    q = env.placer.robot_qpos
    env.data.qpos[q:q + 2] = [x - 0.3, y]       # robot front ~3 cm inside the block's left side
    env.data.qpos[q + 3:q + 7] = [1, 0, 0, 0]   # facing +x, towards the block
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    env.prev_goal_dist = env._goal_dist()       # the teleport itself is not progress

    _, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated
    assert info["outcome"] == "collision"
    assert reward < -40


def test_start_position_is_not_a_collision():
    env = NavEnv(n_obstacles=10)
    for seed in range(20):
        env.reset(seed=seed)
        _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert not terminated, f"seed {seed}: {info['outcome']}"

def test_close_to_a_block_is_penalized_but_not_a_collision():
    env = NavEnv(n_obstacles=10)
    env.reset(seed=0)
    row, col = np.argwhere(env.layout.grid == WALL)[0]
    x, y = cell_center(int(row), int(col), env.placer.grid_size, env.placer.cell_size)
    q = env.placer.robot_qpos
    env.data.qpos[q:q + 2] = [x - 0.40, y]      # ~7 cm gap between the robot front and the block
    env.data.qpos[q + 3:q + 7] = [1, 0, 0, 0]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    env.prev_goal_dist = env._goal_dist()

    _, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert not terminated
    assert info["outcome"] is None
    assert env._proximity_penalty() < 0
    assert reward < TIME_PENALTY       # worse than just standing far from walls
