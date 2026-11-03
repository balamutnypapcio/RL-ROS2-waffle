"""Gymnasium environment: TurtleBot3 Waffle Pi drives to a goal on a generated grid map.

Reuses the existing parts:
    make_scene.py        -> scene XML with a pool of blocks (generate it first)
    layout_generator.py  -> random map, start and goal
    show_layout.py       -> LayoutPlacer puts the layout into the simulation
    lidar.py             -> LiDAR scan (part of the observation)
    controller.py        -> differential drive: [v, omega] -> wheel velocities

Quick check (from src/):
    python -c "from nav_env import NavEnv; from stable_baselines3.common.env_checker import check_env; check_env(NavEnv())"
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from controller import RobotController
from layout_generator import generate_layout
from lidar import Lidar
from make_scene import MODEL_DIR
from show_layout import LayoutPlacer
from geometry import goal_distance_angle

FRAME_SKIP = 25          # physics steps per env step (timestep 0.002 s -> 0.05 s per action)
MAX_EPISODE_STEPS = 500  # env steps before truncation


class NavEnv(gym.Env):
    """Robot has to reach the goal without hitting walls.

    Action:      [v, omega] in [-1, 1], scaled to the robot's max speed / max turn rate.
    Observation: TODO - decide what goes in (see _get_obs).
    """

    metadata = {"render_modes": []}

    def __init__(self, scene: Path = MODEL_DIR / "scene_grid.xml", mode: str = "random",
                 n_obstacles: int = 3, min_dist_cells: int = 2, n_rays: int = 36, fov_deg: float = 360.0):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)

        self.placer = LayoutPlacer(self.model)
        self.lidar = Lidar(self.model, fov_deg=fov_deg, n_rays=n_rays)
        self.controller = RobotController(self.model, mode="rl")

        # Layout generator settings
        self.mode = mode
        self.n_obstacles = n_obstacles
        self.min_dist_cells = min_dist_cells
        self.max_goal_dist = np.sqrt(2) * self.placer.grid_size * self.placer.cell_size

        self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.robot_wheel_ids = tuple(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                  for name in ("wheel_left", "wheel_right"))

        # Normalized action, scaled in _apply_action
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation size and bounds to match what _get_obs returns
        obs_size = n_rays  + 3 + 2   # n_rays + distance from goal + sin(a) + cos(a) + robot velocity + robot angle velocity
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # Episode state, filled in reset()
        self.layout = None
        self.goal_xy = np.zeros(2)
        self.steps = 0

    # -------------------------- Gymnasium API -------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)   # seeds self.np_random

        self.layout = generate_layout(self.placer.grid_size, self.np_random, mode=self.mode,
                                      n_obstacles=self.n_obstacles, min_dist_cells=self.min_dist_cells)
        self.placer.apply(self.data, self.layout)
        mujoco.mj_forward(self.model, self.data)

        self.goal_xy = self.data.mocap_pos[self.placer.goal_mocap][:2].copy()
        self.steps = 0
        # TODO: reset anything your reward needs from the previous step (e.g. previous distance to goal)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        self._apply_action(action)
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self.steps >= MAX_EPISODE_STEPS

        return obs, reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------ helpers to fill in

    def _apply_action(self, action: np.ndarray) -> None:
        """Scale the normalized action to [v, omega] and send it to the wheels."""
        # TODO: clip action to [-1, 1], scale with self.controller.max_speed / max_turn,
        #       then self.controller.apply_control(self.data, ...)
        raise NotImplementedError

    def _get_obs(self) -> np.ndarray:
        """Observation for the policy, must match self.observation_space (shape, dtype float32, bounds)."""
        # lidar scan (self.lidar.scan(self.data)), normalize by max_range
        lidar_scan = self.lidar.scan(self.data) / self.lidar.get_max_range()

        # goal relative to the robot (distance, sin/cos of the angle in the robot frame)
        robot_xy, robot_yaw = self._robot_pose()
        distance, angle = goal_distance_angle(robot_xy, robot_yaw, self.goal_xy)
        sin = np.sin(angle)
        cos = np.cos(angle)
        normalized_distance = distance / self.max_goal_dist

        w_left_omega = self.data.joint(self.robot_wheel_ids[0]).qvel[0]
        w_right_omega = self.data.joint(self.robot_wheel_ids[1]).qvel[0]
        v = self.controller.wheel_radius * (w_left_omega + w_right_omega) / 2
        omega = self.controller.wheel_radius * (w_right_omega - w_left_omega) / self.controller.wheel_separation
        v = v / self.controller.max_speed
        omega = omega / self.controller.max_turn
        return np.concatenate([lidar_scan,[v, omega], [normalized_distance, sin, cos]]).astype(np.float32)

    def _compute_reward(self) -> float:
        """Reward for the last step."""
        # TODO: progress towards the goal, bonus for reaching it, penalty for collision, time cost
        raise NotImplementedError

    def _is_terminated(self) -> bool:
        """True when the episode ends by itself: goal reached or collision."""
        # TODO
        raise NotImplementedError

    def _get_info(self) -> dict:
        """Extra data for debugging / evaluation (not seen by the policy)."""
        return {"steps": self.steps}

    # ------------------------------------------------------------------ small utilities

    def _robot_pose(self) -> tuple[np.ndarray, float]:
        """Robot (x, y) of the body origin and yaw [rad] in the world frame."""
        xy = self.data.xpos[self.robot_body_id][:2].copy()
        mat = self.data.xmat[self.robot_body_id].reshape(3, 3)
        yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
        return xy, yaw
