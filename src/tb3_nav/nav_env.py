"""Gymnasium environment: TurtleBot3 Waffle Pi drives to a goal on a generated grid map.

Reuses the existing parts:
    make_scene.py        -> scene XML with a pool of blocks (generate it first)
    layout_generator.py  -> random map, start and goal
    show_layout.py       -> LayoutPlacer puts the layout into the simulation
    lidar.py             -> LiDAR scan (part of the observation)
    controller.py        -> differential drive: [v, omega] -> wheel velocities

Quick check (also done in tests/test_nav_env.py):
    python -c "from tb3_nav.nav_env import NavEnv; from stable_baselines3.common.env_checker import check_env; check_env(NavEnv())"
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from tb3_nav.controller import RobotController
from tb3_nav.layout_generator import generate_layout
from tb3_nav.lidar import MAX_RANGE, Lidar
from tb3_nav.make_scene import MODEL_DIR
from tb3_nav.show_layout import LayoutPlacer
from tb3_nav.geometry import goal_distance_angle

PHYSICS_TIMESTEP = 0.005  # seconds per physics step 
FRAME_SKIP = 10          # physics steps per env step (timestep 10 x 0.005s = 0.05s per action)
MAX_EPISODE_STEPS = 500  # env steps before truncation
GOAL_OFFSET = 0.2
SAFE_DIST = 0.25          # [m] LiDAR distance below which the proximity penalty starts 
MIN_DIST = 0.15           # [m] LiDAR distance at which the penalty is full 
PROXIMITY_PENALTY = -0.1  # reward per step at MIN_DIST or closer, scales linearly to 0 at SAFE_DIST
PROGRESS_SCALE = 10
TIME_PENALTY = -0.01
GOAL_REWARD = 30
COLLISION_PENALTY = -50
STALL_STEPS_NORM = 200


class NavEnv(gym.Env):
    """Robot has to reach the goal without hitting walls.

    Action:      [v, omega] in [-1, 1], scaled to the robot's max speed / max turn rate.
    Observation: [lidar scans, v, omega, dist_to_goal, sin(angle_to_goal), cos(angle_to_goal), gap_from_best, steps_since_record]
    """

    metadata = {"render_modes": []}

    def __init__(self, scene: Path = MODEL_DIR / "scene_grid.xml",
                 mode: str | tuple[str, ...] = "random", n_obstacles: int | tuple[int, int] = 3,
                 min_dist_cells: int = 2, max_steps: int = MAX_EPISODE_STEPS,
                 n_rays: int = 36, fov_deg: float = 360.0):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = PHYSICS_TIMESTEP
        self.placer = LayoutPlacer(self.model)
        self.lidar = Lidar(self.model, fov_deg=fov_deg, n_rays=n_rays)
        self.controller = RobotController(self.model, mode="rl")

        # Layout generator settings (difficulty), can be changed later by the curriculum
        self.set_difficulty(mode, n_obstacles, min_dist_cells, max_steps)
        self.max_goal_dist = np.sqrt(2) * self.placer.grid_size * self.placer.cell_size

        self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.robot_wheel_ids = tuple(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                  for name in ("wheel_left", "wheel_right"))


        # Geoms for collision checks -- change from distance to mujoco collisions
        # robot = every geom of the robot's body tree -- base and wheels
        # obstacle = outer walls and stuff
        self.robot_geoms = set()
        self.obstacle_geoms = set()
        for g in range(self.model.ngeom):
            body = self.model.geom_bodyid[g]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if self.model.body_rootid[body] == self.robot_body_id:
                self.robot_geoms.add(g)
            elif name.startswith(("wall_", "block_")):
                self.obstacle_geoms.add(g)

        # Normalized action, scaled in _apply_action
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation size and bounds to match what _get_obs returns
        obs_size = n_rays  + 3 + 2 + 1 + 1 # n_rays + distance from goal + sin(a) + cos(a) + robot velocity 
                                        # + robot angle velocity + gap from the best distance
                                        # + how long no progrss
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # Episode state, filled in reset()
        self.layout = None
        self.goal_xy = np.zeros(2)
        self.steps = 0
        self.prev_goal_dist = 0.0
        self.last_scan = np.zeros(n_rays)
        self.outcome = None
        self.min_obstacle_dist = MAX_RANGE  # distance to the closest obstacle, updated in step()
        self.collided = False 
        self.steps_since_record = 0


    # -------------------------- Gymnasium API -------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)   # seeds self.np_random

        # Difficulty of this episode: random map type and obstacle count from the allowed ranges
        mode = str(self.np_random.choice(self.modes))
        n_obstacles = int(self.np_random.integers(self.n_obstacles_range[0], self.n_obstacles_range[1] + 1))
        self.layout = generate_layout(self.placer.grid_size, self.np_random, mode=mode,
                                      n_obstacles=n_obstacles, min_dist_cells=self.min_dist_cells)
        self.placer.apply(self.data, self.layout)
        mujoco.mj_forward(self.model, self.data)

        self.goal_xy = self.data.mocap_pos[self.placer.goal_mocap][:2].copy()
        self.steps = 0
        self.prev_goal_dist = self._goal_dist()
        self.outcome = None
        self.min_obstacle_dist = MAX_RANGE
        self.steps_since_record = 0

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        self._apply_action(action)
        self.collided = False
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
            if self._touching_obstacle():
                self.collided = True
        self.steps += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self.steps >= self.max_steps
        self.min_obstacle_dist = min(self.min_obstacle_dist, self.last_scan.min())
        if terminated:
            self.outcome = "goal" if self._goal_reached() else "collision"
        elif truncated:
            self.outcome = "timeout"
        else:
            self.outcome = None

        return obs, reward, terminated, truncated, self._get_info()

    def set_difficulty(self, mode: str | tuple[str, ...], n_obstacles: int | tuple[int, int],
                       min_dist_cells: int, max_steps: int = MAX_EPISODE_STEPS) -> None:
        """Set the task difficulty, used from the next reset().

        mode:        one map type or a tuple of them (one is drawn per episode)
        n_obstacles: fixed count or (min, max) range, drawn per episode (only for mode "random")
        Called by the curriculum through VecEnv.env_method("set_difficulty", ...).
        """
        self.modes = (mode,) if isinstance(mode, str) else tuple(mode)
        self.n_obstacles_range = (n_obstacles, n_obstacles) if isinstance(n_obstacles, int) else tuple(n_obstacles)
        self.min_dist_cells = min_dist_cells
        self.max_steps = max_steps

    # ------------------------------------------------------------------ helpers to fill in

    def _apply_action(self, action: np.ndarray) -> None:
        """Scale the normalized action to [v, omega] and send it to the wheels."""
        action = np.clip(action, -1, 1)
        v = action[0] * self.controller.max_speed
        omega = action[1] * self.controller.max_turn
        self.controller.apply_control(self.data, [v, omega])

    def _get_obs(self) -> np.ndarray:
        """Observation for the policy, must match self.observation_space (shape, dtype float32, bounds)."""
        # lidar scan (self.lidar.scan(self.data)), normalize by max_range
        self.last_scan = self.lidar.scan(self.data)
        lidar_scan = self.last_scan / self.lidar.get_max_range()

        # goal relative to the robot (distance, sin/cos of the angle in the robot frame)
        robot_xy, robot_yaw = self._robot_pose()
        distance, angle = goal_distance_angle(robot_xy, robot_yaw, self.goal_xy)
        sin = np.sin(angle)
        cos = np.cos(angle)
        normalized_distance = distance / self.max_goal_dist
        gap_from_best = max(0.0, distance - self.prev_goal_dist) / self.max_goal_dist #memory -- how much further form the  goal than the best distance so far
        stall = min(self.steps_since_record / STALL_STEPS_NORM, 1.0)   # 0 = progress just now, 1 = stuck >= 10 s


        w_left_omega = self.data.joint(self.robot_wheel_ids[0]).qvel[0]
        w_right_omega = self.data.joint(self.robot_wheel_ids[1]).qvel[0]
        v = self.controller.wheel_radius * (w_left_omega + w_right_omega) / 2
        omega = self.controller.wheel_radius * (w_right_omega - w_left_omega) / self.controller.wheel_separation
        v = v / self.controller.max_speed
        omega = omega / self.controller.max_turn
        return np.concatenate([lidar_scan,[v, omega], [normalized_distance, sin, cos, gap_from_best, stall]]).astype(np.float32)

    def _compute_reward(self) -> float:
        """Reward for the last step."""
        dist = self._goal_dist()
        progress = max(0.0, self.prev_goal_dist - dist)
        # memory for the observation: how long since the last new record
        if progress > 0:
            self.steps_since_record = 0
        else:
            self.steps_since_record += 1

        self.prev_goal_dist = min(self.prev_goal_dist, dist)
        reward = PROGRESS_SCALE * progress + TIME_PENALTY + self._proximity_penalty()
        if self._goal_reached():
            reward += GOAL_REWARD
        elif self._collided():
            reward += COLLISION_PENALTY
        return reward

    def _proximity_penalty(self) -> float:
        """Negative reward for being close to an obstacle, 0 when farther than SAFE_DIST."""
        closest = float(self.last_scan.min())
        closeness = (SAFE_DIST - closest) / (SAFE_DIST - MIN_DIST)   # 0 at SAFE_DIST, 1 at MIN_DIST
        closeness = float(np.clip(closeness, 0.0, 1.0))
        return PROXIMITY_PENALTY * closeness

    

    def _is_terminated(self) -> bool:
        """True when the episode ends by itself: goal reached or collision."""
        return bool(self._goal_reached() or self._collided())

    def _get_info(self) -> dict:
        """Extra data for debugging / evaluation (not seen by the policy)."""
        # "is_success" is read by SB3 (EvalCallback -> eval/success_rate) and by the curriculum
        return {"steps": self.steps,
                "is_success": self._goal_reached(),
                "outcome": self.outcome,
                "min_obstacle_dist": self.min_obstacle_dist}

    def _goal_reached(self) -> bool:
        return self._goal_dist() <= GOAL_OFFSET

    def _collided(self) -> bool:
        """True if the robot touched a wall or block during the last step (set in step())."""
        return self.collided

    def _touching_obstacle(self) -> bool:
        """True if MuJoCo reports a contact between the robot and a wall / block right now."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if ((g1 in self.robot_geoms and g2 in self.obstacle_geoms)
                    or (g2 in self.robot_geoms and g1 in self.obstacle_geoms)):
                return True
        return False



    def _goal_dist(self) -> float:
        robot_xy, robot_yaw = self._robot_pose()
        return goal_distance_angle(robot_xy, robot_yaw, self.goal_xy)[0]


    # ----------------------- small utilities -------------------------------------------

    def _robot_pose(self) -> tuple[np.ndarray, float]:
        """Robot (x, y) of the body origin and yaw [rad] in the world frame."""
        xy = self.data.xpos[self.robot_body_id][:2].copy()
        mat = self.data.xmat[self.robot_body_id].reshape(3, 3)
        yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
        return xy, yaw
