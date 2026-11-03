"""Show generated layouts in MuJoCo: blocks from the pool become walls of the map.

Connects the parts:
    make_scene.py        -> XML with a pool of blocks (generate it first)
    layout_generator.py  -> grid with walls, start and goal
    lidar.py             -> simulated 2D LiDAR, rays are drawn in the viewer
    controller.py        -> robot control (keyboard, later RL)
    logger.py            -> LiDAR scans to the terminal or a file

Usage:
    python src/make_scene.py --grid 8 --cell 0.5
    python src/show_layout.py --mode random --obstacles 12
    python src/show_layout.py --mode maze
    python src/show_layout.py --fov 360 --rays 36 --log-mode terminal

Keys (click the viewer window first):
    N       - new random layout
    arrows  - drive (up/down: forward/back, left/right: turn)
    Space   - stop
"""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from controller import RobotController
from layout_generator import MODES, WALL, Layout, generate_layout, layout_to_str
from lidar import Lidar
from logger import DataLogger
from make_scene import MODEL_DIR, PARKING_POS, cell_center

KEY_N = 78  # GLFW key code of 'N'

LOG_INTERVAL = 0.5  # seconds of simulation time between logged scans

# Colors of the drawn LiDAR rays (RGBA): ray that hit something / ray with no hit
RAY_HIT_RGBA = [1.0, 0.2, 0.2, 0.8]
RAY_MISS_RGBA = [0.3, 0.8, 0.3, 0.3]

# The robot's body origin is on the wheel axle, but its footprint is shifted backwards:
# the base box spans about -0.197 .. +0.077 m along the robot's x axis (center ~ -0.06 m).
# Placing the FOOTPRINT center (not the origin) at the cell center keeps the robot away
# from the walls of neighboring cells for every heading.
FOOTPRINT_CENTER_X = -0.06


class LayoutPlacer:
    """Moves pool blocks, the robot and the goal marker according to a Layout."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        name = lambda obj, i: mujoco.mj_id2name(model, obj, i) or ""

        # Pool blocks: mocap indices, sorted by name (block_000, block_001, ...)
        block_bodies = sorted(
            (name(mujoco.mjtObj.mjOBJ_BODY, b), b) for b in range(model.nbody)
            if name(mujoco.mjtObj.mjOBJ_BODY, b).startswith("block_")
        )
        self.block_mocap = [model.body_mocapid[b] for _, b in block_bodies]
        if not self.block_mocap:
            raise SystemExit("No block_* bodies in the scene - generate it with make_scene.py")

        goal_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self.goal_mocap = model.body_mocapid[goal_body]

        # Robot free joint: where its position and orientation live in qpos / qvel
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_joint")
        self.robot_qpos = model.jnt_qposadr[joint]
        self.robot_qvel = model.jnt_dofadr[joint]

        # Grid parameters read from the model itself, so they always match the XML
        first_block_geom = model.body_geomadr[block_bodies[0][1]]
        half_x, _, half_z = model.geom_size[first_block_geom]
        self.cell_size = 2 * float(half_x)
        self.block_z = float(half_z)                      # center height = half of the wall height
        self.grid_size = int(round(np.sqrt(len(self.block_mocap))))

    def apply(self, data: mujoco.MjData, layout: Layout) -> None:
        """Put the layout into the simulation (call before mj_forward / mj_step)."""
        n = self.grid_size
        if layout.grid.shape != (n, n):
            raise ValueError(f"layout is {layout.grid.shape}, scene grid is {n}x{n}")

        mujoco.mj_resetData(self.model, data)   # zero velocities, time, controls

        # 1. walls: one pool block per WALL cell
        wall_cells = np.argwhere(layout.grid == WALL)
        if len(wall_cells) > len(self.block_mocap):
            raise ValueError(f"{len(wall_cells)} walls but only {len(self.block_mocap)} blocks in the pool")
        for mocap_id, (row, col) in zip(self.block_mocap, wall_cells):
            x, y = cell_center(int(row), int(col), n, self.cell_size)
            data.mocap_pos[mocap_id] = [x, y, self.block_z]

        # 2. park the rest
        for mocap_id in self.block_mocap[len(wall_cells):]:
            data.mocap_pos[mocap_id] = PARKING_POS

        # 3. robot: footprint center at the start cell center, heading = start_yaw
        cx, cy = cell_center(*layout.start, n, self.cell_size)
        yaw = layout.start_yaw
        x = cx - FOOTPRINT_CENTER_X * np.cos(yaw)
        y = cy - FOOTPRINT_CENTER_X * np.sin(yaw)
        q = self.robot_qpos
        data.qpos[q:q + 3] = [x, y, 0.0]
        data.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]  # quaternion (w, x, y, z), rotation about z

        # 4. goal marker
        gx, gy = cell_center(*layout.goal, n, self.cell_size)
        data.mocap_pos[self.goal_mocap] = [gx, gy, 0.0]


def draw_rays(viewer: mujoco.viewer.Handle, ray_points: list) -> None:
    """Draw LiDAR rays as lines in the viewer's user scene (replaces the previous frame's rays)."""
    scn = viewer.user_scn
    scn.ngeom = 0
    for start, end, hit in ray_points:
        if scn.ngeom >= scn.maxgeom:
            break
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, start, end)
        geom.rgba[:] = RAY_HIT_RGBA if hit else RAY_MISS_RGBA
        scn.ngeom += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show generated layouts in the MuJoCo viewer.")
    parser.add_argument("--scene", type=Path, default=MODEL_DIR / "scene_grid.xml",
                        help="scene generated by make_scene.py (default: models/scene_grid.xml)")
    parser.add_argument("--mode", choices=MODES, default="random", help="map type (default: random)")
    parser.add_argument("--obstacles", type=int, default=12, help="obstacles for mode 'random' (default: 12)")
    parser.add_argument("--min-dist", type=int, default=4, help="min BFS distance start-goal in cells (default: 4)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: different every run)")

    parser.add_argument("--fov", type=float, default=360.0, help="LiDAR field of view in degrees (default: 360)")
    parser.add_argument("--rays", type=int, default=36, help="number of LiDAR rays (default: 36)")
    parser.add_argument("--log-mode", choices=["terminal", "file", "none"], default="none",
                        help="where to log LiDAR scans (default: none)")
    parser.add_argument("--log-file", type=str, default="lidar_data.txt",
                        help="file for --log-mode file (default: lidar_data.txt)")
    parser.add_argument("--control", choices=["keyboard", "rl"], default="keyboard",
                        help="robot control source (default: keyboard)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    placer = LayoutPlacer(model)
    rng = np.random.default_rng(args.seed)
    print(f"Scene: {placer.grid_size}x{placer.grid_size} cells of {placer.cell_size:g} m, "
          f"{len(placer.block_mocap)} blocks in the pool")

    lidar = Lidar(model, fov_deg=args.fov, n_rays=args.rays)
    logger = DataLogger(mode=args.log_mode, file_path=args.log_file)
    controller = RobotController(model, mode=args.control)
    last_log_time = 0.0

    def new_layout() -> None:
        nonlocal last_log_time
        layout = generate_layout(placer.grid_size, rng, mode=args.mode,
                                 n_obstacles=args.obstacles, min_dist_cells=args.min_dist)
        placer.apply(data, layout)
        mujoco.mj_forward(model, data)
        last_log_time = 0.0   # apply() resets data.time to 0
        start = tuple(int(v) for v in layout.start)   # plain ints print nicer than np.int64
        goal = tuple(int(v) for v in layout.goal)
        print(f"\n{args.mode} layout, start {start} yaw {np.degrees(layout.start_yaw):.0f} deg, goal {goal}:")
        print(layout_to_str(layout))

    # The key callback runs in the viewer's thread, so it only sets a flag
    # (or the controller's command); the main loop below does the actual work.
    want_new = {"flag": False}

    def on_key(key: int) -> None:
        if key == KEY_N:
            want_new["flag"] = True
        else:
            controller.handle_key(key)

    new_layout()
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        print("\nPress N in the viewer for a new layout, arrows to drive, Space to stop.")
        while viewer.is_running():
            t0 = time.time()
            if want_new["flag"]:
                want_new["flag"] = False
                new_layout()

            # 1. control: action [linear_vel, angular_vel] -> wheel velocities
            action = controller.get_action()
            controller.apply_control(data, action)

            # 2. physics step
            mujoco.mj_step(model, data)

            # 3. LiDAR scan every step, rays drawn in the viewer
            distances, ray_points = lidar.scan(data, return_points=True)
            with viewer.lock():
                draw_rays(viewer, ray_points)

            # 4. log a scan every LOG_INTERVAL seconds of simulation time
            if data.time - last_log_time >= LOG_INTERVAL:
                logger.log(distances, data.time)
                last_log_time = data.time

            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - t0)))


if __name__ == "__main__":
    main()
