import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from layout_generator import MODES, WALL, Layout, generate_layout, layout_to_str
from make_scene import MODEL_DIR, PARKING_POS, cell_center

from lidar import Lidar
from logger import DataLogger
from controller import RobotController

KEY_N = 78  
FOOTPRINT_CENTER_X = -0.06

class LayoutPlacer:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        name = lambda obj, i: mujoco.mj_id2name(model, obj, i) or ""

        block_bodies = sorted(
            (name(mujoco.mjtObj.mjOBJ_BODY, b), b) for b in range(model.nbody)
            if name(mujoco.mjtObj.mjOBJ_BODY, b).startswith("block_")
        )
        self.block_mocap = [model.body_mocapid[b] for _, b in block_bodies]

        goal_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self.goal_mocap = model.body_mocapid[goal_body]

        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_joint")
        self.robot_qpos = model.jnt_qposadr[joint]
        
        first_block_geom = model.body_geomadr[block_bodies[0][1]]
        half_x, _, half_z = model.geom_size[first_block_geom]
        self.cell_size = 2 * float(half_x)
        self.block_z = float(half_z)                      
        self.grid_size = int(round(np.sqrt(len(self.block_mocap))))

    def apply(self, data: mujoco.MjData, layout: Layout) -> None:
        n = self.grid_size
        mujoco.mj_resetData(self.model, data)   

        wall_cells = np.argwhere(layout.grid == WALL)
        for mocap_id, (row, col) in zip(self.block_mocap, wall_cells):
            x, y = cell_center(int(row), int(col), n, self.cell_size)
            data.mocap_pos[mocap_id] = [x, y, self.block_z]

        for mocap_id in self.block_mocap[len(wall_cells):]:
            data.mocap_pos[mocap_id] = PARKING_POS

        cx, cy = cell_center(*layout.start, n, self.cell_size)
        yaw = layout.start_yaw
        x = cx - FOOTPRINT_CENTER_X * np.cos(yaw)
        y = cy - FOOTPRINT_CENTER_X * np.sin(yaw)
        q = self.robot_qpos
        data.qpos[q:q + 3] = [x, y, 0.0]
        data.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]  
        
        gx, gy = cell_center(*layout.goal, n, self.cell_size)
        data.mocap_pos[self.goal_mocap] = [gx, gy, 0.0]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=MODEL_DIR / "scene_grid.xml")
    parser.add_argument("--mode", choices=MODES, default="random")
    parser.add_argument("--obstacles", type=int, default=12)
    parser.add_argument("--min-dist", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    
    parser.add_argument("--fov", type=float, default=360.0)
    parser.add_argument("--rays", type=int, default=36)
    parser.add_argument("--log-mode", choices=["terminal", "file", "none"], default="none")
    parser.add_argument("--log-file", type=str, default="lidar_data.txt")
    parser.add_argument("--control", choices=["keyboard", "rl"], default="keyboard")
    
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    placer = LayoutPlacer(model)
    rng = np.random.default_rng(args.seed)
    
    lidar = Lidar(model, fov_deg=args.fov, n_rays=args.rays)
    logger = DataLogger(mode=args.log_mode, file_path=args.log_file)
    controller = RobotController(model, mode=args.control)
    
    last_log_time = 0.0
    LOG_INTERVAL = 0.5  

    def new_layout() -> None:
        layout = generate_layout(placer.grid_size, rng, mode=args.mode,
                                 n_obstacles=args.obstacles, min_dist_cells=args.min_dist)
        placer.apply(data, layout)
        mujoco.mj_forward(model, data)

    want_new = {"flag": False}

    def on_key(key: int) -> None:
        if key == KEY_N:
            want_new["flag"] = True
        else:
            controller.handle_key(key)

    new_layout()

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            t0 = time.time()
            if want_new["flag"]:
                want_new["flag"] = False
                new_layout()
            
            # 1. Action input calculation
            action = controller.get_action()
            controller.apply_control(data, action)
            
            # 2. Physics symulation step
            mujoco.mj_step(model, data)
            
            # 3. Lidar readout every frame and optionally visualize rays
            distances, ray_points = lidar.scan(data, return_points=True)
            
            # Geometry visualization in the viewer
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                for start, end in ray_points:
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                        break
                    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, start, end)
                    geom.rgba[:] = [1.0, 0.2, 0.2, 0.8]  # Kolor czerwony
                    viewer.user_scn.ngeom += 1

            # 4. Data logging every LOG_INTERVAL seconds
            if data.time - last_log_time >= LOG_INTERVAL:
                logger.log(distances, data.time)
                last_log_time = data.time
                
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - t0)))

if __name__ == "__main__":
    main()
