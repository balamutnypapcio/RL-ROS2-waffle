import numpy as np
import mujoco


MAX_RANGE = 1.5  # meters

class Lidar:
    def __init__(self, model: mujoco.MjModel, fov_deg: float, n_rays: int, max_range: float = MAX_RANGE):
        self.model = model
        self.fov = np.radians(fov_deg)
        self.n_rays = n_rays
        self.max_range = max_range
        
        self.robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    def scan(self, data: mujoco.MjData, return_points: bool = False):
        pos = data.xpos[self.robot_body_id]
        mat = data.xmat[self.robot_body_id].reshape(3, 3)
        
        forward = mat[:, 0]
        left = mat[:, 1]
        
        start_pos = pos.copy()
        start_pos[2] += 0.2  

        distances = np.zeros(self.n_rays, dtype=np.float32)
        angles = np.linspace(-self.fov/2, self.fov/2, self.n_rays)
        
        points = [] if return_points else None

        for i, angle in enumerate(angles):
            ray_dir = np.cos(angle) * forward + np.sin(angle) * left
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            
            geomid = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(self.model, data, start_pos, ray_dir, None, 1, -1, geomid)
            
            if dist < 0 or dist > self.max_range:
                dist = self.max_range
            distances[i] = dist
            
            if return_points:
                end_pos = start_pos + ray_dir * dist
                points.append((start_pos.copy(), end_pos))
                
        if return_points:
            return distances, points
        return distances
