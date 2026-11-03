"""Simulated 2D LiDAR (LDS-01 on the TurtleBot3 Waffle Pi) built on mujoco.mj_ray.

Rays are cast in the horizontal plane of the scanner, from the real position of the
LDS on the robot. Angle 0 = robot front (+x), positive angles turn left (+y).

scan() returns, for every ray, the distance from the scanner axis to the first hit,
or NO_HIT (-1) when nothing was found within max_range.
"""
import numpy as np
import mujoco


MAX_RANGE = 1.5  # meters
NO_HIT = -1.0    # returned for rays that hit nothing within max_range

# Scanner axis in the frame of the robot body "base": the origin of the 'lds' mesh
# in turtlebot3_waffle_pi.xml (geom pos="-0.049 0 0.1255"). The LDS housing spans
# z = 0.095 .. 0.135 m, the top plate of the base ends at z = 0.119 m.
LIDAR_POS = np.array([-0.049, 0.0, 0.1255])

# Rays start this far from the scanner axis, so they don't hit the LDS housing itself
# (it reaches 0.048 m from the axis at scan height). The offset is added back to the
# measured distance, so distances are still measured from the axis.
RAY_START_OFFSET = 0.06  # meters


class Lidar:
    def __init__(self, model: mujoco.MjModel, fov_deg: float, n_rays: int, max_range: float = MAX_RANGE):
        self.model = model
        self.fov = np.radians(fov_deg)
        self.n_rays = n_rays
        self.max_range = max_range

        self.robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if self.robot_body_id < 0:
            raise ValueError("No body named 'base' in the model")

        # For a full circle drop the last angle, otherwise -180 and +180 deg are the same ray
        endpoint = not np.isclose(fov_deg % 360.0, 0.0)
        self.angles = np.linspace(-self.fov / 2, self.fov / 2, n_rays, endpoint=endpoint)

    def scan(self, data: mujoco.MjData, return_points: bool = False):
        """Cast all rays. With return_points also returns (start, end, hit) per ray for drawing."""
        mat = data.xmat[self.robot_body_id].reshape(3, 3)
        center = data.xpos[self.robot_body_id] + mat @ LIDAR_POS

        forward = mat[:, 0]
        left = mat[:, 1]

        distances = np.full(self.n_rays, NO_HIT, dtype=np.float32)
        points = [] if return_points else None
        geomid = np.array([-1], dtype=np.int32)

        for i, angle in enumerate(self.angles):
            ray_dir = np.cos(angle) * forward + np.sin(angle) * left
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            start = center + RAY_START_OFFSET * ray_dir

            # geomgroup=None: all groups, flg_static=1: include walls, bodyexclude=-1: none
            hit = mujoco.mj_ray(self.model, data, start, ray_dir, None, 1, -1, geomid)
            dist = hit + RAY_START_OFFSET if hit >= 0 else np.inf
            found = dist <= self.max_range
            if found:
                distances[i] = dist

            if return_points:
                end = center + ray_dir * (dist if found else self.max_range)
                points.append((center.copy(), end, found))

        if return_points:
            return distances, points
        return distances
