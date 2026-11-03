import math
import numpy as np

def goal_distance_angle(robot_xy: np.ndarray,
                         robot_yaw: float, 
                         goal_xy: np.ndarray) -> tuple[float, float]:

    dx, dy = goal_xy - robot_xy 
    distance = math.hypot(dx,dy)
    angle = math.atan2(dy, dx) - robot_yaw
    angle = math.atan2(math.sin(angle), math.cos(angle))  # betwen [-π, π]

    return distance, angle