import math

import numpy as np
import pytest

from tb3_nav.geometry import goal_distance_angle


@pytest.mark.parametrize("robot_yaw, goal_xy, expected_angle", [
    (0.0, (1.0, 0.0), 0.0),                   # goal straight ahead
    (0.0, (0.0, 1.0), math.pi / 2),           # goal on the left
    (math.pi / 2, (0.0, 1.0), 0.0),           # robot already faces the goal
    (0.0, (0.0, -1.0), -math.pi / 2),         # goal on the right
])
def test_angle_in_robot_frame(robot_yaw, goal_xy, expected_angle):
    distance, angle = goal_distance_angle(np.zeros(2), robot_yaw, np.array(goal_xy))
    assert distance == pytest.approx(1.0)
    assert angle == pytest.approx(expected_angle)


def test_goal_behind_is_plus_or_minus_pi():
    _, angle = goal_distance_angle(np.zeros(2), 0.0, np.array([-1.0, 0.0]))
    assert abs(angle) == pytest.approx(math.pi)


def test_angle_is_wrapped_to_pi():
    # world angle -153 deg, robot yaw +172 deg -> -325 deg before wrapping = +35 deg
    _, angle = goal_distance_angle(np.zeros(2), 3.0, np.array([-1.0, -0.5]))
    assert -math.pi <= angle <= math.pi
    assert math.degrees(angle) == pytest.approx(34.7, abs=0.1)
