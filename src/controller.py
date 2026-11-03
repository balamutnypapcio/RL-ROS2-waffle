import time
import numpy as np
import mujoco

class RobotController:
    def __init__(self, model: mujoco.MjModel, mode: str = "keyboard"):
        self.mode = mode
        self.model = model
        
        # Parametry Waffle Pi
        self.max_speed = 0.26  # m/s
        self.max_turn = 1.82   # rad/s
        self.wheel_radius = 0.033
        self.wheel_separation = 0.287
        
        # Wewnętrzny stan komend (v - prędkość linowa, omega - kątowa)
        self.cmd_v = 0.0
        self.cmd_omega = 0.0
        self.last_key_time = 0.0
        
        if self.model.nu < 2:
            print("ERROR: Less than 2 engines found in the model.")

    def handle_key(self, key: int):
        """Command readout(GLFW codes)."""
        if self.mode != "keyboard":
            return
        
        self.last_key_time = time.time()
        if key == 265:    # ^
            self.cmd_v = self.max_speed
        elif key == 264:  # v
            self.cmd_v = -self.max_speed
        elif key == 263:  # <
            self.cmd_omega = self.max_turn
        elif key == 262:  # >
            self.cmd_omega = -self.max_turn
        elif key == 32:  # Space (Brake)
            self.cmd_v = 0.0
            self.cmd_omega = 0.0

    def get_action(self, obs=None) -> np.ndarray:
        """Returns action [linear_vel, angular_vel] for the current frame."""
        if self.mode == "keyboard":
            # Timeout: relase the key then system stops after 0.15s 
            if time.time() - self.last_key_time > 0.15:
                self.cmd_v = 0.0
                self.cmd_omega = 0.0
            return np.array([self.cmd_v, self.cmd_omega])
        
        elif self.mode == "rl":
            # todo: implement RL-based control
            return np.array([0.0, 0.0])

    def apply_control(self, data: mujoco.MjData, action: np.ndarray):
        """Tłumaczy akcję na kinematykę Differential Drive (na koła)."""
        v, omega = action[0], action[1]
        
        v_left = (v - omega * self.wheel_separation / 2) / self.wheel_radius
        v_right = (v + omega * self.wheel_separation / 2) / self.wheel_radius
        
        if self.model.nu >= 2:
            data.ctrl[0] = v_left
            data.ctrl[1] = v_right
