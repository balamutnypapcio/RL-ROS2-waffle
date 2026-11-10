import numpy as np

class DataLogger:
    def __init__(self, mode: str, file_path: str = "lidar_data.txt"):
        self.mode = mode
        self.file_path = file_path
        
        if self.mode == "file":
            with open(self.file_path, "w") as f:
                f.write("--- Start symulacji ---\n")

    def log(self, scan_data: np.ndarray, timestamp: float):
        if self.mode == "terminal":
            formatted = np.round(scan_data, 2)
            print(f"[Time: {timestamp:.2f}s] Lidar scan: {formatted}")
            
        elif self.mode == "file":
            with open(self.file_path, "a") as f:
                scan_str = ",".join(f"{x:.4f}" for x in scan_data)
                f.write(f"{timestamp:.4f},{scan_str}\n")
                
        elif self.mode == "none":
            pass
