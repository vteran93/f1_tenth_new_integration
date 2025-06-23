
import numpy as np
import pandas as pd

class PurePursuitPlanner:
    def __init__(self, map_path, lookahead_distance, wheelbase, max_steer):
        self.map_data = pd.read_csv(map_path, sep=';')
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.max_steer = max_steer

    def get_control(self, x, y, theta, speed):
        closest_idx = np.argmin(np.sqrt((self.map_data['x_m'] - x)**2 + (self.map_data['y_m'] - y)**2))
        target_x, target_y = self.map_data['x_m'][closest_idx], self.map_data['y_m'][closest_idx]
        dx = target_x - x
        dy = target_y - y
        alpha = np.arctan2(dy, dx) - theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        steer = np.arctan2(2 * self.wheelbase * np.sin(alpha), self.lookahead_distance)
        steer = np.clip(steer, -self.max_steer, self.max_steer)
        return speed, steer
