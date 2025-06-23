#!/usr/bin/env python3
"""
Planner implementations for F1TENTH, including PurePursuitPlanner.
"""

import numpy as np
import pandas as pd
import os

class PurePursuitPlanner:
    def __init__(self, map_path, lookahead_distance=1.5, wheelbase=0.33, max_steer=0.5, max_speed=8.0, curvature_scale=1.0):
        """
        Initialize PurePursuitPlanner.
        
        Args:
            map_path (str): Path to raceline CSV file.
            lookahead_distance (float): Lookahead distance in meters.
            wheelbase (float): Vehicle wheelbase in meters.
            max_steer (float): Maximum steering angle in radians.
            max_speed (float): Maximum speed in m/s.
            curvature_scale (float): Scaling factor for curvature-based speed adjustment.
        """
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.curvature_scale = curvature_scale
        
        # Load raceline
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map file {map_path} not found.")
        try:
            # Leer CSV sin encabezados, usando delimitador ';'
            self.raceline = pd.read_csv(map_path, sep=';', header=None, comment='#', engine='python')
            # Asignar nombres de columnas
            self.raceline.columns = ['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2']
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV {map_path}: {e}")
        
        print(f"CSV columns in {map_path}: {list(self.raceline.columns)}")
        required_columns = ['x_m', 'y_m', 'kappa_radpm', 'vx_mps']
        missing_columns = [col for col in required_columns if col not in self.raceline.columns]
        if missing_columns:
            print(f"CSV content (first 5 rows):\n{self.raceline.head()}")
            raise KeyError(f"Missing columns {missing_columns} in {map_path}")
        
        self.points = np.array([self.raceline['x_m'], self.raceline['y_m']]).T
        self.curvatures = np.array(self.raceline['kappa_radpm'])
        self.speeds = np.array(self.raceline['vx_mps'])

    def compute_action(self, x, y, theta):
        """
        Compute steering angle and speed.
        
        Args:
            x, y (float): Current position in meters.
            theta (float): Current orientation in radians.
        
        Returns:
            steer (float): Steering angle in radians.
            speed (float): Speed in m/s.
        """
        # Find closest point
        current_pos = np.array([x, y]).reshape(1, 2)
        distances = np.linalg.norm(self.points - current_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        # Find lookahead point
        lookahead_idx = closest_idx
        dist = 0.0
        while dist < self.lookahead_distance and lookahead_idx < len(self.points) - 1:
            lookahead_idx += 1
            dist = np.linalg.norm(self.points[lookahead_idx] - current_pos)
        
        lookahead_point = self.points[lookahead_idx]
        
        # Compute steering angle
        alpha = np.arctan2(lookahead_point[1] - y, lookahead_point[0] - x) - theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # Normalize angle
        steer = np.arctan2(2 * self.wheelbase * np.sin(alpha), self.lookahead_distance)
        steer = np.clip(steer, -self.max_steer, self.max_steer)
        
        # Compute speed based on raceline and curvature
        curvature = self.curvatures[closest_idx]
        base_speed = self.speeds[closest_idx]
        speed = min(base_speed, self.max_speed * np.exp(-self.curvature_scale * np.abs(curvature)))
        
        return steer, speed

    def compute_errors(self, x, y, theta):
        """
        Compute cross-track error (CTE) and heading error (HE).
        
        Args:
            x, y (float): Current position in meters.
            theta (float): Current orientation in radians.
        
        Returns:
            cte (float): Cross-track error in meters.
            he (float): Heading error in radians.
        """
        current_pos = np.array([x, y]).reshape(1, 2)
        distances = np.linalg.norm(self.points - current_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        closest_point = self.points[closest_idx]
        cte = distances[closest_idx]
        
        # Approximate heading error
        if closest_idx < len(self.points) - 1:
            next_point = self.points[closest_idx + 1]
            track_heading = np.arctan2(next_point[1] - closest_point[1], next_point[0] - closest_point[0])
        else:
            prev_point = self.points[closest_idx - 1]
            track_heading = np.arctan2(closest_point[1] - prev_point[1], closest_point[0] - prev_point[0])
        
        he = track_heading - theta
        he = np.arctan2(np.sin(he), np.cos(he))  # Normalize angle
        
        return cte, he

    def compute_curvature(self, x, y, theta):
        """
        Compute curvature at the closest point on the raceline.
        
        Args:
            x, y (float): Current position in meters.
            theta (float): Current orientation in radians.
        
        Returns:
            curvature (float): Curvature in rad/m.
        """
        current_pos = np.array([x, y]).reshape(1, 2)
        distances = np.linalg.norm(self.points - current_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        print(f"Closest idx: {closest_idx}, Curvature: {self.curvatures[closest_idx]}")
        
        return self.curvatures[closest_idx]