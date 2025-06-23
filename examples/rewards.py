
import numpy as np
import pandas as pd

class RacePerformanceReward:
    def __init__(self, progress_scale, speed_scale, collision_penalty, overtake_bonus, stall_penalty, jerk_penalty, min_speed_threshold):
        self.progress_scale = progress_scale
        self.speed_scale = speed_scale
        self.collision_penalty = collision_penalty
        self.overtake_bonus = overtake_bonus
        self.stall_penalty = stall_penalty
        self.jerk_penalty = jerk_penalty
        self.min_speed_threshold = min_speed_threshold
        self.prev_progress = {}
        self.prev_speed = {}
        self.prev_actions = {}

    def reset(self, positions):
        self.prev_progress = {i: 0.0 for i in range(len(positions))}
        self.prev_speed = {i: 0.0 for i in range(len(positions))}
        self.prev_actions = {i: np.zeros(2) for i in range(len(positions))}

    def __call__(self, state, agent_id):
        agent_idx = int(agent_id.split('_')[1])
        # Usar la acción de velocidad como proxy si linear_vels_x/y no están disponibles
        speed = state['actions'][0]  # Usar speed de la acción
        collision = state['collisions'][agent_idx]
        actions = state['actions']
        prev_actions = state['prev_actions']
        # Estimar progreso basado en lap_counts y poses_x
        progress = state['lap_counts'][agent_idx] + state['poses_x'][agent_idx] / 100

        progress_reward = self.progress_scale * (progress - self.prev_progress[agent_idx])
        speed_reward = self.speed_scale * (speed - self.min_speed_threshold) if speed > self.min_speed_threshold else -self.stall_penalty
        collision_reward = -self.collision_penalty if collision else 0.0
        jerk = np.sum(np.abs(actions - prev_actions)) / len(actions)
        jerk_reward = -self.jerk_penalty * jerk
        overtake_reward = 0.0
        for i in range(len(state['poses_x'])):
            if i != agent_idx and state['lap_counts'][agent_idx] > state['lap_counts'][i]:
                overtake_reward += self.overtake_bonus

        total_reward = progress_reward + speed_reward + collision_reward + overtake_reward + jerk_reward
        components = {
            'progress': progress_reward,
            'speed': speed_reward,
            'collision': collision_reward,
            'overtake': overtake_reward,
            'jerk': jerk_reward
        }

        self.prev_progress[agent_idx] = progress
        self.prev_speed[agent_idx] = speed
        self.prev_actions[agent_idx] = actions

        return total_reward, components

class CrossTrackHeadReward:
    def __init__(self, map_path, lookahead_distance, wheelbase, max_steer, cte_scale, he_scale, max_cte, max_he, collision_penalty):
        # Asignar nombres a las columnas al cargar el CSV
        self.map_data = pd.read_csv(map_path, sep=';', names=['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'v_mps', 'unused'])
        print(f"Map data columns: {self.map_data.columns}")  # Para depuración
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.cte_scale = cte_scale
        self.he_scale = he_scale
        self.max_cte = max_cte
        self.max_he = max_he
        self.collision_penalty = collision_penalty
        self.prev_cte = {}
        self.prev_he = {}

    def reset(self, positions):
        self.prev_cte = {i: 0.0 for i in range(len(positions))}
        self.prev_he = {i: 0.0 for i in range(len(positions))}

    def __call__(self, state, agent_id):
        agent_idx = int(agent_id.split('_')[1])
        x, y, theta = state['poses_x'][agent_idx], state['poses_y'][agent_idx], state['poses_theta'][agent_idx]
        collision = state['collisions'][agent_idx]

        closest_idx = np.argmin(np.sqrt((self.map_data['x_m'] - x)**2 + (self.map_data['y_m'] - y)**2))
        target_x, target_y = self.map_data['x_m'][closest_idx], self.map_data['y_m'][closest_idx]
        target_theta = self.map_data['psi_rad'][closest_idx]
        cte = np.sqrt((x - target_x)**2 + (y - target_y)**2)
        he = theta - target_theta
        he = np.arctan2(np.sin(he), np.cos(he))

        cte_reward = -self.cte_scale * min(cte, self.max_cte)
        he_reward = -self.he_scale * min(abs(he), self.max_he)
        collision_reward = -self.collision_penalty if collision else 0.0

        total_reward = cte_reward + he_reward + collision_reward
        components = {
            'cte': cte_reward,
            'he': he_reward,
            'collision': collision_reward
        }

        self.prev_cte[agent_idx] = cte
        self.prev_he[agent_idx] = he

        return total_reward, components
