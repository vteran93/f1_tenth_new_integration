
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
        self.prev_speed = None
        self.prev_action = None
        self.last_positions = None
        self.lap_count = 0
        self.agent_positions = None

    def reset(self, positions):
        self.last_positions = positions
        self.prev_speed = None
        self.prev_action = None
        self.lap_count = 0
        self.agent_positions = np.argsort([p[0] for p in positions])[::-1]

    def __call__(self, state: dict, agent_id: str) -> tuple[float, dict]:
        idx = int(agent_id.split('_')[1])
        progress = 0.0
        if self.last_positions is not None:
            last_x, last_y = self.last_positions[idx]
            curr_x, curr_y = state['poses_x'][idx], state['poses_y'][idx]
            distance = np.sqrt((curr_x - last_x)**2 + (curr_y - last_y)**2)
            progress = self.progress_scale * distance
        speed = np.sqrt(state['linear_vels_x'][idx]**2 + state['linear_vels_y'][idx]**2)
        collision = state['collisions'][idx]
        overtake = 0.0
        if self.agent_positions is not None and len(self.agent_positions) > 1:
            new_x_positions = state['poses_x']
            new_agent_positions = np.argsort(new_x_positions)[::-1]
            old_pos = np.where(self.agent_positions == idx)[0][0]
            new_pos = np.where(new_agent_positions == idx)[0][0]
            if new_pos < old_pos:
                overtake = self.overtake_bonus * (old_pos - new_pos)
            self.agent_positions = new_agent_positions
        stall = 1.0 if speed < self.min_speed_threshold else 0.0
        jerk = 0.0
        if self.prev_action is not None:
            jerk = np.sum(np.abs(state['actions'] - self.prev_action))
        lap = 0.0
        if self.last_positions is not None and state['poses_x'][idx] - self.last_positions[idx][0] < -10.0:
            self.lap_count += 1
            lap = 100.0 * self.lap_count
        components = {
            'progress': progress,
            'speed': self.speed_scale * (speed - self.min_speed_threshold),
            'collision': -self.collision_penalty * collision,
            'overtake': overtake,
            'stall': -self.stall_penalty * stall,
            'jerk': -self.jerk_penalty * jerk,
            'lap': lap
        }
        reward = sum(components.values())
        self.prev_speed = speed
        self.prev_action = state['actions'].copy()
        self.last_positions = [(state['poses_x'][i], state['poses_y'][i]) for i in range(len(state['poses_x']))]
        return reward, components

class CrossTrackHeadReward:
    def __init__(self, map_path, lookahead_distance, wheelbase, max_steer, cte_scale, he_scale, max_cte, max_he, collision_penalty):
        self.map_path = map_path
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.cte_scale = cte_scale
        self.he_scale = he_scale
        self.max_cte = max_cte
        self.max_he = max_he
        self.collision_penalty = collision_penalty
        self.raceline = pd.read_csv(map_path, sep=';', comment='#', header=None, names=['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2']).to_numpy()
        self.points = self.raceline[:, 1:3]  # x_m, y_m
        self.thetas = self.raceline[:, 3]  # psi_rad
        self.length = len(self.points)

    def get_reference(self, pose_x, pose_y, pose_theta):
        pose = np.array([pose_x, pose_y])
        distances = np.linalg.norm(self.points - pose, axis=1)
        closest_idx = np.argmin(distances)
        current_s = closest_idx
        total_dist = 0.0
        while total_dist < self.lookahead_distance and current_s < self.length - 1:
            next_s = (current_s + 1) % self.length
            segment_dist = np.linalg.norm(self.points[next_s] - self.points[current_s])
            total_dist += segment_dist
            current_s = next_s
        ref_point = self.points[current_s]
        ref_theta = self.thetas[current_s]
        cte = np.cross(ref_point - pose, np.array([np.cos(ref_theta), np.sin(ref_theta)]))
        he = pose_theta - ref_theta
        he = np.arctan2(np.sin(he), np.cos(he))
        return cte, he

    def __call__(self, state: dict, agent_id: str) -> tuple[float, dict]:
        idx = int(agent_id.split('_')[1])
        pose_x = state['poses_x'][idx]
        pose_y = state['poses_y'][idx]
        pose_theta = state['poses_theta'][idx]
        collision = state['collisions'][idx]
        cte, he = self.get_reference(pose_x, pose_y, pose_theta)
        cte = np.clip(cte, -self.max_cte, self.max_cte)
        he = np.clip(he, -self.max_he, self.max_he)
        components = {
            'cte': -self.cte_scale * abs(cte),
            'he': -self.he_scale * abs(he),
            'collision': -self.collision_penalty * collision
        }
        reward = sum(components.values())
        return reward, components
