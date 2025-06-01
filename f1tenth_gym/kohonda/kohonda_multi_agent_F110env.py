import numpy as np
from typing import Tuple, Any
from .utils import nearest_point_on_trajectory, calculate_curvatures
from .custom_vector_observation import MultiAgentVectorObservation
from f1tenth_gym.envs import F110Env

from f1tenth_gym.envs import f110_env
from f1tenth_gym.envs.observation import Observation

# Guarda la original antes de reemplazarla
_original_observation_factory = f110_env.observation_factory

# Redefiní la nueva función que llama a la original solo si no es "multi_rl"


def custom_observation_factory(env, type: str | None, **kwargs) -> Observation:
    type = type or "original"
    if type == "multi_rl":
        features = [
            "scan",
            # Agrega más si los necesitas (x, y, yaw, etc.)
        ]
        return MultiAgentVectorObservation(env, features)
    else:
        return _original_observation_factory(env, type=type, **kwargs)


# Monkey patch
f110_env.observation_factory = custom_observation_factory


def to_array(x):
    try:
        return np.asarray(x)
    except Exception:
        return np.array([x])


class KohondaMultiAgentF110Env(F110Env):
    """
    Extended F110Env with waypoint-based reward and observation logic,
    adapted from Honda 2024 for multi-agent compatibility using gymnasium.
    Enforces original observation dict from F110Env by setting observation_config type to 'original'.
    """

    def __init__(self, config: dict = None, render_mode=None, **kwargs):
        # Don't force observation type, use whatever is configured in sac_example.py
        if config is None:
            config = {}
        # Call parent with original config (don't force 'original' type)
        super().__init__(config=config, render_mode=render_mode, **kwargs)

        # Raceline waypoints
        self._waypoints = np.stack(
            [self.track.raceline.xs, self.track.raceline.ys], axis=-1).astype(np.float32)
        # State per agent
        self._current_waypoints = np.zeros(
            (self.num_agents, 2), dtype=np.float32)
        self._current_indices = np.zeros((self.num_agents,), dtype=int)
        self.prev_waypoints = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.prev_vels = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.prev_steer_angle = np.zeros(self.num_agents, dtype=np.float32)
        self.prev_yaw = np.zeros(self.num_agents, dtype=np.float32)

        # Override observation space to match our custom vectorized observation
        # The observation has 29 elements: 6 velocities + 1 yaw_dev + 18 depths + 4 additional features
        import gymnasium as gym
        if self.num_agents == 1:
            # Single agent: return Box observation directly
            self.observation_space = gym.spaces.Box(
                low=-1e30, high=1e30, shape=(29,), dtype=np.float32
            )
        else:
            # Multi-agent: return list of Box observations
            single_agent_space = gym.spaces.Box(
                low=-1e30, high=1e30, shape=(29,), dtype=np.float32
            )
            self.observation_space = gym.spaces.Tuple(
                [single_agent_space] * self.num_agents)

    def calc_current_waypoint(self, idx: int) -> Tuple[np.ndarray, int]:
        pos = np.array([self.poses_x[idx], self.poses_y[idx]],
                       dtype=self._waypoints.dtype)
        pt, _, _, index = nearest_point_on_trajectory(
            point=pos, trajectory=self._waypoints)
        return pt, index

    def _get_reward(self) -> float:
        total = 0.0
        for i in range(self.num_agents):
            dist = np.linalg.norm(
                self._current_waypoints[i] - self.prev_waypoints[i])
            pen = 0.0
            if self.collisions[i] > 0:
                v = self.prev_vels[i]
                pen = -0.05 * np.dot(v, v)
            r = dist + pen
            if not np.isfinite(r):
                print(
                    f"[WARN] Agent {i} invalid reward dist={dist}, pen={pen}")
                r = -1.0
            total += r
        return total

    def _observation(self, idx: int, original_obs: np.ndarray) -> np.ndarray:
        # original_obs is from VectorObservation (type "rl") which returns just scan data
        # We need to extract scan and augment with our custom observations

        # Get scan data from the original observation
        scan_data = original_obs  # This should be the scan array

        # Get additional data from environment state
        vx = float(self.poses_x[idx] - self.prev_waypoints[idx][0]) / \
            self.timestep if hasattr(self, 'timestep') else 0.0
        vy = float(self.poses_y[idx] - self.prev_waypoints[idx][1]) / \
            self.timestep if hasattr(self, 'timestep') else 0.0
        ang_v = 0.0  # We'll compute this differently

        dvx = vx - self.prev_vels[idx][0]
        dvy = vy - self.prev_vels[idx][1]

        yaw = float(self.poses_theta[idx])
        index = self._current_indices[idx]
        next_idx = (index + 1) % len(self._waypoints)
        dx = self._waypoints[next_idx, 0] - self._waypoints[index, 0]
        dy = self._waypoints[next_idx, 1] - self._waypoints[index, 1]
        yaw_ref = np.arctan2(dy, dx)
        yaw_dev = np.arctan2(np.sin(yaw - yaw_ref), np.cos(yaw - yaw_ref))

        # Reduce scan to 18 elements and ensure it's a 1D array
        if len(scan_data) > 18:
            idxs = np.linspace(0, len(scan_data) - 1, num=18, dtype=int)
            depths = scan_data[idxs]
        else:
            depths = scan_data

        # Flatten and ensure we have exactly 18 elements
        depths = np.asarray(depths).flatten()
        if len(depths) < 18:
            # Pad with the last value if we have fewer than 18 elements
            depths = np.pad(depths, (0, 18 - len(depths)), mode='edge')
        elif len(depths) > 18:
            # Truncate if we have more than 18 elements
            depths = depths[:18]

        ahead = np.stack([self._waypoints[(index + j) %
                         len(self._waypoints)] for j in range(10)])
        curvs = calculate_curvatures(ahead)
        path_curv = curvs[0]

        delta = yaw - self.prev_yaw[idx]
        delta_yaw = np.arctan2(np.sin(delta), np.cos(delta))

        # Create vectorized observation (29 elements total)
        obs = np.zeros(29, dtype=np.float32)
        obs[0:6] = [vx, vy, ang_v, dvx, dvy, ang_v]
        obs[6] = yaw_dev
        obs[7:25] = depths  # Now guaranteed to be exactly 18 elements
        obs[25] = self.prev_steer_angle[idx]
        obs[26] = path_curv
        obs[27] = float(self.collisions[idx])
        obs[28] = delta_yaw

        if not np.all(np.isfinite(obs)):
            raise ValueError(f"Invalid observation agent {idx}: {obs}")

        return obs

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, dict]:
        # raw original_obs from parent (numpy array for "rl" type observation)
        original_obs, _, done, truncated, info = super().step(action)
        obs_list = []
        for i in range(self.num_agents):
            pt, idx = self.calc_current_waypoint(i)
            self._current_waypoints[i] = pt
            self._current_indices[i] = idx
            obs_i = self._observation(i, original_obs)
            obs_list.append(obs_i)
            self.prev_waypoints[i] = pt
            # Get velocities from environment state instead of observation dict
            agent = self.sim.agents[i]
            std_state = agent.standard_state
            self.prev_vels[i] = np.array(
                [std_state["v_x"], std_state["v_y"]], dtype=np.float32)
            # Corrección: action[i] = [velocidad, steering_angle]
            # steering_angle es el segundo elemento
            self.prev_steer_angle[i] = float(action[i][1])
            self.prev_yaw[i] = float(std_state["yaw"])
        reward = self._get_reward()
        # Return appropriate format based on number of agents
        if self.num_agents == 1:
            return obs_list[0], reward, done, truncated, info
        else:
            return obs_list, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        original_obs, info = super().reset(seed=seed, options=options)
        obs_list = []
        for i in range(self.num_agents):
            pt, idx = self.calc_current_waypoint(i)
            self._current_waypoints[i] = pt
            self._current_indices[i] = idx
            self.prev_waypoints[i] = pt
            self.prev_vels[i] = np.zeros(2, dtype=np.float32)
            self.prev_steer_angle[i] = 0.0
            self.prev_yaw[i] = 0.0
            obs_list.append(self._observation(i, original_obs))
        # Return appropriate format based on number of agents
        if self.num_agents == 1:
            return obs_list[0], info
        else:
            return obs_list, info


class VictorMultiAgentEnv(KohondaMultiAgentF110Env):

    def __init__(self, config: dict = None, render_mode=None, **kwargs):
        import gymnasium as gym
        from gymnasium.spaces import Dict as SpaceDict

        if config is None:
            config = {}

        # Llama al constructor base de F110Env
        super().__init__(config=config, render_mode=render_mode, **kwargs)

        # Raceline waypoints
        self._waypoints = np.stack(
            [self.track.raceline.xs, self.track.raceline.ys], axis=-1
        ).astype(np.float32)

        # Estado por agente
        self._current_waypoints = np.zeros(
            (self.num_agents, 2), dtype=np.float32)
        self._current_indices = np.zeros((self.num_agents,), dtype=int)
        self.prev_waypoints = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.prev_vels = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.prev_steer_angle = np.zeros(self.num_agents, dtype=np.float32)
        self.prev_yaw = np.zeros(self.num_agents, dtype=np.float32)

        # Define el espacio de observación
        single_agent_space = gym.spaces.Box(
            low=-1e30, high=1e30, shape=(29,), dtype=np.float32
        )

        if self.num_agents == 1:
            # Single agent: return Box observation directly
            self.observation_space = single_agent_space
        else:
            # Multi-agent: return Dict of Box observations indexed by agent_id
            self.observation_space = SpaceDict({
                agent_id: single_agent_space for agent_id in self.agent_ids
            })

    def _observation(self, idx: int, original_obs) -> np.ndarray:
        # original_obs can be either a dict (type "original") or array (type "rl")
        # We need to extract scan and augment with our custom observations

        # Get scan data from the original observation
        if isinstance(original_obs, dict):
            # For multi-agent, original_obs is a dict with agent keys
            if 'agent_0' in original_obs:
                agent_key = f'agent_{idx}'
                scan_data = original_obs[agent_key]
            else:
                # Single agent case
                scan_data = original_obs['scans']
        else:
            # For type "rl", original_obs is directly the scan array
            scan_data = original_obs

        # Get additional data from environment state
        vx = float(self.poses_x[idx] - self.prev_waypoints[idx][0]) / \
            self.timestep if hasattr(self, 'timestep') else 0.0
        vy = float(self.poses_y[idx] - self.prev_waypoints[idx][1]) / \
            self.timestep if hasattr(self, 'timestep') else 0.0
        ang_v = 0.0  # We'll compute this differently

        dvx = vx - self.prev_vels[idx][0]
        dvy = vy - self.prev_vels[idx][1]

        yaw = float(self.poses_theta[idx])
        index = self._current_indices[idx]
        next_idx = (index + 1) % len(self._waypoints)
        dx = self._waypoints[next_idx, 0] - self._waypoints[index, 0]
        dy = self._waypoints[next_idx, 1] - self._waypoints[index, 1]
        yaw_ref = np.arctan2(dy, dx)
        yaw_dev = np.arctan2(np.sin(yaw - yaw_ref), np.cos(yaw - yaw_ref))

        # Reduce scan to 18 elements and ensure it's a 1D array
        if len(scan_data) > 18:
            idxs = np.linspace(0, len(scan_data) - 1, num=18, dtype=int)
            depths = scan_data[idxs]
        else:
            depths = scan_data

        # Flatten and ensure we have exactly 18 elements
        depths = np.asarray(depths).flatten()
        if len(depths) < 18:
            # Pad with the last value if we have fewer than 18 elements
            depths = np.pad(depths, (0, 18 - len(depths)), mode='edge')
        elif len(depths) > 18:
            # Truncate if we have more than 18 elements
            depths = depths[:18]

        ahead = np.stack([self._waypoints[(index + j) %
                         len(self._waypoints)] for j in range(10)])
        curvs = calculate_curvatures(ahead)
        path_curv = curvs[0]

        delta = yaw - self.prev_yaw[idx]
        delta_yaw = np.arctan2(np.sin(delta), np.cos(delta))

        # Create vectorized observation (29 elements total)
        obs = np.zeros(29, dtype=np.float32)
        obs[0:6] = [vx, vy, ang_v, dvx, dvy, ang_v]
        obs[6] = yaw_dev
        obs[7:25] = depths  # Now guaranteed to be exactly 18 elements
        obs[25] = self.prev_steer_angle[idx]
        obs[26] = path_curv
        obs[27] = float(self.collisions[idx])
        obs[28] = delta_yaw

        if not np.all(np.isfinite(obs)):
            raise ValueError(f"Invalid observation agent {idx}: {obs}")

        return obs

    def reset(self, seed=None, options=None):
        original_obs, info = F110Env.reset(self,
                                           seed=seed, options=options)  # Llamemos al original reset de F110Env
        if self.num_agents == 1:
            pt, idx = self.calc_current_waypoint(0)
            self._current_waypoints[0] = pt
            self._current_indices[0] = idx
            self.prev_waypoints[0] = pt
            self.prev_vels[0] = np.zeros(2, dtype=np.float32)
            self.prev_steer_angle[0] = 0.0
            self.prev_yaw[0] = 0.0
            obs = self._observation(0, original_obs)
            return obs, info

        # Multi-agent: return dict of observations
        obs_dict = {}
        for i, agent_id in enumerate(self.agent_ids):
            pt, idx = self.calc_current_waypoint(i)
            self._current_waypoints[i] = pt
            self._current_indices[i] = idx
            self.prev_waypoints[i] = pt
            self.prev_vels[i] = np.zeros(2, dtype=np.float32)
            self.prev_steer_angle[i] = 0.0
            self.prev_yaw[i] = 0.0
            obs_dict[agent_id] = self._observation(i, original_obs)

        return obs_dict, info

    def step(self, action: dict) -> Tuple[dict, float, bool, bool, dict]:
        original_obs, _, done, truncated, info = super().step(action)

        if self.num_agents == 1:
            pt, idx = self.calc_current_waypoint(0)
            self._current_waypoints[0] = pt
            self._current_indices[0] = idx
            obs = self._observation(0, original_obs)
            agent = self.sim.agents[0]
            std_state = agent.standard_state
            self.prev_vels[0] = np.array(
                [std_state["v_x"], std_state["v_y"]], dtype=np.float32)
            self.prev_steer_angle[0] = float(action[0][1])
            self.prev_yaw[0] = float(std_state["yaw"])
            reward = self._get_reward()
            return obs, reward, done, truncated, info

        # Multi-agent
        obs_dict = {}
        for agent_index, agent_id in enumerate(self.agent_ids):
            pt, idx = self.calc_current_waypoint(agent_index)
            self._current_waypoints[agent_index] = pt
            self._current_indices[agent_index] = idx
            obs_i = self._observation(agent_index, original_obs)
            obs_dict[agent_id] = obs_i
            self.prev_waypoints[agent_index] = pt
            agent = self.sim.agents[agent_index]
            std_state = agent.standard_state
            self.prev_vels[agent_index] = np.array(
                [std_state["v_x"], std_state["v_y"]], dtype=np.float32)

            self.prev_steer_angle[agent_index] = float(action[agent_index][1])
            self.prev_yaw[agent_index] = float(std_state["yaw"])

        reward = self._get_reward()
        return obs_dict, reward, done, truncated, info

    def _get_reward(self):
        # Parámetros de la política
        PENALIZACION_QUIETO = -5.0  # Penalización fuerte por quedarse quieto
        PREMIO_VEL_CONST = 2.0      # Premio por velocidad constante deseada
        PREMIO_VUELTA = 100.0       # Premio muy alto por vuelta completada
        VEL_MIN = 7.0
        VEL_MAX = 10.0
        PREMIO_ACERCARSE_WP = 0.5   # Premio por acercarse al siguiente waypoint
        PENALIZACION_DESVIO = -1.0  # Penalización progresiva por desviación lateral
        PREMIO_PASO_SIN_COLISION = 0.1  # Premio pequeño por cada paso sin colisión
        COLISION_FACTOR = 500  # Factor de penalización por colisión

        total = 0.0
        for i in range(self.num_agents):
            # Distancia recorrida desde el último paso
            dist = np.linalg.norm(
                self._current_waypoints[i] - self.prev_waypoints[i])
            # Velocidad actual
            v = self.prev_vels[i]
            speed = np.linalg.norm(v)
            # Penalización si está quieto
            if speed < 1e-2:
                dist += PENALIZACION_QUIETO
            # Premio por velocidad constante en rango deseado
            if VEL_MIN <= speed <= VEL_MAX:
                dist += PREMIO_VEL_CONST
            # Penalización por colisión (igual que el padre)
            pen = 0.0
            if self.collisions[i] > 0:
                pen = COLISION_FACTOR * np.dot(v, v)
            # Premio por vuelta completada (si el índice de waypoint se reinicia)
            idx_actual = self._current_indices[i]
            idx_prev = getattr(self, '_prev_indices', [0]*self.num_agents)[i]
            vuelta_completada = idx_actual < idx_prev  # Se reinició el índice
            if vuelta_completada:
                dist += PREMIO_VUELTA
            # Guardar el índice actual para la próxima llamada
            if not hasattr(self, '_prev_indices'):
                self._prev_indices = [0]*self.num_agents
            self._prev_indices[i] = idx_actual

            # --- Recompensa adicional 1: Premio por acercarse al siguiente waypoint ---
            # Calcula la distancia al siguiente waypoint antes y después
            next_idx = (idx_prev + 1) % len(self._waypoints)
            pos_prev = self.prev_waypoints[i]
            pos_now = self._current_waypoints[i]
            wp_next = self._waypoints[next_idx]
            dist_prev = np.linalg.norm(pos_prev - wp_next)
            dist_now = np.linalg.norm(pos_now - wp_next)
            if dist_now < dist_prev:
                dist += PREMIO_ACERCARSE_WP

            # --- Recompensa adicional 2: Penalización progresiva por desviación lateral ---
            # Calcula la distancia lateral al raceline (waypoint actual)
            lateral_dev = np.linalg.norm(pos_now - wp_next)
            dist += PENALIZACION_DESVIO * lateral_dev

            # --- Recompensa adicional 3: Premio por paso sin colisión ---
            if self.collisions[i] == 0:
                dist += PREMIO_PASO_SIN_COLISION

            # Suma total
            r = dist + pen
            if not np.isfinite(r):
                print(
                    f"[WARN] Agent {i} invalid reward dist={dist}, pen={pen}")
                r = -1.0
            total += r
        return total
