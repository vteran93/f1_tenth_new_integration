from typing import List
from gymnasium.spaces import Dict as SpaceDict
import gymnasium as gym
from f1tenth_gym.envs.observation import Observation
import numpy as np


class MultiAgentVectorObservation(Observation):
    """
    Custom observation class that returns a vector observation.
    This class is used to create a vector observation from the environment's state.
    """

    def __init__(self, env, features: List[str]):
        super().__init__(env)
        self.features = features

    def space(self):
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        large_num = 1e30

        num_agents = len(self.env.unwrapped.agent_ids)

        obs_size_dict = {
            "scan": scan_size,
            "pose_x": 1,
            "pose_y": 1,
            "pose_theta": 1,
            "linear_vel_x": 1,
            "linear_vel_y": 1,
            "ang_vel_z": 1,
            "delta": 1,
            "beta": 1,
            "collision": 1,
            "lap_time": 1,
            "lap_count": 1,
        }

        complete_space_size = sum(obs_size_dict[k] for k in self.features)

        single_obs_space = gym.spaces.Box(
            low=-large_num,
            high=large_num,
            shape=(complete_space_size,),
            dtype=float,
        )

        if num_agents == 1:
            print("type of single_obs_space:", type(single_obs_space))
            return single_obs_space
        else:
            return SpaceDict({
                agent_id: single_obs_space for agent_id in self.env.unwrapped.agent_ids
            })

    def observe(self):
        multi_obs = []
        num_agents = len(self.env.unwrapped.agent_ids)

        for i in range(num_agents):
            scan = self.env.unwrapped.sim.agent_scans[i]
            agent = self.env.unwrapped.sim.agents[i]
            lap_time = self.env.unwrapped.lap_times[i]
            lap_count = self.env.unwrapped.lap_counts[i]

            std_state = agent.standard_state

            x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]
            delta = std_state["delta"]
            beta = std_state["slip"]
            vx = std_state["v_x"]
            vy = std_state["v_y"]
            angvel = std_state["yaw_rate"]

            agent_obs = {
                "scan": scan,
                "pose_x": x,
                "pose_y": y,
                "pose_theta": theta,
                "linear_vel_x": vx,
                "linear_vel_y": vy,
                "ang_vel_z": angvel,
                "delta": delta,
                "beta": beta,
                "collision": int(agent.in_collision),
                "lap_time": lap_time,
                "lap_count": lap_count,
            }

            vec_obs = []
            for k in self.features:
                vec_obs.extend(list(agent_obs[k]) if hasattr(
                    agent_obs[k], '__iter__') and not isinstance(agent_obs[k], str) else [agent_obs[k]])

            multi_obs.append(np.array(vec_obs, dtype=np.float32))

        return multi_obs
