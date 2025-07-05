import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add the examples directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.rewards import (
    ProgressRewardEnv,
    SpeedRewardEnv,
    SACBasicReward,
    SACGeminiReward,
    SpeedReward,
    SafetyReward,
)
from lib.multiagent_env import MultiAgentF110


class TestRewards(unittest.TestCase):
    def setUp(self):
        # Mock the F110Env
        self.f110_env = MagicMock()
        self.f110_env.num_agents = 2
        self.f110_env.poses_x = np.array([1.0, 2.0])
        self.f110_env.poses_y = np.array([3.0, 4.0])
        self.f110_env.linear_vels_x = np.array([5.0, 6.0])
        self.f110_env.linear_vels_y = np.array([7.0, 8.0])
        self.f110_env.scans = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        self.f110_env.collisions = np.array([False, False])
        self.f110_env.track.centerline.spline.calc_arclength_inaccurate.return_value = (10.0, 0.0)
        self.f110_env.track.centerline.spline.s = np.array([0, 100])
        self.f110_env.track.raceline.spline.calc_arclength_inaccurate.return_value = (10.0, 0.0)
        self.f110_env.track.raceline.spline.s = np.array([0, 100])
        self.f110_env.config = {"timestep": 0.01}

        # Mock the MultiAgentF110 wrapper for ProgressRewardEnv and SpeedRewardEnv
        self.multi_agent_env = MagicMock(spec=MultiAgentF110)
        self.multi_agent_env.env = self.f110_env
        self.multi_agent_env.num_agents = self.f110_env.num_agents
        self.multi_agent_env.agents = [f"agent_{i}" for i in range(self.f110_env.num_agents)]
        self.multi_agent_env._crashed_agents = set()
        self.multi_agent_env._last_s = [0.0] * self.f110_env.num_agents
        self.multi_agent_env.config = self.f110_env.config
        # Directly use the mocked f110_env attributes
        self.multi_agent_env.poses_x = self.f110_env.poses_x
        self.multi_agent_env.poses_y = self.f110_env.poses_y
        self.multi_agent_env.track = self.f110_env.track


    def test_progress_reward(self):
        with patch('f1tenth_gym.envs.F110Env', return_value=self.f110_env):
            reward_env = ProgressRewardEnv(env_config={})
            
            # Manually set attributes that would be set by MultiAgentF110.__init__
            reward_env.agents = self.multi_agent_env.agents
            reward_env._crashed_agents = self.multi_agent_env._crashed_agents
            reward_env._last_s = self.multi_agent_env._last_s
            reward_env.env = self.f110_env

            try:
                rewards = reward_env._get_rewards(set())
                self.assertEqual(len(rewards), self.f110_env.num_agents)
            except TypeError as e:
                self.fail(f"ProgressRewardEnv._get_rewards raised TypeError unexpectedly: {e}")

    def test_speed_reward_env(self):
        with patch('f1tenth_gym.envs.F110Env', return_value=self.f110_env):
            reward_env = SpeedRewardEnv(env_config={})

            # Manually set attributes
            reward_env.agents = self.multi_agent_env.agents
            reward_env._crashed_agents = self.multi_agent_env._crashed_agents
            reward_env._last_s = self.multi_agent_env._last_s
            reward_env.env = self.f110_env
            reward_env.last_positions = {f"agent_{i}": (0.0, 0.0) for i in range(self.f110_env.num_agents)}
            reward_env.timestep = 0.01

            try:
                rewards = reward_env._get_rewards(set())
                self.assertEqual(len(rewards), self.f110_env.num_agents)
            except TypeError as e:
                self.fail(f"SpeedRewardEnv._get_rewards raised TypeError unexpectedly: {e}")

    def _test_base_reward(self, reward_class):
        reward_env = reward_class()
        
        # Mock the env passed to _get_rewards
        mock_env = MagicMock()
        mock_env.env = self.f110_env
        mock_env.agents = [f"agent_{i}" for i in range(self.f110_env.num_agents)]
        mock_env._crashed_agents = set()
        mock_env._last_s = [0.0] * self.f110_env.num_agents

        try:
            rewards = reward_env._get_rewards(mock_env, set())
            self.assertEqual(len(rewards), self.f110_env.num_agents)
        except TypeError as e:
            self.fail(f"{reward_class.__name__}._get_rewards raised TypeError unexpectedly: {e}")

    def test_sac_basic_reward(self):
        self._test_base_reward(SACBasicReward)

    def test_sac_gemini_reward(self):
        self._test_base_reward(SACGeminiReward)

    def test_speed_reward(self):
        self._test_base_reward(SpeedReward)

    def test_safety_reward(self):
        self._test_base_reward(SafetyReward)

if __name__ == '__main__':
    unittest.main()
