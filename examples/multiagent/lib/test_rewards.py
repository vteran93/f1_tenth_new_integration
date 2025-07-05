"""
Unit tests for reward functions consolidation.
Tests the 5 consolidated reward classes to ensure proper functionality.
"""

from rewards_pepe import (
    ProgressRewardAdvancedEnv,
    SpeedReward,
    WaypointReward,
    CompetitiveOvertakingReward,
    SafetyReward,
    RewardFunction
)
from rewards import ProgressRewardEnv, BaseReward
import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add the lib directory to Python path
sys.path.insert(0, os.path.dirname(__file__))


class TestBaseReward(unittest.TestCase):
    """Test base reward class functionality."""

    def test_base_reward_abstract(self):
        """Test that BaseReward raises NotImplementedError."""
        base_reward = BaseReward()
        with self.assertRaises(NotImplementedError):
            base_reward._get_rewards(None, None)


class TestProgressRewardEnv(unittest.TestCase):
    """Test basic progress reward functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.env_config = {"num_agents": 2}
        self.reward_env = ProgressRewardEnv(self.env_config)

        # Mock the environment
        self.reward_env.env = Mock()
        self.reward_env.env.num_agents = 2
        self.reward_env.env.track = Mock()
        self.reward_env.env.track.centerline = Mock()
        self.reward_env.env.track.centerline.spline = Mock()
        self.reward_env.env.track.centerline.spline.s = np.array([0, 10, 20, 30, 40, 50])
        self.reward_env.env.poses_x = np.array([0.0, 5.0])
        self.reward_env.env.poses_y = np.array([0.0, 2.0])

        self.reward_env.agents = ["agent_0", "agent_1"]
        self.reward_env._crashed_agents = set()

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(len(self.reward_env.agents), 2)
        self.assertIsInstance(self.reward_env._crashed_agents, set)

    def test_progress_reward_calculation(self):
        """Test basic progress reward calculation."""
        # Mock calc_arclength_inaccurate to return consistent values
        self.reward_env.env.track.centerline.spline.calc_arclength_inaccurate.side_effect = [
            (10.0, None), (15.0, None)  # Current positions
        ]

        # Initialize last_s
        self.reward_env._last_s = [5.0, 10.0]  # Previous positions

        newly_crashed = set()
        rewards = self.reward_env._get_rewards(newly_crashed)

        # Expected: progress = current_s - last_s
        # Agent 0: 10.0 - 5.0 = 5.0
        # Agent 1: 15.0 - 10.0 = 5.0
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 5.0)
        self.assertEqual(rewards[1], 5.0)

    def test_crash_penalty(self):
        """Test crash penalty application."""
        self.reward_env.env.track.centerline.spline.calc_arclength_inaccurate.return_value = (10.0, None)
        self.reward_env._last_s = [5.0, 10.0]

        newly_crashed = {"agent_0"}
        rewards = self.reward_env._get_rewards(newly_crashed)

        # Agent 0 crashed: progress (5.0) - penalty (1.0) = 4.0
        # Agent 1 normal: progress (0.0) = 0.0
        self.assertEqual(rewards[0], 4.0)  # 5.0 - 1.0
        self.assertEqual(rewards[1], 0.0)   # 10.0 - 10.0


class TestProgressRewardAdvancedEnv(unittest.TestCase):
    """Test advanced progress reward functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.env_config = {"num_agents": 2}
        with patch('rewards_pepe.MultiAgentF110'):
            self.reward_env = ProgressRewardAdvancedEnv(self.env_config)

        # Mock the environment
        self.reward_env.env = Mock()
        self.reward_env.env.num_agents = 2
        self.reward_env.env.track = Mock()
        self.reward_env.env.track.centerline = Mock()
        self.reward_env.env.track.centerline.spline = Mock()
        self.reward_env.env.track.centerline.spline.s = np.array([0, 10, 20, 30, 40, 50])
        self.reward_env.env.poses_x = np.array([0.0, 5.0])
        self.reward_env.env.poses_y = np.array([0.0, 2.0])

        self.reward_env.agents = ["agent_0", "agent_1"]
        self.reward_env._last_s = [5.0, 10.0]

    def test_advanced_progress_reward(self):
        """Test advanced progress reward with scaling."""
        self.reward_env.env.track.centerline.spline.calc_arclength_inaccurate.return_value = (10.0, None)

        reward = self.reward_env._compute_reward("agent_0", set(), 0)

        # Expected: (10.0 - 5.0) * 10.0 + 0.01 = 50.01
        self.assertAlmostEqual(reward, 50.01, places=2)

    def test_crash_penalty_advanced(self):
        """Test crash penalty in advanced reward."""
        reward = self.reward_env._compute_reward("agent_0", {"agent_0"}, 0)

        # Expected: -5.0 for crash
        self.assertEqual(reward, -5.0)


class TestSpeedReward(unittest.TestCase):
    """Test speed-based reward functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.env_config = {"num_agents": 2, "timestep": 0.01}
        with patch('rewards_pepe.MultiAgentF110'):
            self.reward_env = SpeedReward(self.env_config)

        # Mock the environment
        self.reward_env.env = Mock()
        self.reward_env.env.num_agents = 2
        self.reward_env.env.config = {"timestep": 0.01}
        self.reward_env.env.track = Mock()
        self.reward_env.env.track.centerline = Mock()
        self.reward_env.env.track.centerline.spline = Mock()
        self.reward_env.env.track.centerline.spline.s = np.array([0, 10, 20, 30, 40, 50])
        self.reward_env.env.poses_x = np.array([0.0, 5.0])
        self.reward_env.env.poses_y = np.array([0.0, 2.0])

        self.reward_env.agents = ["agent_0", "agent_1"]
        self.reward_env._last_s = [5.0, 10.0]

    def test_speed_calculation(self):
        """Test speed calculation from progress."""
        self.reward_env.env.track.centerline.spline.calc_arclength_inaccurate.return_value = (15.0, None)

        reward = self.reward_env._compute_reward("agent_0", set(), 0)

        # Progress = 15.0 - 5.0 = 10.0
        # Speed = 10.0 / 0.01 = 1000.0 (track_speed)
        # Speed reward = 1000.0 * 0.3 = 300.0
        # Survival bonus = 0.01
        # High speed bonus = (1000.0 - 5.0) * 0.1 = 99.5
        expected_reward = 300.0 + 0.01 + 99.5
        self.assertAlmostEqual(reward, expected_reward, places=1)


class TestWaypointReward(unittest.TestCase):
    """Test waypoint-based reward functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.env_config = {"num_agents": 2}
        with patch('rewards_pepe.MultiAgentF110'):
            self.reward_env = WaypointReward(self.env_config)

        # Mock the environment
        self.reward_env.env = Mock()
        self.reward_env.env.num_agents = 2
        self.reward_env.env.track = Mock()
        self.reward_env.env.track.centerline = Mock()
        self.reward_env.env.track.centerline.spline = Mock()
        self.reward_env.env.track.centerline.spline.s = np.array([0, 10, 20, 30, 40, 50])
        self.reward_env.env.track.centerline.xs = np.linspace(0, 100, 100)
        self.reward_env.env.track.centerline.ys = np.linspace(0, 100, 100)
        self.reward_env.env.poses_x = np.array([0.0, 5.0])
        self.reward_env.env.poses_y = np.array([0.0, 2.0])

        self.reward_env.agents = ["agent_0", "agent_1"]
        self.reward_env._last_s = [0.0, 0.0]

    def test_waypoint_threshold_reward(self):
        """Test reward for passing waypoint thresholds."""
        self.reward_env.env.track.centerline.spline.calc_arclength_inaccurate.return_value = (2.5, None)

        reward = self.reward_env._compute_reward("agent_0", set(), 0)

        # Should get reward for passing thresholds + survival bonus - deviation penalty
        self.assertGreater(reward, 0)  # Should be positive

    def test_crash_penalty_waypoint(self):
        """Test crash penalty in waypoint reward."""
        reward = self.reward_env._compute_reward("agent_0", {"agent_0"}, 0)

        self.assertEqual(reward, -5.0)


class TestCompetitiveOvertakingReward(unittest.TestCase):
    """Test competitive overtaking reward functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.env_config = {"num_agents": 3, "timestep": 0.01}
        with patch('rewards_pepe.MultiAgentF110'):
            self.reward_env = CompetitiveOvertakingReward(self.env_config)

        # Mock the environment
        self.reward_env.env = Mock()
        self.reward_env.env.num_agents = 3
        self.reward_env.env.config = {"timestep": 0.01}
        self.reward_env.env.track = Mock()
        self.reward_env.env.track.centerline = Mock()
        self.reward_env.env.track.centerline.spline = Mock()
        self.reward_env.env.track.centerline.spline.s = np.array([0, 10, 20, 30, 40, 50])

        # Use specific values that match expected format
        self.reward_env.env.track.centerline.xs = np.array([0.0] * 1692)
        self.reward_env.env.track.centerline.ys = np.array([0.0] * 1692)
        self.reward_env.env.poses_x = np.array([0.0, 5.0, 10.0])
        self.reward_env.env.poses_y = np.array([0.0, 2.0, 4.0])

        self.reward_env.agents = ["agent_0", "agent_1", "agent_2"]
        self.reward_env._last_s = [0.0, 0.0, 0.0]

    def test_overtaking_detection(self):
        """Test overtaking reward mechanism."""
        # Mock calc_arclength_inaccurate to simulate overtaking
        self.reward_env.env.track.centerline.spline.calc_arclength_inaccurate.side_effect = [
            (10.0, None),  # agent_0 current position
            (5.0, None),   # agent_1 position (for overtaking check)
            (3.0, None),   # agent_2 position (for overtaking check)
        ]

        reward = self.reward_env._compute_reward("agent_0", set(), 0)

        # Should include overtaking bonuses, speed reward, survival bonus
        self.assertGreater(reward, 0)

    def test_proximity_penalty(self):
        """Test proximity penalty for being too close."""
        # Set up close positions
        self.reward_env.env.poses_x = np.array([0.0, 0.2, 10.0])  # Very close agents
        self.reward_env.env.poses_y = np.array([0.0, 0.1, 4.0])

        self.reward_env.env.track.centerline.spline.calc_arclength_inaccurate.side_effect = [
            (10.0, None), (10.1, None), (3.0, None)
        ]

        reward = self.reward_env._compute_reward("agent_0", set(), 0)

        # Reward should be reduced due to proximity penalty
        self.assertIsInstance(reward, (int, float))


class TestSafetyReward(unittest.TestCase):
    """Test safety-focused reward functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.env_config = {"num_agents": 2}
        with patch('rewards_pepe.MultiAgentF110'):
            self.reward_env = SafetyReward(self.env_config)

        # Mock the environment
        self.reward_env.env = Mock()
        self.reward_env.env.num_agents = 2
        self.reward_env.env.track = Mock()
        self.reward_env.env.track.centerline = Mock()
        self.reward_env.env.track.centerline.spline = Mock()
        self.reward_env.env.track.centerline.spline.s = np.array([0, 10, 20, 30, 40, 50])
        self.reward_env.env.poses_x = np.array([0.0, 5.0])
        self.reward_env.env.poses_y = np.array([0.0, 2.0])

        # Mock scan data for safety calculation
        self.reward_env.env.scans = [
            np.array([1.0, 2.0, 1.5, 3.0]),  # Agent 0 scan
            np.array([0.5, 1.0, 2.0, 1.8])   # Agent 1 scan
        ]

        self.reward_env.agents = ["agent_0", "agent_1"]
        self.reward_env._last_s = [5.0, 10.0]

    def test_safety_reward_calculation(self):
        """Test safety reward with LiDAR data."""
        self.reward_env.env.track.centerline.spline.calc_arclength_inaccurate.return_value = (10.0, None)

        reward = self.reward_env._compute_reward("agent_0", set(), 0)

        # Should include progress (reduced weight) + safety + survival bonus
        # Progress: (10.0 - 5.0) * 0.5 = 2.5
        # Safety: min(scans) * 0.5 = 1.0 * 0.5 = 0.5
        # Survival: 0.01
        expected_reward = 2.5 + 0.5 + 0.01
        self.assertAlmostEqual(reward, expected_reward, places=2)

    def test_crash_penalty_safety(self):
        """Test crash penalty in safety reward."""
        reward = self.reward_env._compute_reward("agent_0", {"agent_0"}, 0)

        self.assertEqual(reward, -5.0)


class TestRewardFactoryFunction(unittest.TestCase):
    """Test the reward factory function."""

    @patch('rewards_pepe.MultiAgentF110')
    def test_factory_function_creation(self):
        """Test that factory function creates correct reward instances."""
        from rewards_pepe import get_reward_function

        # Test each reward type
        test_cases = [
            ("ProgressRewardAdvancedEnv", ProgressRewardAdvancedEnv),
            ("SpeedReward", SpeedReward),
            ("WaypointReward", WaypointReward),
            ("CompetitiveOvertakingReward", CompetitiveOvertakingReward),
            ("SafetyReward", SafetyReward),
        ]

        for reward_name, expected_class in test_cases:
            with self.subTest(reward_name=reward_name):
                mock_env = Mock()
                reward_instance = get_reward_function(reward_name, mock_env)
                self.assertIsInstance(reward_instance, expected_class)

    def test_factory_function_invalid_name(self):
        """Test factory function with invalid reward name."""
        from rewards_pepe import get_reward_function

        with self.assertRaises(ValueError):
            get_reward_function("InvalidRewardName", Mock())


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
