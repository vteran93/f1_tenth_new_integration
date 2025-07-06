"""
Consolidated reward functions for F1TENTH multi-agent racing environment.
This file contains all optimized reward functions after redundancy elimination.

Author: Victor
Date: July 4, 2025
"""

import warnings
from examples.multiagent.lib.multiagent_env import MultiAgentF110
import numpy as np

# Define deprecated decorator if not available
try:
    from warnings import deprecated
except ImportError:
    def deprecated(func):
        """Fallback deprecated decorator for older Python versions."""
        def wrapper(*args, **kwargs):
            warnings.warn(f"{func.__name__} is deprecated", DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper


# ============================================================================
# BASE CLASSES
# ============================================================================

class BaseReward:
    """Base class for reward functions using polymorphism."""

    def _get_rewards(self, env, newly_crashed):
        """
        Calculate rewards for each agent.

        Args:
            env: MultiAgentF110 environment instance
            newly_crashed: Set of agents that crashed this step

        Returns:
            List of rewards for each agent
        """
        raise NotImplementedError("Subclasses must implement _get_rewards")


class RewardFunction(MultiAgentF110):
    """Abstract base class for reward functions."""

    def __init__(self, env_config=None):
        super().__init__(env_config=env_config)
        # Store the F1TENTH environment instance to access state information

    def _compute_reward(self, agent, newly_crashed, i):
        """Compute reward for a single agent.

        Args:
            agent (str): The ID of the agent (e.g., 'agent_0').
            newly_crashed (list): List of agents that crashed in this step.
            i (int): Index of the agent in the environment.

        Returns:
            float: The computed reward for the agent.
        """
        raise NotImplementedError("Subclasses must implement _compute_reward")

    def _get_rewards(self, newly_crashed) -> list:
        """Calculate individual rewards for each agent."""

        # Initialize last_s tracking if not exists
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents

        rewards = []
        for i in range(self.env.num_agents):
            agent = self.agents[i]
            reward = self._compute_reward(agent, newly_crashed, i)
            rewards.append(reward)

        return rewards


# ============================================================================
# CONSOLIDATED REWARD FUNCTIONS
# ============================================================================

class ProgressRewardEnv(MultiAgentF110):
    """Basic progress reward - original implementation that works."""

    def __init__(self, env_config=None):
        super().__init__(env_config)

    def _get_rewards(self, newly_crashed) -> list:
        """Calculate individual rewards for each agent based on F110Env reward function."""

        # Initialize last_s tracking if not exists
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents

        rewards = []
        for i in range(self.env.num_agents):
            agent = self.agents[i]
            reward = self._compute_reward(agent, newly_crashed, i)
            rewards.append(reward)

        return rewards

    def _compute_reward(self, agent, newly_crashed, i):
        if agent in self._crashed_agents and agent not in newly_crashed:
            # Agent was already crashed - no reward calculation needed
            reward = 0.0
        else:
            # Calculate track progress using centerline spline
            current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
                self.env.poses_x[i].item(), self.env.poses_y[i].item()
            )

            # Calculate progress since last step
            prog = current_s - self._last_s[i]

            # Handle lap completion (when current_s wraps around to beginning)
            if prog > 0.9 * self.env.track.centerline.spline.s[-1]:
                prog = (self.env.track.centerline.spline.s[-1] - self._last_s[i]) + current_s

            # Start with progress reward (main component from F110Env)
            reward = prog

            # Apply collision penalty (from F110Env)
            if agent in newly_crashed:  # Only penalize when agent crashes this step
                reward -= 1.0

            # Update last track position for this agent
            self._last_s[i] = current_s

        return reward


class ProgressRewardAdvancedEnv(RewardFunction):
    """Advanced progress reward with enhanced scaling and crash handling."""

    def __init__(self, env_config=None):
        super().__init__(env_config=env_config)
        # Initialize last_s tracking for each agent
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents
        # Store crashed agents to avoid repeated calculations
        self._crashed_agents = set()

    def _compute_reward(self, agent, newly_crashed, i):
        """Calculate reward with progress and survival incentives.

        Features:
        - Enhanced crash penalty (-5.0)
        - Scaled progress reward (×10.0)
        - Survival bonus (+0.01)
        - Robust lap completion handling
        """
        if agent in self._crashed_agents and agent not in newly_crashed:
            return 0.0

        # Track newly crashed agents
        if agent in newly_crashed:
            self._crashed_agents.add(agent)
            return -5.0  # Large penalty for crashing

        # Calculate track progress using centerline spline
        current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
            float(self.env.poses_x[i]), float(self.env.poses_y[i])
        )

        # Calculate progress since last step
        prog = current_s - self._last_s[i]
        track_length = self.env.track.centerline.spline.s[-1]

        # Adjust for track loop completion or backward movement
        if prog < -0.5 * track_length:
            prog += track_length  # Handle crossing the finish line
        elif prog > 0.5 * track_length:
            prog -= track_length  # Handle backward movement

        # Update last track position for this agent
        self._last_s[i] = current_s

        # Reward progress (scaled by 10 for significance) and add survival bonus
        progress_reward = prog * 10.0
        survival_reward = 0.01
        return progress_reward + survival_reward


class SpeedReward(RewardFunction):
    """Speed-focused reward encouraging higher speeds with crash penalties."""

    def __init__(self, env_config=None):
        super().__init__(env_config=env_config)
        # Initialize last_s tracking for each agent
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents
        # Track last positions for speed calculation
        self.last_positions = {f"agent_{i}": (0.0, 0.0) for i in range(self.env.num_agents)}
        # Environment timestep for speed calculation
        self.timestep = self.env.config.get("timestep", 0.01)
        # Store crashed agents to avoid repeated calculations
        self._crashed_agents = set()

    def _compute_reward(self, agent, newly_crashed, i):
        """Calculate reward based on speed computed from track progress and crash status.

        Speed calculation methods:
        1. From track progress (current_s - last_s) / timestep
        2. From position changes using Euclidean distance

        Uses the higher of the two speeds to encourage both forward motion
        and overall movement, preventing agents from getting stuck.
        """
        if agent in self._crashed_agents and agent not in newly_crashed:
            return 0.0

        # Track newly crashed agents
        if agent in newly_crashed:
            self._crashed_agents.add(agent)
            return -5.0  # Large penalty for crashing

        # Calculate track progress using centerline spline
        current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
            float(self.env.poses_x[i]), float(self.env.poses_y[i])
        )

        # Get current position with robust access methods
        if hasattr(self.env, 'sim') and hasattr(self.env.sim, 'agent_poses'):
            current_x = float(self.env.sim.agent_poses[i, 0])
            current_y = float(self.env.sim.agent_poses[i, 1])
        elif hasattr(self.env, 'poses_x') and hasattr(self.env, 'poses_y'):
            current_x = float(self.env.poses_x[i])
            current_y = float(self.env.poses_y[i])
        else:
            # Fallback: use last known position if available
            if agent in self.last_positions:
                current_x, current_y = self.last_positions[agent]
            else:
                current_x, current_y = 0.0, 0.0

        current_pos = (current_x, current_y)

        # Calculate speed from track progress (primary method)
        progress = current_s - self._last_s[i]
        track_length = self.env.track.centerline.spline.s[-1]

        # Handle track loop completion
        if progress < -0.5 * track_length:
            progress += track_length  # Crossed finish line
        elif progress > 0.5 * track_length:
            progress -= track_length  # Going backwards

        # Speed from track progress (m/s)
        track_speed = max(0.0, progress / self.timestep)

        # Calculate speed from position changes (secondary method for robustness)
        if agent in self.last_positions:
            last_x, last_y = self.last_positions[agent]
            distance_moved = np.sqrt((current_x - last_x)**2 + (current_y - last_y)**2)
            position_speed = distance_moved / self.timestep
        else:
            position_speed = 0.0

        # Use the higher of the two speeds
        speed = max(track_speed, position_speed)

        # Update last position and last_s for next iteration
        self.last_positions[agent] = current_pos
        self._last_s[i] = current_s

        # Reward calculation
        speed_reward = speed * 0.3  # Scaled for F1TENTH speeds
        survival_reward = 0.01      # Small survival bonus

        # Bonus for very high speeds (aggressive driving)
        speed_bonus = (speed - 5.0) * 0.1 if speed > 5.0 else 0.0

        total_reward = speed_reward + survival_reward + speed_bonus
        return total_reward


class WaypointReward(RewardFunction):
    """Waypoint-based reward encouraging structured track progression."""

    def __init__(self, env_config=None):
        super().__init__(env_config=env_config)
        # Initialize last_s tracking for each agent
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents

        # Compute arc lengths for waypoints from xs, ys
        xs, ys = self.env.track.centerline.xs, self.env.track.centerline.ys

        # Calculate distances between consecutive waypoints
        distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        # Compute cumulative arc lengths
        self.waypoint_s = np.concatenate(([0.0], np.cumsum(distances)))
        self.num_waypoints = len(self.waypoint_s)

        # Track the last arc length threshold passed by each agent
        self.last_s_threshold = {
            agent: 0.0 for agent in [f"agent_{i}" for i in range(self.env.num_agents)]
        }
        # Define arc length threshold for "waypoints" (every 1 meter)
        self.threshold_distance = 1.0  # Meters
        # Store crashed agents to avoid repeated calculations
        self._crashed_agents = set()

    def _compute_reward(self, agent, newly_crashed, i):
        """Calculate reward based on passing arc length thresholds and crash status.

        Features:
        - Reward for passing distance thresholds (+1.0 per threshold)
        - Lap completion bonus (+10.0)
        - Centerline deviation penalty (-deviation * 0.1)
        - Survival bonus (+0.01)
        """
        if agent in self._crashed_agents and agent not in newly_crashed:
            return 0.0

        # Track newly crashed agents
        if agent in newly_crashed:
            self._crashed_agents.add(agent)
            return -5.0  # Large penalty for crashing

        # Calculate track progress using centerline spline
        current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
            float(self.env.poses_x[i]), float(self.env.poses_y[i])
        )

        # Initialize reward
        reward = 0.0
        track_length = self.env.track.centerline.spline.s[-1]

        # Handle track loop completion
        if current_s < self._last_s[i] and (self._last_s[i] - current_s) > 0.5 * track_length:
            reward += 10.0  # Bonus for completing a lap
            self.last_s_threshold[agent] = 0.0  # Reset threshold for new lap

        # Calculate number of "waypoints" (thresholds) passed
        last_threshold = self.last_s_threshold[agent]
        current_threshold = np.floor(current_s / self.threshold_distance) * self.threshold_distance
        thresholds_passed = int((current_threshold - last_threshold) / self.threshold_distance)

        if thresholds_passed < 0 and current_s < self._last_s[i]:
            # Handle loop completion
            thresholds_passed = int((track_length - last_threshold + current_s) / self.threshold_distance)

        # Reward for passing thresholds
        reward += thresholds_passed * 1.0

        # Update last threshold
        self.last_s_threshold[agent] = current_threshold

        # Penalize deviation from the centerline
        x, y = float(self.env.poses_x[i]), float(self.env.poses_y[i])
        # Find the closest waypoint's (x, y) coordinates
        closest_idx = np.argmin(np.abs(self.waypoint_s - current_s))
        closest_x = self.env.track.centerline.xs[closest_idx]
        closest_y = self.env.track.centerline.ys[closest_idx]
        # Calculate Euclidean distance to the closest waypoint
        deviation = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
        # Apply a small penalty for deviation
        deviation_penalty = -deviation * 0.1
        reward += deviation_penalty

        # Add a small survival bonus
        reward += 0.01

        # Update last track position for this agent
        self._last_s[i] = current_s

        return reward


class CompetitiveOvertakingReward(RewardFunction):
    """Competitive racing reward with lap completion, overtaking, and safety."""

    def __init__(self, env_config=None):
        super().__init__(env_config=env_config)
        # Initialize last_s tracking for each agent
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents

        # Compute arc lengths for waypoints from xs, ys
        xs, ys = self.env.track.centerline.xs, self.env.track.centerline.ys

        # Verify xs and ys dimensions (expecting 1692 for specific track)
        if len(xs) != 1692 or len(ys) != 1692:
            warnings.warn(f"Expected 1692 waypoints, got {len(xs)} for xs and {len(ys)} for ys")

        # Calculate distances between consecutive waypoints
        distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        # Compute cumulative arc lengths
        self.waypoint_s = np.concatenate(([0.0], np.cumsum(distances)))
        self.num_waypoints = len(self.waypoint_s)

        # Track the last arc length threshold and arc length for each agent
        self.last_s_threshold = {
            agent: 0.0 for agent in [f"agent_{i}" for i in range(self.env.num_agents)]
        }
        # Track relative positions to detect overtaking
        self.was_behind = {
            agent: {
                other: False for other in [f"agent_{i}" for i in range(self.env.num_agents)]
            } for agent in [f"agent_{i}" for i in range(self.env.num_agents)]
        }
        # Timestep for speed calculation
        self.timestep = self.env.config.get("timestep", 0.01)
        self.threshold_distance = 1.0  # Meters for "waypoints"
        # Store crashed agents to avoid repeated calculations
        self._crashed_agents = set()

    def _compute_reward(self, agent, newly_crashed, i):
        """Calculate reward based on lap completion, overtaking, speed, and safety.

        Features:
        - Passing arc length thresholds (+1.0 per threshold)
        - Completing a lap (+10.0)
        - Overtaking another agent (+5.0 per overtake)
        - Speed reward (progress per timestep × 0.5)
        - Survival bonus (+0.01)
        - Proximity penalty (-0.1 × distance if < 0.5m)
        - Crash penalty (-5.0)
        """
        if agent in self._crashed_agents and agent not in newly_crashed:
            return 0.0

        # Track newly crashed agents
        if agent in newly_crashed:
            self._crashed_agents.add(agent)
            return -5.0  # Large penalty for crashing

        # Calculate track progress using centerline spline
        current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
            float(self.env.poses_x[i]), float(self.env.poses_y[i])
        )

        # Initialize reward
        reward = 0.0
        track_length = self.env.track.centerline.spline.s[-1]

        # --- Threshold Progress ---
        if current_s < self._last_s[i] and (self._last_s[i] - current_s) > 0.5 * track_length:
            reward += 10.0  # Bonus for completing a lap
            self.last_s_threshold[agent] = 0.0  # Reset threshold

        last_threshold = self.last_s_threshold[agent]
        current_threshold = np.floor(current_s / self.threshold_distance) * self.threshold_distance
        thresholds_passed = int((current_threshold - last_threshold) / self.threshold_distance)

        if thresholds_passed < 0 and current_s < self._last_s[i]:
            thresholds_passed = int((track_length - last_threshold + current_s) / self.threshold_distance)

        reward += thresholds_passed * 1.0  # Reward for passing thresholds
        self.last_s_threshold[agent] = current_threshold

        # --- Overtaking Reward ---
        agent_x, agent_y = float(self.env.poses_x[i]), float(self.env.poses_y[i])

        for other_idx, other_agent in enumerate([f"agent_{j}" for j in range(self.env.num_agents)]):
            if other_agent == agent:
                continue

            other_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
                float(self.env.poses_x[other_idx]), float(self.env.poses_y[other_idx])
            )

            # Check for overtaking
            if self.was_behind[agent][other_agent] and current_s > other_s:
                reward += 5.0  # Reward for overtaking
                self.was_behind[agent][other_agent] = False

            # Update relative positions
            if current_s <= other_s:
                self.was_behind[agent][other_agent] = True
            else:
                self.was_behind[other_agent][agent] = True

        # --- Proximity Penalty ---
        for other_idx, other_agent in enumerate([f"agent_{j}" for j in range(self.env.num_agents)]):
            if other_agent == agent:
                continue

            other_x, other_y = float(self.env.poses_x[other_idx]), float(self.env.poses_y[other_idx])
            distance = np.sqrt((agent_x - other_x)**2 + (agent_y - other_y)**2)

            if distance < 0.5:
                reward += -0.1 * distance  # Penalty for being too close

        # --- Speed Reward ---
        progress = current_s - self._last_s[i]
        if progress < -0.5 * track_length:
            progress += track_length
        elif progress > 0.5 * track_length:
            progress -= track_length

        speed = progress / self.timestep
        reward += speed * 0.5  # Reward proportional to speed

        # --- Survival Bonus ---
        reward += 0.01

        # Update last arc length
        self._last_s[i] = current_s

        return reward


class SafetyReward(RewardFunction):
    """Safety-focused reward encouraging careful driving with LiDAR awareness."""

    def __init__(self, env_config=None):
        super().__init__(env_config=env_config)
        # Initialize last_s tracking for each agent
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents
        # Store crashed agents to avoid repeated calculations
        self._crashed_agents = set()

    def _compute_reward(self, agent, newly_crashed, i):
        """Compute safety-focused reward for individual agent.

        Features:
        - Progress reward with reduced weight (×0.5)
        - Safety reward based on minimum LiDAR distance (×0.5)
        - Survival bonus (+0.01)
        - Crash penalty (-5.0)
        """
        if agent in self._crashed_agents and agent not in newly_crashed:
            return 0.0

        # Track newly crashed agents
        if agent in newly_crashed:
            self._crashed_agents.add(agent)
            return -5.0  # Large penalty for crashing

        # Calculate track progress using centerline spline
        current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
            float(self.env.poses_x[i]), float(self.env.poses_y[i])
        )

        # Calculate progress since last step
        prog = current_s - self._last_s[i]
        track_length = self.env.track.centerline.spline.s[-1]

        # Handle lap completion (when current_s wraps around to beginning)
        if prog < -0.5 * track_length:
            prog += track_length  # Handle crossing the finish line
        elif prog > 0.5 * track_length:
            prog -= track_length  # Handle backward movement

        # Safety component (minimum distance to walls from LiDAR)
        if hasattr(self.env, 'scans') and len(self.env.scans) > i:
            min_scan_distance = np.min(self.env.scans[i])
            safety_reward = min_scan_distance * 0.5  # Reward staying away from walls
        else:
            # Fallback if scans not available
            safety_reward = 0.0

        # Combined reward (reduced weight for progress, emphasis on safety)
        progress_reward = prog * 0.5
        base_reward = progress_reward + safety_reward

        # Add small survival bonus
        survival_bonus = 0.01
        total_reward = base_reward + survival_bonus

        # Update last track position for this agent
        self._last_s[i] = current_s

        return total_reward


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

@deprecated
def get_reward_function(reward_name, env):
    """Factory function to return the appropriate reward function instance.

    Args:
        reward_name (str): Name of the reward function.
        env: The F1TENTH environment instance.

    Returns:
        RewardFunction: An instance of the specified reward function class.

    Raises:
        ValueError: If the reward_name is not recognized.
    """
    reward_classes = {
        "ProgressRewardEnv": ProgressRewardEnv,
        "ProgressRewardAdvancedEnv": ProgressRewardAdvancedEnv,
        "SpeedReward": SpeedReward,
        "WaypointReward": WaypointReward,
        "CompetitiveOvertakingReward": CompetitiveOvertakingReward,
        "SafetyReward": SafetyReward
    }

    if reward_name not in reward_classes:
        available_rewards = list(reward_classes.keys())
        raise ValueError(f"Unknown reward function: {reward_name}. Available: {available_rewards}")

    return reward_classes[reward_name](env)
