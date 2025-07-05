import warnings
from .multiagent_env import MultiAgentF110
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

# Abstract base class for defining reward functions in a multi-agent F1TENTH environment.
# This class ensures all reward functions follow a consistent interface for computing
# rewards for individual agents, making it easy to add new reward strategies.


class RewardFunction(MultiAgentF110):
    """Abstract base class for reward functions."""

    def __init__(self, env_config=None):
        super().__init__(env_config=env_config)
        # Store the F1TENTH environment instance to access state information
        # (e.g., positions, speeds, track data) needed for reward computation.

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
        """Calculate individual rewards for each agent based on the original F110Env reward function."""

        # Initialize last_s tracking if not exists (track progress for each agent)
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents

        rewards = []
        for i in range(self.env.num_agents):
            agent = self.agents[i]

            reward = self._compute_reward(agent, newly_crashed, i)

            rewards.append(reward)

        return rewards

# Reward function that encourages track progress and survival.
# This is the original 'geminiReward' from the F1TENTH multi-agent setup,
# rewarding agents for moving forward along the track while penalizing crashes.


class ProgressRewardAdvancedEnv(RewardFunction):
    """Reward function based on track progress and survival (original geminiReward)."""

    def __init__(self, env_config=None):
        super().__init__(env_config=env_config)
        # Initialize last_s tracking for each agent
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents
        # Store crashed agents to avoid repeated calculations
        self._crashed_agents = set()

    def _compute_reward(self, agent, newly_crashed, i):
        """Calculate reward with progress and survival incentives.

        If the agent crashes, a large negative reward is given.
        Otherwise, the reward is based on the agent's progress along the track
        (difference in arc length) plus a small survival bonus for staying active.
        Progress is adjusted to handle track loop completion or backward movement.

        Args:
            agent (str): The ID of the agent (e.g., 'agent_0').
            newly_crashed (list): List of agents that crashed in this step.
            i (int): Index of the agent in the environment.

        Returns:
            float: The computed reward, combining progress and survival terms.
        """
        if agent in self._crashed_agents and agent not in newly_crashed:
            # Agent was already crashed - no reward calculation needed
            return 0.0

        # Track newly crashed agents
        if agent in newly_crashed:
            self._crashed_agents.add(agent)
            return -5.0  # Large penalty for crashing to discourage collisions.

        # Calculate track progress using centerline spline
        current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
            self.env.poses_x[i], self.env.poses_y[i]
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

        # Reward progress (scaled by 10 for significance) and add a small survival bonus
        progress_reward = prog * 10.0
        survival_reward = 0.01
        return progress_reward + survival_reward

# Reward function that encourages higher speeds while penalizing crashes.
# This implementation correctly calculates speed from position changes and
# progress along the track, providing meaningful rewards for aggressive driving.


class SpeedReward(RewardFunction):
    """Reward function focused on encouraging higher speeds with crash penalties."""

    def __init__(self, env_config=None):
        """Initialize the SpeedReward with environment reference and tracking variables."""
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

        Speed is calculated in two ways for robustness:
        1. From track progress (current_s - last_s) / timestep
        2. From position changes using Euclidean distance

        The higher of the two speeds is used to encourage both forward motion
        and overall movement, preventing the agent from getting stuck.

        Args:
            agent (str): The ID of the agent (e.g., 'agent_0').
            newly_crashed (list): List of agents that crashed in this step.
            i (int): Index of the agent in the environment.

        Returns:
            float: The computed reward, based on speed and survival.
        """
        if agent in self._crashed_agents and agent not in newly_crashed:
            # Agent was already crashed - no reward calculation needed
            return 0.0

        # Track newly crashed agents
        if agent in newly_crashed:
            self._crashed_agents.add(agent)
            return -5.0  # Large penalty for crashing to discourage collisions.

        # Calculate track progress using centerline spline
        current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
            self.env.poses_x[i], self.env.poses_y[i]
        )

        # Get current position - access through sim.agent_poses for compatibility
        if hasattr(self.env, 'sim') and hasattr(self.env.sim, 'agent_poses'):
            current_x = self.env.sim.agent_poses[i, 0]
            current_y = self.env.sim.agent_poses[i, 1]
        elif hasattr(self.env, 'poses_x') and hasattr(self.env, 'poses_y'):
            current_x = self.env.poses_x[i]
            current_y = self.env.poses_y[i]
        else:
            # Fallback: use last known position if available, otherwise use (0, 0)
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
            progress -= track_length  # Going backwards (shouldn't happen normally)

        # Speed from track progress (m/s)
        track_speed = max(0.0, progress / self.timestep)  # Only positive speeds

        # Calculate speed from position changes (secondary method for robustness)
        if agent in self.last_positions:
            last_x, last_y = self.last_positions[agent]
            distance_moved = np.sqrt((current_x - last_x)**2 + (current_y - last_y)**2)
            position_speed = distance_moved / self.timestep
        else:
            position_speed = 0.0

        # Use the higher of the two speeds (encourages both progress and movement)
        speed = max(track_speed, position_speed)

        # Update last position and last_s for next iteration
        self.last_positions[agent] = current_pos
        self._last_s[i] = current_s

        # Reward calculation
        # Speed reward: scaled for typical F1TENTH speeds (up to ~10-15 m/s)
        # Using a moderate scaling factor to make speed rewards meaningful but not overwhelming
        speed_reward = speed * 0.3  # Reduced from 0.5 to prevent excessive rewards

        # Small survival bonus to encourage staying active
        survival_reward = 0.01

        # Bonus for very high speeds (encourage aggressive driving)
        if speed > 5.0:  # Above 5 m/s is considered fast
            speed_bonus = (speed - 5.0) * 0.1
        else:
            speed_bonus = 0.0

        total_reward = speed_reward + survival_reward + speed_bonus

        return total_reward

# Reward function that encourages passing waypoints on the track.
# This rewards agents for progressing through arc length thresholds (proxy for waypoints),
# promoting lap completion in a competitive racing context, while penalizing crashes
# and deviations from the centerline.


class WaypointReward(RewardFunction):
    """Reward function that rewards agents for passing arc length thresholds."""

    def __init__(self, env_config=None):
        """Initialize the reward function with environment and waypoint data.

        Args:
            env_config: The environment configuration dictionary.
        """
        super().__init__(env_config=env_config)
        # Initialize last_s tracking for each agent
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents
        # Compute arc lengths for waypoints from xs, ys since centerline.s is unavailable
        xs, ys = self.env.track.centerline.xs, self.env.track.centerline.ys
        # Verify xs and ys have 1692 elements as per environment
        """
        if len(xs) != 1692 or len(ys) != 1692:
            raise ValueError(f"Expected 1692 waypoints, got {len(xs)} for xs and {len(ys)} for ys")
        """

        # Calculate distances between consecutive waypoints
        distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        # Compute cumulative arc lengths
        self.waypoint_s = np.concatenate(([0.0], np.cumsum(distances)))
        self.num_waypoints = len(self.waypoint_s)
        # Track the last arc length threshold passed by each agent
        self.last_s_threshold = {agent: 0.0 for agent in [f"agent_{i}" for i in range(self.env.num_agents)]}
        # Define arc length threshold for "waypoints" (every 1 meter)
        self.threshold_distance = 1.0  # Meters
        # Store crashed agents to avoid repeated calculations
        self._crashed_agents = set()

    def _compute_reward(self, agent, newly_crashed, i):
        """Calculate reward based on passing arc length thresholds and crash status.

        The agent receives a reward for passing arc length thresholds (e.g., every 1m),
        a bonus for completing a lap, and a penalty for crashing. A penalty is applied
        for deviating from the centerline to encourage precise navigation.

        Args:
            agent (str): The ID of the agent (e.g., 'agent_0').
            newly_crashed (list): List of agents that crashed in this step.
            i (int): Index of the agent in the environment.

        Returns:
            float: The computed reward, based on progress and crash status.
        """
        if agent in self._crashed_agents and agent not in newly_crashed:
            # Agent was already crashed - no reward calculation needed
            return 0.0

        # Track newly crashed agents
        if agent in newly_crashed:
            self._crashed_agents.add(agent)
            return -5.0  # Large penalty for crashing to discourage collisions

        # Calculate track progress using centerline spline
        current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
            self.env.poses_x[i], self.env.poses_y[i]
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

        # Reward for passing thresholds (1.0 per threshold)
        reward += thresholds_passed * 1.0

        # Update last threshold
        self.last_s_threshold[agent] = current_threshold

        # Penalize deviation from the centerline
        x, y = self.env.poses_x[i], self.env.poses_y[i]
        # Find the closest waypoint's (x, y) coordinates
        closest_idx = np.argmin(np.abs(self.waypoint_s - current_s))
        closest_x, closest_y = self.env.track.centerline.xs[closest_idx], self.env.track.centerline.ys[closest_idx]
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

# Reward function that encourages lap completion, overtaking, and safe competitive driving.
# This rewards agents for passing arc length thresholds, completing laps, overtaking
# opponents, maintaining speed, and avoiding collisions or risky proximity.


class CompetitiveOvertakingReward(RewardFunction):
    """Reward function for competitive racing with lap completion and overtaking."""

    def __init__(self, env_config=None):
        """Initialize the reward function with environment and tracking data.

        Args:
            env_config: The environment configuration dictionary.
        """
        super().__init__(env_config=env_config)
        # Initialize last_s tracking for each agent
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents
        # Compute arc lengths for waypoints from xs, ys since centerline.s is unavailable
        xs, ys = self.env.track.centerline.xs, self.env.track.centerline.ys
        # Verify xs and ys have 1692 elements as per environment
        if len(xs) != 1692 or len(ys) != 1692:
            raise ValueError(f"Expected 1692 waypoints, got {len(xs)} for xs and {len(ys)} for ys")
        # Calculate distances between consecutive waypoints
        distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        # Compute cumulative arc lengths
        self.waypoint_s = np.concatenate(([0.0], np.cumsum(distances)))
        self.num_waypoints = len(self.waypoint_s)
        # Track the last arc length threshold and arc length for each agent
        self.last_s_threshold = {agent: 0.0 for agent in [f"agent_{i}" for i in range(self.env.num_agents)]}
        # Track relative positions to detect overtaking
        self.was_behind = {agent: {other: False for other in [f"agent_{i}" for i in range(self.env.num_agents)]}
                           for agent in [f"agent_{i}" for i in range(self.env.num_agents)]}
        # Timestep for speed calculation
        self.timestep = self.env.config.get("timestep", 0.01)
        self.threshold_distance = 1.0  # Meters for "waypoints"
        # Store crashed agents to avoid repeated calculations
        self._crashed_agents = set()

    def _compute_reward(self, agent, newly_crashed, i):
        """Calculate reward based on lap completion, overtaking, speed, and safety.

        The agent receives rewards for:
        - Passing arc length thresholds (+1.0 per threshold).
        - Completing a lap (+10.0).
        - Overtaking another agent (+5.0 per overtake).
        - Maintaining speed (computed as arc length progress per timestep).
        - A small survival bonus (+0.01).
        And penalties for:
        - Crashing (-5.0).
        - Being too close to another agent (-0.1 * distance if < 0.5m).

        Args:
            agent (str): The ID of the agent (e.g., 'agent_0').
            newly_crashed (list): List of agents that crashed in this step.
            i (int): Index of the agent in the environment.

        Returns:
            float: The computed reward, combining all components.
        """
        if agent in self._crashed_agents and agent not in newly_crashed:
            # Agent was already crashed - no reward calculation needed
            return 0.0

        # Track newly crashed agents
        if agent in newly_crashed:
            self._crashed_agents.add(agent)
            return -5.0  # Large penalty for crashing

        # Calculate track progress using centerline spline
        current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
            self.env.poses_x[i], self.env.poses_y[i]
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
        agent_x, agent_y = self.env.poses_x[i], self.env.poses_y[i]
        for other_idx, other_agent in enumerate([f"agent_{j}" for j in range(self.env.num_agents)]):
            if other_agent == agent:
                continue
            other_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
                self.env.poses_x[other_idx], self.env.poses_y[other_idx]
            )
            if self.was_behind[agent][other_agent] and current_s > other_s:
                reward += 5.0  # Reward for overtaking
                self.was_behind[agent][other_agent] = False
            if current_s <= other_s:
                self.was_behind[agent][other_agent] = True
            else:
                self.was_behind[other_agent][agent] = True

        # --- Proximity Penalty ---
        for other_idx, other_agent in enumerate([f"agent_{j}" for j in range(self.env.num_agents)]):
            if other_agent == agent:
                continue
            other_x, other_y = self.env.poses_x[other_idx], self.env.poses_y[other_idx]
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


# Reward function that encourages careful driving by emphasizing safety.
# This reward function is designed to promote behaviors that avoid crashes
# and encourage smooth, controlled navigation of the track.


class SafetyReward(RewardFunction):
    """Safety-focused reward encouraging careful driving."""

    def __init__(self, env_config=None):
        super().__init__(env_config=env_config)
        # Initialize last_s tracking for each agent
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents
        # Store crashed agents to avoid repeated calculations
        self._crashed_agents = set()

    def _compute_reward(self, agent, newly_crashed, i):
        """Compute safety-focused reward for individual agent."""
        if agent in self._crashed_agents and agent not in newly_crashed:
            # Agent was already crashed - no reward calculation needed
            return 0.0

        # Track newly crashed agents
        if agent in newly_crashed:
            self._crashed_agents.add(agent)
            return -5.0  # Large penalty for crashing

        # Calculate track progress using centerline spline
        current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
            self.env.poses_x[i], self.env.poses_y[i]
        )

        # Calculate progress since last step
        prog = current_s - self._last_s[i]
        track_length = self.env.track.centerline.spline.s[-1]

        # Handle lap completion (when current_s wraps around to beginning)
        if prog < -0.5 * track_length:
            prog += track_length  # Handle crossing the finish line
        elif prog > 0.5 * track_length:
            prog -= track_length  # Handle backward movement

        # Safety component (minimum distance to walls)
        min_scan_distance = np.min(self.env.scans[i])
        safety_reward = min_scan_distance * 0.5  # Reward staying away from walls

        # Combined reward (reduced weight for progress, emphasis on safety)
        progress_reward = prog * 0.5
        base_reward = progress_reward + safety_reward

        # Add small survival bonus
        survival_bonus = 0.01
        total_reward = base_reward + survival_bonus

        # Update last track position for this agent
        self._last_s[i] = current_s

        return total_reward

# Factory function to create instances of reward functions based on configuration.
# This allows the main script to dynamically select reward functions for each agent
# as specified in config.yaml, making the system flexible and extensible.


@deprecated
def get_reward_function(reward_name, env):
    """Factory function to return the appropriate reward function instance.

    Args:
        reward_name (str): Name of the reward function (e.g., 'GeminiReward', 'SpeedReward', 'WaypointReward', 'CompetitiveOvertakingReward').
        env: The F1TENTH environment instance, passed to the reward function.

    Returns:
        RewardFunction: An instance of the specified reward function class.

    Raises:
        ValueError: If the reward_name is not recognized.
    """
    reward_classes = {
        "ProgressRewardAdvancedEnv": ProgressRewardAdvancedEnv,
        "SpeedReward": SpeedReward,
        "WaypointReward": WaypointReward,
        "CompetitiveOvertakingReward": CompetitiveOvertakingReward,
        "SafetyReward": SafetyReward
    }
    if reward_name not in reward_classes:
        raise ValueError(f"Unknown reward function: {reward_name}")
    return reward_classes[reward_name](env)
