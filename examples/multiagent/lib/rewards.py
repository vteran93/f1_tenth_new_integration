from .multiagent_env import MultiAgentF110
import numpy as np


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


# COPILOT NO TOCAR Esta funciona
class ProgressRewardEnv(MultiAgentF110):
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

    def _compute_reward(self, agent, newly_crashed, i):

        if agent in self._crashed_agents and agent not in newly_crashed:
            # Agent was already crashed - no reward calculation needed
            reward = 0.0
        else:
            # Calculate track progress using centerline spline (from F110Env)
            current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
                self.env.poses_x[i], self.env.poses_y[i]
            )
            # TODO Argument of type "ndarray[Any, dtype[float64]] | Unknown" cannot be assigned to parameter "x" of type "float" in function "calc_arclength_inaccurate"
#   Type "ndarray[Any, dtype[float64]] | Unknown" is not assignable to type "float"
#     "ndarray[Any, dtype[float64]]" is not assignable to "float"

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


class SpeedRewardEnv(MultiAgentF110):
    def __init__(self, env, timestep=0.1):
        self.env = env
        super().__init__(env)
        # Track last positions for speed calculation
        self.last_positions = {f"agent_{i}": (0.0, 0.0) for i in range(self.env.num_agents)}
        # Environment timestep for speed calculation
        self.timestep = self.env.config.get("timestep", 0.01)

    def _get_rewards(self, newly_crashed) -> list:
        """Calculate individual rewards for each agent based on the original F110Env reward function."""

        # Initialize last_s tracking if not exists (track progress for each agent)
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents

        rewards = []
        reward = 0.0
        for i in range(self.env.num_agents):
            agent = self.agents[i]
            if agent in self._crashed_agents and agent not in newly_crashed:
                reward = 0.0
            else:
                current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
                    self.env.poses_x[i], self.env.poses_y[i]
                )

                reward = self._compute_reward(agent,
                                              current_s,
                                              self.env.last_s[i],
                                              (agent in newly_crashed),
                                              self.env.track.centerline.spline.s[-1]
                                              )

            rewards.append(reward)

        return rewards

    def _compute_reward(self, agent, current_s, last_s, is_crashed, track_length):
        """Calculate reward based on speed computed from track progress and crash status.

        Speed is calculated in two ways for robustness:
        1. From track progress (current_s - last_s) / timestep
        2. From position changes using Euclidean distance

        The higher of the two speeds is used to encourage both forward motion
        and overall movement, preventing the agent from getting stuck.

        Args:
            agent (str): The ID of the agent (e.g., 'agent_0').
            current_s (float): Current arc length along the track centerline.
            last_s (float): Previous arc length along the track centerline.
            is_crashed (bool): Whether the agent crashed in this step.
            track_length (float): Total length of the track.

        Returns:
            float: The computed reward, based on speed and survival.
        """
        if is_crashed:
            return -5.0  # Large penalty for crashing to discourage collisions.

        # Extract the agent index from the agent ID (e.g., 'agent_0' -> 0)
        agent_idx = int(agent.split("_")[1])

        # Get current position
        current_x = self.env.poses_x[agent_idx]
        current_y = self.env.poses_y[agent_idx]
        current_pos = (current_x, current_y)

        # Calculate speed from track progress (primary method)
        progress = current_s - last_s
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

        # Update last position for next iteration
        self.last_positions[agent] = current_pos

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


class SACBasicReward(BaseReward):
    """SAC basic progress reward (from MultiAgentF110 base class in multiagent_sac.py)."""

    def _get_rewards(self, env, newly_crashed):
        """SAC basic reward function - similar to PPO but for SAC algorithm."""
        rewards = []
        for i in range(env.env.num_agents):
            agent = env.agents[i]

            if agent in env._crashed_agents and agent not in newly_crashed:
                reward = 0.0
            else:
                # Calculate track progress using centerline spline (from F110Env)
                current_s, _ = env.env.track.centerline.spline.calc_arclength_inaccurate(
                    env.env.poses_x[i], env.env.poses_y[i]
                )

                # Calculate progress since last step
                prog = current_s - env._last_s[i]

                # Handle lap completion (when current_s wraps around to beginning)
                if prog > 0.9 * env.env.track.centerline.spline.s[-1]:
                    prog = (env.env.track.centerline.spline.s[-1] - env._last_s[i]) + current_s

                # Start with progress reward (main component from F110Env)
                reward = prog

                # Apply collision penalty (from F110Env)
                if agent in newly_crashed:  # Only penalize when agent crashes this step
                    reward -= 1.0

                # Update last track position for this agent
                env._last_s[i] = current_s

            rewards.append(reward)

        return rewards


class SACGeminiReward(BaseReward):
    """SAC Gemini reward function (from geminiReward class in multiagent_sac.py)."""

    def _get_rewards(self, env, newly_crashed):
        """
        Improved SAC reward combining track progress with survival incentives.
        Features:
        - Scaled progress reward for stronger learning signal
        - Survival reward for non-crashed steps  
        - Enhanced collision penalty
        - Corrected lap completion logic
        """
        # Initialize last_s tracking if not exists (track progress for each agent)
        if not hasattr(env, '_last_s'):
            env._last_s = [0.0] * env.env.num_agents

        rewards = []
        for i in range(env.env.num_agents):
            agent = env.agents[i]
            reward = self._compute_reward(env, agent, newly_crashed, i)
            rewards.append(reward)

        return rewards

    def _compute_reward(self, env, agent, newly_crashed, i):
        """Compute SAC Gemini reward for individual agent."""
        track_length = env.env.track.centerline.spline.s[-1]

        if agent in env._crashed_agents and agent not in newly_crashed:
            # Agent was already crashed - no reward
            reward = 0.0
        else:
            # Calculate track progress using centerline spline
            current_s, _ = env.env.track.centerline.spline.calc_arclength_inaccurate(
                env.env.poses_x[i], env.env.poses_y[i]
            )

            # Calculate progress since last step
            prog = current_s - env._last_s[i]

            # Correctly handle lap completion (wrap-around)
            if prog < -0.5 * track_length:
                # Crossed finish line going forward
                prog += track_length
            elif prog > 0.5 * track_length:
                # Crossed finish line going backward (unlikely but possible)
                prog -= track_length

            # Apply rewards based on state
            if agent in newly_crashed:
                # Strong penalty for collision
                reward = -5.0
            else:
                # 1. Scaled progress reward (main incentive)
                progress_reward = prog * 10.0

                # 2. Small survival reward for each step not crashed
                survival_reward = 0.01

                reward = progress_reward + survival_reward

            # Update last track position for this agent
            env._last_s[i] = current_s

        return reward


class SpeedReward(BaseReward):
    """Speed-based reward for encouraging faster lap times."""

    def _get_rewards(self, env, newly_crashed):
        """Reward based on progress and speed combination."""
        rewards = []
        for i in range(env.env.num_agents):
            agent = env.agents[i]

            if agent in env._crashed_agents and agent not in newly_crashed:
                reward = 0.0
            else:
                # Progress component
                current_s, _ = env.env.track.raceline.spline.calc_arclength_inaccurate(
                    env.env.poses_x[i], env.env.poses_y[i]
                )
                prog = current_s - env._last_s[i]

                if prog > 0.9 * env.env.track.raceline.spline.s[-1]:
                    prog = (env.env.track.raceline.spline.s[-1] - env._last_s[i]) + current_s

                # Speed component
                speed = np.sqrt(env.env.linear_vels_x[i]**2 + env.env.linear_vels_y[i]**2)
                speed_reward = speed * 0.1  # Scale speed reward

                # Combined reward
                reward = prog + speed_reward

                # Collision penalty
                if agent in newly_crashed:
                    reward -= 2.0

                env._last_s[i] = current_s

            rewards.append(reward)

        return rewards


class SafetyReward(BaseReward):
    """Safety-focused reward encouraging careful driving."""

    def _get_rewards(self, env, newly_crashed):
        """Reward emphasizing safety with distance to walls."""
        # Initialize last_s tracking if not exists (track progress for each agent)
        if not hasattr(env, '_last_s'):
            env._last_s = [0.0] * env.env.num_agents

        rewards = []
        for i in range(env.env.num_agents):
            agent = env.agents[i]
            reward = self._compute_reward(env, agent, newly_crashed, i)
            rewards.append(reward)

        return rewards

    def _compute_reward(self, env, agent, newly_crashed, i):
        """Compute safety-focused reward for individual agent."""
        if agent in env._crashed_agents and agent not in newly_crashed:
            # Agent was already crashed - no reward calculation needed
            reward = 0.0
        else:
            # Progress component (reduced weight)
            current_s, _ = env.env.track.centerline.spline.calc_arclength_inaccurate(
                env.env.poses_x[i], env.env.poses_y[i]
            )
            prog = current_s - env._last_s[i]

            # Handle lap completion (when current_s wraps around to beginning)
            if prog > 0.9 * env.env.track.centerline.spline.s[-1]:
                prog = (env.env.track.centerline.spline.s[-1] - env._last_s[i]) + current_s

            # Safety component (minimum distance to walls)
            min_scan_distance = np.min(env.env.scans[i])
            safety_reward = min_scan_distance * 0.5  # Reward staying away from walls

            # Combined reward
            base_reward = prog * 0.5 + safety_reward

            # Apply collision penalty
            if agent in newly_crashed:
                reward = base_reward - 5.0  # Large collision penalty
            else:
                reward = base_reward

            # Update last track position for this agent
            env._last_s[i] = current_s

        return reward
    """Safety-focused reward encouraging careful driving."""

    def _get_rewards(self, env, newly_crashed):
        """Reward emphasizing safety with distance to walls."""
        rewards = []
        for i in range(env.env.num_agents):
            agent = env.agents[i]

            if agent in env._crashed_agents and agent not in newly_crashed:
                reward = 0.0
            else:
                # Progress component (reduced weight)
                current_s, _ = env.env.track.centerline.spline.calc_arclength_inaccurate(
                    env.env.poses_x[i], env.env.poses_y[i]
                )
                prog = current_s - env._last_s[i]

                if prog > 0.9 * env.env.track.centerline.spline.s[-1]:
                    prog = (env.env.track.centerline.spline.s[-1] - env._last_s[i]) + current_s

                # Safety component (minimum distance to walls)
                min_scan_distance = np.min(env.env.scans[i])
                safety_reward = min_scan_distance * 0.5  # Reward staying away from walls

                # Combined reward
                reward = prog * 0.5 + safety_reward

                # Large collision penalty
                if agent in newly_crashed:
                    reward -= 5.0

                env._last_s[i] = current_s

            rewards.append(reward)

        return rewards
