#!/usr/bin/env python3
"""
Polymorphic Reward System for F1TENTH Multi-Agent Environment
Contains reward classes and control dictionary for algorithm-specific rewards.
"""

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


class DefaultProgressReward(BaseReward):
    """Default progress-based reward function (shared base implementation)."""

    def _get_rewards(self, env, newly_crashed):
        """Basic progress-based reward with collision penalty."""
        rewards = []
        for i in range(env.env.num_agents):
            agent = env.agents[i]

            if agent in env._crashed_agents and agent not in newly_crashed:
                reward = 0.0
            else:
                # Calculate track progress using centerline spline
                current_s, _ = env.env.track.centerline.spline.calc_arclength_inaccurate(
                    env.env.poses_x[i], env.env.poses_y[i]
                )

                # Calculate progress since last step
                prog = current_s - env._last_s[i]

                # Handle lap completion (when current_s wraps around to beginning)
                if prog > 0.9 * env.env.track.centerline.spline.s[-1]:
                    prog = (env.env.track.centerline.spline.s[-1] - env._last_s[i]) + current_s

                # Start with progress reward
                reward = prog

                # Apply collision penalty
                if agent in newly_crashed:
                    reward -= 1.0

                # Update last track position for this agent
                env._last_s[i] = current_s

            rewards.append(reward)

        return rewards


class PPOProgressReward(BaseReward):
    """PPO-specific progress reward (from multiagent_ppo.py)."""

    def _get_rewards(self, env, newly_crashed):
        """PPO reward function - direct progress with collision penalty."""
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
        rewards = []
        track_length = env.env.track.centerline.spline.s[-1]

        for i in range(env.env.num_agents):
            agent = env.agents[i]

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

            rewards.append(reward)

        return rewards


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
                current_s, _ = env.env.track.centerline.spline.calc_arclength_inaccurate(
                    env.env.poses_x[i], env.env.poses_y[i]
                )
                prog = current_s - env._last_s[i]

                if prog > 0.9 * env.env.track.centerline.spline.s[-1]:
                    prog = (env.env.track.centerline.spline.s[-1] - env._last_s[i]) + current_s

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


# Reward Function Control Dictionary
# Maps algorithm and reward type to specific reward class
REWARD_FUNCTIONS = {
    # PPO Rewards
    "ppo": {
        "default": PPOProgressReward(),
        "progress": PPOProgressReward(),
        "speed": SpeedReward(),
        "safety": SafetyReward(),
    },

    # SAC Rewards
    "sac": {
        "default": SACBasicReward(),
        "basic": SACBasicReward(),
        "gemini": SACGeminiReward(),
        "progress": SACBasicReward(),
        "speed": SpeedReward(),
        "safety": SafetyReward(),
    },

    # Shared rewards (algorithm-agnostic)
    "shared": {
        "default": DefaultProgressReward(),
        "progress": DefaultProgressReward(),
        "speed": SpeedReward(),
        "safety": SafetyReward(),
    }
}


def get_reward_function(algorithm, reward_type="default"):
    """
    Get reward function for specific algorithm and type.

    Args:
        algorithm: "ppo", "sac", or "shared"
        reward_type: "default", "progress", "speed", "safety", "gemini" (SAC only)

    Returns:
        Reward function instance

    Raises:
        ValueError: If algorithm or reward_type not found
    """
    if algorithm not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(REWARD_FUNCTIONS.keys())}")

    algo_rewards = REWARD_FUNCTIONS[algorithm]
    if reward_type not in algo_rewards:
        raise ValueError(
            f"Unknown reward type '{reward_type}' for algorithm '{algorithm}'. Available: {list(algo_rewards.keys())}")

    return algo_rewards[reward_type]


def list_available_rewards():
    """List all available reward functions by algorithm."""
    print("🏆 Available Reward Functions:")
    print("=" * 40)

    for algo, rewards in REWARD_FUNCTIONS.items():
        print(f"\n📊 {algo.upper()} Algorithm:")
        for reward_name, reward_obj in rewards.items():
            print(f"  - {reward_name}: {reward_obj.__class__.__name__}")

    print(f"\n💡 Usage: get_reward_function('ppo', 'progress')")


if __name__ == "__main__":
    # Demo the reward system
    list_available_rewards()

    # Test reward function retrieval
    try:
        ppo_reward = get_reward_function("ppo", "progress")
        sac_reward = get_reward_function("sac", "gemini")
        print(f"\n✅ Successfully loaded: {ppo_reward.__class__.__name__} and {sac_reward.__class__.__name__}")
    except Exception as e:
        print(f"❌ Error: {e}")
