#!/usr/bin/env python3
"""
F1TENTH Training Script with Custom Oval Track
Supports both PPO and SAC algorithms with polymorphic reward system.
"""

import argparse
import os
import time
import tempfile
from datetime import datetime

import numpy as np
import ray
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
import torch.nn as nn

# Import our custom environment and reward system
from multiagent_env import MultiAgentF110PPO, MultiAgentF110SAC
from rewards import get_reward_function, list_available_rewards


def install_custom_tracks():
    """Install custom tracks to f1tenth_gym maps directory."""
    source_dir = "./tracks"
    target_dir = "/home/victor/repositories/tfm/new_integration/rl_examples/f1tenth_gym/maps"

    if not os.path.exists(source_dir):
        print(f"❌ Source tracks directory '{source_dir}' not found")
        print("💡 Run 'python create_tracks.py' first to generate tracks")
        return False

    if not os.path.exists(target_dir):
        print(f"❌ Target maps directory '{target_dir}' not found")
        return False

    print("📥 Installing custom tracks to f1tenth_gym...")

    # Copy tracks
    tracks_copied = 0
    for track in os.listdir(source_dir):
        source_track = os.path.join(source_dir, track)
        target_track = os.path.join(target_dir, track)

        if os.path.isdir(source_track):
            # Check if track has required files
            if (os.path.exists(f"{source_track}/{track}_map.png") and
                    os.path.exists(f"{source_track}/{track}_map.yaml")):

                # Copy track directory
                import shutil
                if os.path.exists(target_track):
                    shutil.rmtree(target_track)
                shutil.copytree(source_track, target_track)
                print(f"  ✅ Installed: {track}")
                tracks_copied += 1
            else:
                print(f"  ⚠️  Skipped: {track} (missing files)")

    print(f"📊 Installed {tracks_copied} custom tracks")
    return tracks_copied > 0


def get_env_config(track_name="oval_small", render_mode=None):
    """
    Get environment configuration for custom track.

    Args:
        track_name: Name of track (maps are loaded from f1tenth_gym/maps/ directory)
        render_mode: Rendering mode (None, "human", etc.)
    """
    # Check if track exists in f1tenth_gym maps directory
    maps_dir = "/home/victor/repositories/tfm/new_integration/rl_examples/f1tenth_gym/maps"
    track_dir = f"{maps_dir}/{track_name}"

    if os.path.exists(track_dir) and os.path.exists(f"{track_dir}/{track_name}_map.yaml"):
        # Use our custom track (track name without path)
        track_path = track_name
        print(f"✅ Using custom track: {track_name}")
    else:
        # Check what tracks are available
        print(f"⚠️  Track '{track_name}' not found in maps directory")
        print("Available tracks:")
        if os.path.exists(maps_dir):
            for track in os.listdir(maps_dir):
                if os.path.isdir(f"{maps_dir}/{track}") and os.path.exists(f"{maps_dir}/{track}/{track}_map.yaml"):
                    print(f"  - {track}")
        track_path = "Spielberg"  # Fallback to default
        print(f"Using fallback track: {track_path}")

    return {
        "map": track_path,
        "num_agents": 2,
        "timestep": 0.01,
        "num_beams": 36,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "original"},
        "reset_config": {"type": "cl_grid_static"},  # PPO uses cl_grid_static
        "render_mode": render_mode,
    }


def get_sac_env_config(track_name="oval_small", render_mode=None):
    """SAC-specific environment configuration."""
    config = get_env_config(track_name, render_mode)
    config["reset_config"] = {"type": "rl_random_static"}  # SAC uses rl_random_static
    return config


def setup_ppo_training(track_name="oval_small", reward_type="default"):
    """Setup PPO training configuration."""
    print(f"🚀 Setting up PPO training on track '{track_name}' with '{reward_type}' rewards")

    # Get reward function
    reward_function = get_reward_function("ppo", reward_type)

    # Register environment
    def create_env(env_config):
        return MultiAgentF110PPO(env_config, reward_function=reward_function)

    register_env("f1tenth_ppo", create_env)

    # Create temporary environment for policy setup
    temp_env = MultiAgentF110PPO(get_env_config(track_name), reward_function=reward_function)
    policies = {agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
                for agent in temp_env.agents}
    temp_env.close()

    # Configure PPO
    config = (PPOConfig()
              .environment("f1tenth_ppo", env_config=get_env_config(track_name))
              .framework("torch")
              .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
              .env_runners(num_env_runners=0)
              .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id)
              .training(train_batch_size=4000)
              )

    return policies, config


def setup_sac_training(track_name="oval_small", reward_type="default"):
    """Setup SAC training configuration."""
    print(f"🚀 Setting up SAC training on track '{track_name}' with '{reward_type}' rewards")

    # Get reward function
    reward_function = get_reward_function("sac", reward_type)

    # Register environment
    def create_env(env_config):
        return MultiAgentF110SAC(env_config, reward_function=reward_function)

    register_env("f1tenth_sac", create_env)

    # Create temporary environment for policy setup
    temp_env = MultiAgentF110SAC(get_sac_env_config(track_name), reward_function=reward_function)
    policies = {agent: PolicySpec(None, temp_env.observation_space[agent], temp_env.action_space[agent], {})
                for agent in temp_env.agents}
    temp_env.close()

    # Configure SAC
    config = (SACConfig()
              .environment("f1tenth_sac", env_config=get_sac_env_config(track_name))
              .framework("torch")
              .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
              .env_runners(num_env_runners=0)
              .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id)
              .training(replay_buffer_config={'type': 'MultiAgentPrioritizedReplayBuffer',
                                              'prioritized_replay_alpha': 0.6,
                                              'prioritized_replay_beta': 0.4,
                                              'prioritized_replay_eps': 0.2
                                              })
              )

    return policies, config


def custom_log_creator(custom_path: str, prefix: str):
    """Create custom logger for TensorBoard output."""
    def logger_creator(config):
        os.makedirs(custom_path, exist_ok=True)
        run_dir = tempfile.mkdtemp(
            prefix=f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_",
            dir=custom_path
        )
        return UnifiedLogger(config, run_dir, loggers=None)
    return logger_creator


def setup_ray_and_algo(config, algorithm):
    """Initialize Ray and build algorithm with TensorBoard logging."""
    ray.init(ignore_reinit_error=True)

    # Create per-run log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.abspath(f"runs/multiagent_{algorithm}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Build with custom logger
    return config.build(
        logger_creator=custom_log_creator(log_dir, "rllib_run")
    )


def run_training(algorithm, track_name="oval_small", reward_type="default", total_timesteps=100_000):
    """Run training for specified algorithm."""
    print(f"🏁 Starting {algorithm.upper()} training...")
    print(f"📊 Track: {track_name}")
    print(f"🏆 Reward: {reward_type}")
    print(f"⏱️  Target timesteps: {total_timesteps:,}")
    print("=" * 50)

    # Setup algorithm-specific configuration
    if algorithm == "ppo":
        policies, config = setup_ppo_training(track_name, reward_type)
    elif algorithm == "sac":
        policies, config = setup_sac_training(track_name, reward_type)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Initialize algorithm
    algo = setup_ray_and_algo(config, algorithm)

    # Create model directory
    timestamp = str(int(time.time()))
    model_dir = f"models/multiagent_{algorithm}_oval_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    print(f"💾 Model will be saved to: {model_dir}")

    # Training loop
    save_every = 1000 if algorithm == "sac" else 5000  # SAC saves more frequently
    current_timesteps = 0
    iteration = 0

    try:
        while current_timesteps < total_timesteps:
            result = algo.train()
            iteration += 1
            current_timesteps = result["timesteps_total"]

            # Print progress
            if iteration % 10 == 0:
                reward_mean = result.get("episode_reward_mean", 0)
                print(f"Iteration {iteration:3d}: {current_timesteps:6,} steps, "
                      f"reward_mean: {reward_mean:8.2f}")

            # Save checkpoint using timesteps_total instead of timesteps_this_iter
            if current_timesteps // save_every > (current_timesteps - result.get("timesteps_total", 0)) // save_every:
                checkpoint_path = algo.save(model_dir)
                print(f"💾 Checkpoint saved: {checkpoint_path}")

            # Check if training is complete
            if current_timesteps >= total_timesteps:
                break

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    # Save final model
    final_checkpoint = algo.save(model_dir)
    print(f"✅ Training completed!")
    print(f"📊 Final timesteps: {current_timesteps:,}")
    print(f"💾 Final model saved: {final_checkpoint}")

    algo.stop()
    return model_dir


def run_evaluation(algorithm, track_name="oval_small", reward_type="default", num_episodes=3):
    """Run evaluation with latest trained model."""
    print(f"🎯 Starting {algorithm.upper()} evaluation...")

    # Find latest model
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"❌ Models directory '{models_dir}' not found")
        print("💡 Run training first with --train flag")
        return

    # Look for models matching algorithm
    run_dirs = [d for d in os.listdir(models_dir) if d.startswith(f"multiagent_{algorithm}_oval_")]
    if not run_dirs:
        print(f"❌ No {algorithm.upper()} models found")
        print(f"💡 Available models: {[d for d in os.listdir(models_dir) if d.startswith('multiagent_')]}")
        return

    latest_model = max(run_dirs, key=lambda x: int(x.split("_")[-1]))
    model_path = os.path.abspath(os.path.join(models_dir, latest_model))
    print(f"📁 Using model: {latest_model}")

    # Setup algorithm
    if algorithm == "ppo":
        policies, config = setup_ppo_training(track_name, reward_type)
        env_config = get_env_config(track_name, render_mode="human")
        reward_function = get_reward_function("ppo", reward_type)
        eval_env = MultiAgentF110PPO(env_config, reward_function=reward_function)
    else:
        policies, config = setup_sac_training(track_name, reward_type)
        env_config = get_sac_env_config(track_name, render_mode="human")
        reward_function = get_reward_function("sac", reward_type)
        eval_env = MultiAgentF110SAC(env_config, reward_function=reward_function)

    algo = setup_ray_and_algo(config, algorithm)
    algo.restore(model_path)

    # Run evaluation episodes
    for episode in range(num_episodes):
        print(f"\n🏃 Episode {episode + 1}/{num_episodes}")
        obs, info = eval_env.reset(seed=episode)
        episode_reward = {agent: 0.0 for agent in eval_env.agents}
        step = 0

        while True:
            # Get actions from policy (without exploration for evaluation)
            actions = {}
            for agent in obs.keys():
                actions[agent] = algo.compute_single_action(obs[agent], policy_id=agent, explore=False)

            # Step environment
            obs, rewards, terminated, truncated, info = eval_env.step(actions)
            
            # Render after each step for smooth animation
            eval_env.render()

            # Accumulate rewards
            for agent, reward in rewards.items():
                episode_reward[agent] += reward

            step += 1

            # Check if episode is done
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

            if step >= 10000:  # Increase max episode length for better evaluation
                print("⏱️  Max episode length reached")
                break

        print(f"📊 Episode {episode + 1} results:")
        for agent, reward in episode_reward.items():
            print(f"  {agent}: {reward:.2f} total reward")
        print(f"  Duration: {step} steps")

    eval_env.close()
    algo.stop()
    print("✅ Evaluation completed!")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="F1TENTH Multi-Agent Training with Oval Track")
    parser.add_argument("--train", action="store_true", help="Run training mode")
    parser.add_argument("--algo", choices=["ppo", "sac"],
                        help="Algorithm to use (ppo or sac)")
    parser.add_argument("--track", default="oval_small",
                        help="Track name (default: oval_small)")
    parser.add_argument("--reward", default="default",
                        help="Reward function type (default: default)")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps (default: 100,000)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of evaluation episodes (default: 3)")
    parser.add_argument("--list-rewards", action="store_true",
                        help="List available reward functions")
    parser.add_argument("--install-tracks", action="store_true",
                        help="Install custom tracks to f1tenth_gym maps directory")

    args = parser.parse_args()

    # Handle special options
    if args.list_rewards:
        list_available_rewards()
        return

    if args.install_tracks:
        install_custom_tracks()
        return

    # Validate that algo is provided for training/evaluation
    if not args.algo:
        parser.error("--algo is required for training or evaluation")

    print("🏁 F1TENTH Multi-Agent Training Script")
    print("=" * 50)

    try:
        if args.train:
            run_training(args.algo, args.track, args.reward, args.timesteps)
            print("\n💡 To evaluate this model, run:")
            print(f"python {__file__} --algo {args.algo} --track {args.track} --reward {args.reward}")
        else:
            run_evaluation(args.algo, args.track, args.reward, args.episodes)

    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
