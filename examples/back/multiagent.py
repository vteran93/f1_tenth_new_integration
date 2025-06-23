"""Train two independent PPO agents on a single F1tenth track using Ray RLlib.

This example wraps :class:`~f1tenth_gym.envs.F110Env` in RLlib's
:class:`~ray.rllib.env.multi_agent_env.MultiAgentEnv` interface and launches two
PPO policies. Each agent learns completely independently (no parameter or reward
sharing).
"""

from __future__ import annotations

import numpy as np
import ray
import gymnasium as gym
import os
import time
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig

from f1tenth_gym.envs import F110Env

# Workaround for gymnasium compatibility issue with RLlib
import gymnasium.envs.registration
from enum import Enum

class VectorizeMode(Enum):
    ASYNC = "async"
    SYNC = "sync"

# Monkey patch the missing VectorizeMode
gymnasium.envs.registration.VectorizeMode = VectorizeMode


class RLLibF110(MultiAgentEnv):
    """Expose :class:`F110Env` in a format compatible with RLlib."""

    def __init__(self, env_config: dict | None = None):
        config = env_config or {}
        # Extract render_mode from config if provided, otherwise use None
        render_mode = config.get("render_mode", None)
        self.env = F110Env(config=config, render_mode=render_mode)
        self._num_agents = self.env.num_agents
        self.agents = [f"agent_{i}" for i in range(self._num_agents)]

        # observation/action spaces are identical for all agents
        # Extract single agent action space from multi-agent action space
        single_agent_action_space = gym.spaces.Box(
            low=self.env.action_space.low[0],
            high=self.env.action_space.high[0],
            shape=(self.env.action_space.shape[1],),
            dtype=np.float32
        )
        
        self.action_space = single_agent_action_space
        
        # RLlib expects action_spaces and observation_spaces for multi-agent
        self.action_spaces = {agent: single_agent_action_space for agent in self.agents}
        
        # Create single agent observation space from multi-agent observation space
        # For RLLib, each agent gets a subset of the full observation
        single_obs_spaces = {}
        for key, space in self.env.observation_space.spaces.items():
            if key == 'ego_idx':
                # ego_idx is shared across all agents, convert to array format
                single_obs_spaces[key] = gym.spaces.Box(
                    low=0, high=self._num_agents-1, shape=(1,), dtype=np.int32
                )
            elif key == 'scans':
                # Each agent gets its own scan - allow negative values for RLlib preprocessing
                single_obs_spaces[key] = gym.spaces.Box(
                    low=-space.high.max(),  # Allow negative values for preprocessed scans
                    high=space.high[0], 
                    shape=(space.shape[1],), 
                    dtype=space.dtype
                )
            else:
                # Other observations: each agent gets its own value as 1-element array
                single_obs_spaces[key] = gym.spaces.Box(
                    low=space.low.min(),
                    high=space.high.max(),
                    shape=(1,),
                    dtype=space.dtype
                )
        
        single_agent_obs_space = gym.spaces.Dict(single_obs_spaces)
        self.observation_space = single_agent_obs_space
        
        # RLlib expects observation_spaces for multi-agent
        self.observation_spaces = {agent: single_agent_obs_space for agent in self.agents}

        # keep track of progress for individual rewards
        self._last_s = [0.0] * self._num_agents

    # Helper to compute per-agent rewards following the environment logic
    def _agent_rewards(self):
        rewards = []
        for i in range(self._num_agents):
            current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
                self.env.poses_x[i], self.env.poses_y[i]
            )
            prog = current_s - self._last_s[i]
            if prog > 0.9 * self.env.track.centerline.spline.s[-1]:
                prog = (
                    self.env.track.centerline.spline.s[-1] - self._last_s[i]
                ) + current_s
            r = prog - (1.0 if self.env.collisions[i] else 0.0)
            rewards.append(float(r))
            self._last_s[i] = current_s
        return rewards

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_s = [0.0] * self._num_agents
        
        # Debug logging
        positions = [(self.env.poses_x[i], self.env.poses_y[i]) for i in range(self._num_agents)]
        print(f"🔄 Environment reset - agents at positions: {positions}")
        
        # Split multi-agent observation into per-agent observations
        agent_obs = {}
        for i, agent in enumerate(self.agents):
            agent_obs[agent] = {}
            for key, value in obs.items():
                if key == 'ego_idx':
                    # ego_idx is shared across all agents, convert to numpy array
                    agent_obs[agent][key] = np.array([value], dtype=np.int32)
                elif key == 'scans':
                    # Each agent gets its own scan (already a numpy array)
                    agent_obs[agent][key] = value[i]
                else:
                    # Other observations: extract value and ensure it's a numpy array
                    agent_obs[agent][key] = np.array([value[i]], dtype=np.float32)
        
        return agent_obs, {agent: info for agent in self.agents}

    def step(self, action_dict):
        actions = np.stack([action_dict[a] for a in self.agents])
        obs, _, terminated, truncated, info = self.env.step(actions)
        done = bool(terminated or truncated)

        # Split multi-agent observation into per-agent observations
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            obs_dict[agent] = {}
            for key, value in obs.items():
                if key == 'ego_idx':
                    # ego_idx is shared across all agents, convert to numpy array
                    obs_dict[agent][key] = np.array([value], dtype=np.int32)
                elif key == 'scans':
                    # Each agent gets its own scan (already a numpy array)
                    obs_dict[agent][key] = value[i]
                else:
                    # Other observations: extract value and ensure it's a numpy array
                    obs_dict[agent][key] = np.array([value[i]], dtype=np.float32)

        rew_list = self._agent_rewards()
        rew_dict = {agent: rew_list[i] for i, agent in enumerate(self.agents)}
        terminated_dict = {agent: done for agent in self.agents}
        terminated_dict["__all__"] = done
        truncated_dict = {agent: False for agent in self.agents}
        truncated_dict["__all__"] = False
        info_dict = {agent: info for agent in self.agents}
        
        # Debug logging for rewards and episode completion
        if done:
            total_reward = sum(rew_list)
            print(f"🏁 Episode completed! Agent rewards: {rew_dict}, Total: {total_reward:.3f}")
            print(f"   Collisions: {[bool(self.env.collisions[i]) for i in range(self._num_agents)]}")
        
        return obs_dict, rew_dict, terminated_dict, truncated_dict, info_dict

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


if __name__ == "__main__":
    # Toggle this to train or evaluate
    train = True
    
    if train:
        # TRAINING MODE
        # Create a unique run directory for TensorBoard logs
        timestamp = str(int(time.time()))
        run_id = f"multiagent_ppo_run_{timestamp}"
        log_dir = f"runs/{run_id}"
        model_dir = f"models/{run_id}"
        
        # Create directories if they don't exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"🚀 Starting F1tenth multi-agent PPO training...")
        print(f"📊 TensorBoard logs will be saved to: {log_dir}")
        print(f"💾 Models will be saved to: {model_dir}")
        print(f"📈 To view logs, run: tensorboard --logdir={log_dir}")
        print("=" * 70)
    
    # Configuration of the F1tenth environment
    env_config = {
        "map": "Spielberg",
        "num_agents": 2,
        "timestep": 0.01,
        "num_beams": 36,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "original"},
        "reset_config": {"type": "rl_random_static"},
    }

    def env_creator(config):
        return RLLibF110(config)

    register_env("f1tenth_multi", env_creator)

    # Create an RLlib PPO configuration with independent policies
    temp_env = env_creator(env_config)
    policies = {
        agent: PolicySpec(
            None,
            temp_env.observation_space,
            temp_env.action_space,
            {},
        )
        for agent in temp_env.agents
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return agent_id

    config = (
        PPOConfig()
        .environment("f1tenth_multi", env_config=env_config)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .env_runners(num_env_runners=0)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .debugging(log_level="INFO")
        .callbacks(None)  # Could add custom callbacks here for more detailed logging
    )

    ray.init(ignore_reinit_error=True)
    
    if train:
        # TRAINING CODE
        # Configure the algorithm
        algo = config.build()

        # Train with detailed logging to understand reward metrics
        print("🚀 Starting F1tenth multi-agent PPO training with TensorBoard logging...")
        print("=" * 70)
        print("🚀 Starting F1tenth multi-agent PPO training with TensorBoard logging...")
        print("=" * 70)
        
        iteration_count = 0
        
        # Create a simple logger for TensorBoard (manual logging)
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=log_dir)
            use_tensorboard = True
            print(f"📊 TensorBoard logging enabled: {log_dir}")
        except ImportError:
            tb_writer = None
            use_tensorboard = False
            print("⚠️  TensorBoard not available - install tensorboard with: pip install tensorboard")
        
        while True:
            result = algo.train()
            iteration_count += 1
            
            # Extract detailed metrics
            timesteps_total = result.get("timesteps_total", 0)
            timesteps_this_iter = result.get("timesteps_this_iter", 0)
            episodes_total = result.get("episodes_total", 0)
            episodes_this_iter = result.get("episodes_this_iter", 0)
            
            # Try different reward keys
            episode_reward_mean = result.get("episode_reward_mean")
            episode_reward_max = result.get("episode_reward_max")
            episode_reward_min = result.get("episode_reward_min")
            episode_len_mean = result.get("episode_len_mean")
            
            # Custom reward metrics (per agent)
            custom_metrics = result.get("custom_metrics", {})
            policy_rewards = result.get("policy_reward_mean", {})
            
            print(f"\n📊 Iteration {iteration_count}:")
            print(f"  🎯 Timesteps: {timesteps_this_iter:,} this iter | {timesteps_total:,} total")
            print(f"  🎮 Episodes: {episodes_this_iter} this iter | {episodes_total} total")
            
            if episode_reward_mean is not None:
                print(f"  🏆 Episode Rewards: mean={episode_reward_mean:.3f}, min={episode_reward_min:.3f}, max={episode_reward_max:.3f}")
            else:
                print(f"  ⚠️  Episode reward_mean: N/A (no episodes completed yet)")
                
            if episode_len_mean is not None:
                print(f"  ⏱️  Episode length: {episode_len_mean:.1f} steps")
                
            if policy_rewards:
                print(f"  🤖 Policy rewards: {policy_rewards}")
                
            if custom_metrics:
                print(f"  📈 Custom metrics: {custom_metrics}")
                
            # Print agent-specific info from the info dict
            info = result.get("info", {})
            if "learner" in info:
                learner_info = info["learner"]
                for agent_id, agent_info in learner_info.items():
                    if isinstance(agent_info, dict) and "learner_stats" in agent_info:
                        stats = agent_info["learner_stats"]
                        if "policy_loss" in stats:
                            print(f"  🧠 {agent_id} policy_loss: {stats['policy_loss']:.6f}")
            
            # Check if we have any episodes completed at all
            if episodes_total == 0:
                print(f"  ℹ️  No episodes completed yet - agents still exploring")
            
            print(f"  {'='*50}")
            
            # Log to TensorBoard if available
            if use_tensorboard and tb_writer:
                # Log basic metrics using timesteps_total as x-axis
                tb_writer.add_scalar("training/timesteps_this_iter", timesteps_this_iter, timesteps_total)
                tb_writer.add_scalar("training/episodes_total", episodes_total, timesteps_total)
                tb_writer.add_scalar("training/episodes_this_iter", episodes_this_iter, timesteps_total)
                tb_writer.add_scalar("training/iteration", iteration_count, timesteps_total)
                
                if episode_reward_mean is not None:
                    tb_writer.add_scalar("reward/episode_mean", episode_reward_mean, timesteps_total)
                    tb_writer.add_scalar("reward/episode_max", episode_reward_max, timesteps_total)
                    tb_writer.add_scalar("reward/episode_min", episode_reward_min, timesteps_total)
                
                if episode_len_mean is not None:
                    tb_writer.add_scalar("episode/length_mean", episode_len_mean, timesteps_total)
                
                # Log individual agent rewards if available
                if policy_rewards:
                    for agent_id, reward in policy_rewards.items():
                        tb_writer.add_scalar(f"agent_reward/{agent_id}", reward, timesteps_total)
                
                # Log policy losses if available
                if "learner" in info:
                    learner_info = info["learner"]
                    for agent_id, agent_info in learner_info.items():
                        if isinstance(agent_info, dict) and "learner_stats" in agent_info:
                            stats = agent_info["learner_stats"]
                            if "policy_loss" in stats:
                                tb_writer.add_scalar(f"loss/{agent_id}_policy", stats["policy_loss"], timesteps_total)
                            if "vf_loss" in stats:
                                tb_writer.add_scalar(f"loss/{agent_id}_value", stats["vf_loss"], timesteps_total)
                            if "entropy" in stats:
                                tb_writer.add_scalar(f"loss/{agent_id}_entropy", stats["entropy"], timesteps_total)
                
                tb_writer.flush()
            
            # Stop condition
            if timesteps_total >= 20_000:
                print(f"\n🎯 Training completed after {timesteps_total:,} timesteps!")
                break
                
            # Save checkpoint every 5 iterations
            if iteration_count % 5 == 0:
                checkpoint_path = algo.save(model_dir)
                print(f"💾 Checkpoint saved: {checkpoint_path}")

        # Close TensorBoard writer
        if use_tensorboard and tb_writer:
            tb_writer.close()

        # Save final model
        final_checkpoint = algo.save(model_dir)
        print(f"\n🎉 Training completed!")
        print(f"💾 Final model saved to: {final_checkpoint}")
        print(f"📊 TensorBoard logs saved to: {log_dir}")
        print(f"📈 To view training metrics, run: tensorboard --logdir={log_dir}")
        print(f"🔄 To continue training from checkpoint, use: {final_checkpoint}")

        algo.stop()
    
    else:
        # EVALUATION MODE
        # Update this path to point to your trained model checkpoint
        model_checkpoint_path = os.path.abspath("examples/models/multiagent_ppo_run_1749583871")
        
        print(f"🎮 Starting F1tenth multi-agent PPO evaluation...")
        print(f"📂 Loading model from: {model_checkpoint_path}")
        print("=" * 70)
        
        # Update env_config for evaluation with rendering
        env_config["render_mode"] = "human"
        
        try:
            # Load the trained algorithm from checkpoint
            print(f"📥 Loading trained model...")
            algo = config.build()
            algo.restore(model_checkpoint_path)
            print(f"✅ Model loaded successfully!")
            
            # Create evaluation environment with rendering
            eval_env = env_creator(env_config)
            
            # Run evaluation episodes
            num_episodes = 5
            print(f"🏁 Running {num_episodes} evaluation episodes...")
            
            for episode in range(num_episodes):
                print(f"\n🎯 Episode {episode + 1}/{num_episodes}")
                
                obs_dict, _ = eval_env.reset()
                done = False
                episode_rewards = {agent: 0.0 for agent in eval_env.agents}
                step_count = 0
                
                while not done:
                    # Get actions from trained policies
                    action_dict = {}
                    for agent_id, obs in obs_dict.items():
                        action, _ = algo.compute_single_action(
                            obs, 
                            policy_id=agent_id,
                            explore=False  # Use deterministic actions for evaluation
                        )
                        
                        # The trained model outputs 1D actions (steering only)
                        # Convert to 2D actions [steering, speed] for the environment
                        if isinstance(action, np.ndarray) and action.shape == ():
                            # Single scalar action - treat as steering, add fixed speed
                            steering = float(action)
                            speed = 5.0  # Fixed forward speed
                            action_2d = np.array([steering, speed], dtype=np.float32)
                        elif hasattr(action, '__len__') and len(action) == 2:
                            # Already 2D action
                            action_2d = np.array(action, dtype=np.float32)
                        else:
                            # Single value, convert to 2D
                            steering = float(action)
                            speed = 5.0  # Fixed forward speed  
                            action_2d = np.array([steering, speed], dtype=np.float32)
                        
                        action_dict[agent_id] = action_2d
                        print(f"🎮 {agent_id} - steering: {action_2d[0]:.3f}, speed: {action_2d[1]:.3f}")
                    
                    # Step environment
                    obs_dict, rew_dict, terminated_dict, truncated_dict, info_dict = eval_env.step(action_dict)
                    
                    # Accumulate rewards
                    for agent_id, reward in rew_dict.items():
                        episode_rewards[agent_id] += reward
                    
                    # Render the environment
                    eval_env.render()
                    
                    # Add small delay to make visualization smoother
                    time.sleep(0.01)
                    
                    done = terminated_dict.get("__all__", False)
                    step_count += 1
                    
                    # Safety limit to avoid infinite episodes
                    if step_count > 10000:
                        print("⚠️  Episode reached step limit, ending...")
                        break
                
                print(f"   📊 Episode {episode + 1} results:")
                print(f"   🏆 Agent rewards: {episode_rewards}")
                print(f"   📏 Episode length: {step_count} steps")
                print(f"   💥 Collisions: {[bool(eval_env.env.collisions[i]) for i in range(eval_env._num_agents)]}")
                
                # Wait a bit between episodes
                time.sleep(1.0)
            
            print(f"\n🎉 Evaluation completed!")
            print(f"✅ All {num_episodes} episodes finished successfully!")
            
            eval_env.close()
            algo.stop()
            
        except FileNotFoundError:
            print(f"❌ Error: Model checkpoint not found at {model_checkpoint_path}")
            print(f"💡 Make sure to update the model_checkpoint_path variable to point to your trained model")
            print(f"🔍 Available checkpoints in models/ directory:")
            
            # List available model directories
            models_dir = "models"
            if os.path.exists(models_dir):
                for run_dir in os.listdir(models_dir):
                    run_path = os.path.join(models_dir, run_dir)
                    if os.path.isdir(run_path):
                        print(f"   📁 {run_dir}/")
                        # List checkpoints in each run directory
                        for item in os.listdir(run_path):
                            if item.startswith("checkpoint_"):
                                print(f"      🔗 {item}")
            else:
                print(f"   📁 No models directory found")
        
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()

    ray.shutdown()