#!/usr/bin/env python3
"""
SAC Multi-Agent Cloud Training Script with S3 Integration and Auto-save
Optimized for AWS EC2 GPU instances
"""

import os
import sys
import json
import boto3
import wandb
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
import numpy as np
import glob
from datetime import datetime
import torch
from typing import List, Dict, Any


class MultiAgentCloudSaveCallback(BaseCallback):
    def __init__(self, save_freq: int, s3_bucket: str, save_path: str, agent_id: int, num_agents: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.s3_bucket = s3_bucket
        self.save_path = save_path
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.s3_client = boto3.client('s3')
        self.last_upload_step = 0

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0 and self.n_calls > self.last_upload_step:
            self.last_upload_step = self.n_calls
            self._save_and_upload()
        return True

    def _save_and_upload(self):
        """Save agent model locally and upload to S3"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"sac_multiagent_agent_{self.agent_id}_checkpoint_{self.n_calls}_{timestamp}.zip"
        model_path = os.path.join(self.save_path, model_filename)

        try:
            # Save model locally
            self.model.save(model_path)

            # Upload to S3
            s3_key = f"models/sac_multiagent/agent_{self.agent_id}/{model_filename}"
            self.s3_client.upload_file(model_path, self.s3_bucket, s3_key)

            if self.verbose > 0:
                print(
                    f"✅ Agent {self.agent_id} Checkpoint {self.n_calls}: Model saved and uploaded to S3: s3://{self.s3_bucket}/{s3_key}")

            # Save and upload checkpoint info
            checkpoint_info = {
                "timesteps_completed": self.n_calls,
                "model_path": model_path,
                "s3_path": f"s3://{self.s3_bucket}/{s3_key}",
                "timestamp": timestamp,
                "training_type": "sac_multiagent",
                "agent_id": self.agent_id,
                "num_agents": self.num_agents
            }

            checkpoint_file = os.path.join(self.save_path, f"agent_{self.agent_id}_latest_checkpoint.json")
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_info, f, indent=2)

            # Upload checkpoint info to S3
            self.s3_client.upload_file(checkpoint_file, self.s3_bucket,
                                       f"checkpoints/sac_multiagent_agent_{self.agent_id}_latest_checkpoint.json")

            # Clean up old local models to save disk space (keep last 2 per agent)
            self._cleanup_old_models()

        except Exception as e:
            print(f"❌ Error saving/uploading agent {self.agent_id} model at step {self.n_calls}: {e}")

    def _cleanup_old_models(self):
        """Keep only the last 2 local model files per agent to save disk space"""
        try:
            pattern = os.path.join(self.save_path, f"sac_multiagent_agent_{self.agent_id}_checkpoint_*.zip")
            model_files = glob.glob(pattern)
            if len(model_files) > 2:
                # Sort by modification time, keep newest 2
                model_files.sort(key=os.path.getmtime, reverse=True)
                for old_file in model_files[2:]:
                    os.remove(old_file)
                    if self.verbose > 0:
                        print(f"🗑️  Agent {self.agent_id}: Cleaned up old model: {os.path.basename(old_file)}")
        except Exception as e:
            print(f"Warning: Failed to cleanup old models for agent {self.agent_id}: {e}")


def check_gpu_availability():
    """Check if CUDA/GPU is available"""
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  CUDA not available, using CPU")
        return False


def create_multi_agent_env(num_agents: int, render_mode=None):
    """Create multi-agent F1Tenth environment"""
    config = {
        "map": "Spielberg",
        "num_agents": num_agents,
        "timestep": 0.01,
        "num_beams": 36,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "rl"},
        "reset_config": {"type": "rl_random_static"},
    }

    if render_mode:
        config["render_mode"] = render_mode

    return gym.make("f1tenth_gym:f1tenth-v0", config=config)


def extract_agent_observation(obs, agent_id: int, num_agents: int):
    """Extract observation for a specific agent from multi-agent observation"""
    if isinstance(obs, dict):
        return obs.get(f"agent_{agent_id}", obs)
    elif isinstance(obs, np.ndarray):
        # If observations are concatenated, split them
        obs_dim = len(obs) // num_agents
        start_idx = agent_id * obs_dim
        end_idx = (agent_id + 1) * obs_dim
        return obs[start_idx:end_idx]
    else:
        return obs


def combine_actions(actions: List[np.ndarray]) -> np.ndarray:
    """Combine individual agent actions into multi-agent action"""
    return np.concatenate(actions)


def load_checkpoint_from_s3(s3_bucket: str, agent_id: int) -> Dict[str, Any]:
    """Load the latest checkpoint info for an agent from S3"""
    s3_client = boto3.client('s3')
    checkpoint_key = f"checkpoints/sac_multiagent_agent_{agent_id}_latest_checkpoint.json"

    try:
        response = s3_client.get_object(Bucket=s3_bucket, Key=checkpoint_key)
        checkpoint_info = json.loads(response['Body'].read())
        print(f"📥 Found checkpoint for agent {agent_id} at step {checkpoint_info['timesteps_completed']}")
        return checkpoint_info
    except Exception as e:
        print(f"ℹ️  No checkpoint found for agent {agent_id}: {e}")
        return None


def download_model_from_s3(s3_bucket: str, s3_key: str, local_path: str) -> bool:
    """Download model from S3 to local path"""
    s3_client = boto3.client('s3')
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(s3_bucket, s3_key, local_path)
        print(f"📥 Downloaded model from s3://{s3_bucket}/{s3_key}")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False


def shutdown_instance():
    """Shutdown the EC2 instance"""
    try:
        import requests
        # Get instance ID from metadata
        instance_id = requests.get(
            'http://169.254.169.254/latest/meta-data/instance-id',
            timeout=10
        ).text

        # Shutdown using AWS CLI
        ec2_client = boto3.client('ec2')
        ec2_client.stop_instances(InstanceIds=[instance_id])
        print(f"🛑 Shutting down instance {instance_id}")

    except Exception as e:
        print(f"❌ Failed to shutdown instance: {e}")


def main():
    # Parse environment variables or use defaults
    S3_BUCKET = os.environ.get('S3_BUCKET_NAME', 'f1tenth-rl-models')
    TOTAL_TIMESTEPS = int(os.environ.get('TOTAL_TIMESTEPS', '1000000'))
    SAVE_FREQUENCY = int(os.environ.get('SAVE_FREQUENCY', '50000'))
    NUM_AGENTS = int(os.environ.get('NUM_AGENTS', '2'))

    print("=" * 80)
    print("🏎️  F1Tenth SAC Multi-Agent Cloud Training")
    print("=" * 80)
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Save Frequency: {SAVE_FREQUENCY:,}")
    print(f"Number of Agents: {NUM_AGENTS}")
    print("=" * 80)

    # Check GPU availability
    device = "cuda" if check_gpu_availability() else "cpu"

    # Initialize Weights & Biases
    run = wandb.init(
        project="f1tenth_gym_sac_multiagent_cloud",
        config={
            "total_timesteps": TOTAL_TIMESTEPS,
            "save_frequency": SAVE_FREQUENCY,
            "num_agents": NUM_AGENTS,
            "algorithm": "SAC",
            "device": device,
            "training_type": "multiagent"
        },
        sync_tensorboard=True,
        save_code=True,
    )

    # Create multi-agent environment
    print("🏁 Creating multi-agent environment...")
    env = create_multi_agent_env(NUM_AGENTS)
    print(f"✅ Environment created with {NUM_AGENTS} agents")

    # Create individual SAC models for each agent
    agents = []
    agent_callbacks = []
    save_path = f"/tmp/models/{run.id}"

    for agent_id in range(NUM_AGENTS):
        print(f"🤖 Setting up Agent {agent_id}...")

        # Create a dummy single-agent environment to get action/observation spaces
        single_env = create_multi_agent_env(1)

        # Add action noise for better exploration
        n_actions = single_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        # Check for existing checkpoint and resume if found
        checkpoint_info = load_checkpoint_from_s3(S3_BUCKET, agent_id)
        agent_model = None

        if checkpoint_info:
            # Try to resume from checkpoint
            s3_key = checkpoint_info['s3_path'].replace(f's3://{S3_BUCKET}/', '')
            local_model_path = os.path.join(save_path, f"agent_{agent_id}_checkpoint.zip")

            if download_model_from_s3(S3_BUCKET, s3_key, local_model_path):
                try:
                    agent_model = SAC.load(local_model_path, env=single_env, device=device)
                    print(
                        f"✅ Agent {agent_id}: Resumed from checkpoint at step {checkpoint_info['timesteps_completed']}")
                except Exception as e:
                    print(f"❌ Agent {agent_id}: Failed to load checkpoint, starting fresh: {e}")

        # Create new model if no checkpoint loaded
        if agent_model is None:
            agent_model = SAC(
                "MlpPolicy",
                single_env,
                verbose=1,
                tensorboard_log=f"runs/{run.id}/agent_{agent_id}",
                device=device,
                seed=42 + agent_id,
                action_noise=action_noise,
                learning_rate=3e-4,
                buffer_size=1_000_000,
                learning_starts=10000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef='auto',
                target_update_interval=1,
                policy_kwargs=dict(net_arch=[256, 256])
            )
            print(f"✅ Agent {agent_id}: Created new SAC model")

        # Create callback for this agent
        agent_callback = MultiAgentCloudSaveCallback(
            save_freq=SAVE_FREQUENCY,
            s3_bucket=S3_BUCKET,
            save_path=save_path,
            agent_id=agent_id,
            num_agents=NUM_AGENTS,
            verbose=1
        )

        agents.append(agent_model)
        agent_callbacks.append(agent_callback)
        single_env.close()

    print(f"✅ All {NUM_AGENTS} agents initialized")

    # Multi-agent training loop
    print("🚀 Starting multi-agent training...")

    try:
        obs, info = env.reset()
        step_count = 0

        for timestep in range(TOTAL_TIMESTEPS):
            # Get actions from all agents
            actions = []
            for agent_id in range(NUM_AGENTS):
                agent_obs = extract_agent_observation(obs, agent_id, NUM_AGENTS)
                action, _ = agents[agent_id].predict(agent_obs, deterministic=False)
                actions.append(action)

            # Combine actions and step environment
            combined_action = combine_actions(actions)
            next_obs, rewards, done, trunc, info = env.step(combined_action)

            # Train each agent with their individual experience
            for agent_id in range(NUM_AGENTS):
                agent_obs = extract_agent_observation(obs, agent_id, NUM_AGENTS)
                agent_next_obs = extract_agent_observation(next_obs, agent_id, NUM_AGENTS)
                agent_reward = rewards[agent_id] if isinstance(rewards, (list, np.ndarray)) else rewards

                # Store experience in agent's replay buffer
                agents[agent_id].replay_buffer.add(
                    agent_obs, agent_next_obs, actions[agent_id], agent_reward, done, [info]
                )

                # Train the agent if enough experience collected
                if timestep > agents[agent_id].learning_starts and timestep % agents[agent_id].train_freq == 0:
                    agents[agent_id]._train_step()

                # Call agent's callback
                agent_callbacks[agent_id].n_calls = timestep + 1
                agent_callbacks[agent_id].model = agents[agent_id]
                agent_callbacks[agent_id]._on_step()

            obs = next_obs

            # Reset environment if episode ended
            if done or trunc:
                obs, info = env.reset()

            # Log progress
            if timestep % 10000 == 0:
                print(f"Progress: {timestep:,}/{TOTAL_TIMESTEPS:,} ({timestep/TOTAL_TIMESTEPS*100:.1f}%)")

            step_count += 1

        print("✅ Training completed successfully!")

        # Save final models
        print("💾 Saving final models...")
        for agent_id, agent in enumerate(agents):
            agent_callbacks[agent_id]._save_and_upload()

    except KeyboardInterrupt:
        print("🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    finally:
        env.close()
        run.finish()

        # Auto-shutdown if running on EC2
        if os.environ.get('AUTO_SHUTDOWN', 'true').lower() == 'true':
            print("🔄 Auto-shutdown enabled, shutting down instance...")
            shutdown_instance()

    print("🏁 Multi-agent cloud training session completed!")


if __name__ == "__main__":
    main()
