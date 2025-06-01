#!/usr/bin/env python3
"""
SAC Cloud Training Script with S3 Integration and Auto-save
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


class CloudSaveCallback(BaseCallback):
    def __init__(self, save_freq: int, s3_bucket: str, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.s3_bucket = s3_bucket
        self.save_path = save_path
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
        """Save model locally and upload to S3"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"sac_checkpoint_{self.n_calls}_{timestamp}.zip"
        model_path = os.path.join(self.save_path, model_filename)

        try:
            # Save model locally
            self.model.save(model_path)

            # Upload to S3
            s3_key = f"models/sac/{model_filename}"
            self.s3_client.upload_file(model_path, self.s3_bucket, s3_key)

            if self.verbose > 0:
                print(f"✅ Checkpoint {self.n_calls}: Model saved and uploaded to S3: s3://{self.s3_bucket}/{s3_key}")

            # Save and upload checkpoint info
            checkpoint_info = {
                "timesteps_completed": self.n_calls,
                "model_path": model_path,
                "s3_path": f"s3://{self.s3_bucket}/{s3_key}",
                "timestamp": timestamp,
                "training_type": "sac"
            }

            checkpoint_file = os.path.join(self.save_path, "latest_checkpoint.json")
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_info, f, indent=2)

            # Upload checkpoint info to S3
            self.s3_client.upload_file(checkpoint_file, self.s3_bucket, "checkpoints/sac_latest_checkpoint.json")

            # Clean up old local models to save disk space (keep last 3)
            self._cleanup_old_models()

        except Exception as e:
            print(f"❌ Error saving/uploading model at step {self.n_calls}: {e}")

    def _cleanup_old_models(self):
        """Keep only the last 3 local model files to save disk space"""
        try:
            pattern = os.path.join(self.save_path, "sac_checkpoint_*.zip")
            model_files = glob.glob(pattern)
            if len(model_files) > 3:
                # Sort by modification time, keep newest 3
                model_files.sort(key=os.path.getmtime, reverse=True)
                for old_file in model_files[3:]:
                    os.remove(old_file)
                    if self.verbose > 0:
                        print(f"🗑️  Cleaned up old model: {os.path.basename(old_file)}")
        except Exception as e:
            print(f"Warning: Failed to cleanup old models: {e}")


def check_gpu_availability():
    """Check if CUDA/GPU is available"""
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  CUDA not available, using CPU")
        return False


def download_checkpoint_from_s3(s3_bucket: str, local_path: str):
    """Download latest checkpoint from S3 if available"""
    s3_client = boto3.client('s3')
    try:
        # Check if checkpoint exists in S3
        checkpoint_key = "checkpoints/sac_latest_checkpoint.json"
        checkpoint_file = os.path.join(local_path, "latest_checkpoint.json")

        s3_client.download_file(s3_bucket, checkpoint_key, checkpoint_file)

        with open(checkpoint_file, 'r') as f:
            checkpoint_info = json.load(f)

        # Download the model file
        s3_model_key = checkpoint_info['s3_path'].split('/')[-2:]  # Extract key from s3://bucket/key
        s3_model_key = '/'.join(s3_model_key)
        local_model_path = os.path.join(local_path, os.path.basename(checkpoint_info['model_path']))

        s3_client.download_file(s3_bucket, s3_model_key, local_model_path)

        checkpoint_info['model_path'] = local_model_path
        print(f"✅ Downloaded checkpoint from S3: {checkpoint_info['timesteps_completed']} timesteps completed")
        return checkpoint_info

    except Exception as e:
        print(f"ℹ️  No previous checkpoint found in S3: {e}")
        return None


def main():
    """Main training function"""
    print("🚀 Starting F1Tenth SAC Cloud Training")

    # Configuration from environment variables
    total_timesteps = int(os.environ.get('TOTAL_TIMESTEPS', 1000000))
    save_frequency = int(os.environ.get('SAVE_FREQUENCY', 50000))
    s3_bucket = os.environ.get('S3_BUCKET_NAME')

    if not s3_bucket:
        raise ValueError("❌ S3_BUCKET_NAME environment variable is required")

    print(f"📊 Configuration:")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Save frequency: {save_frequency:,}")
    print(f"   S3 bucket: {s3_bucket}")

    # Check GPU
    gpu_available = check_gpu_availability()
    device = "cuda" if gpu_available else "cpu"

    # Setup directories
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Try to download existing checkpoint
    checkpoint = download_checkpoint_from_s3(s3_bucket, models_dir)

    # Initialize wandb
    if checkpoint and checkpoint.get("run_id"):
        print(f"🔄 Resuming wandb run: {checkpoint['run_id']}")
        run = wandb.init(
            project="f1tenth_gym_sac_cloud",
            id=checkpoint["run_id"],
            resume="must",
            sync_tensorboard=True,
            save_code=True,
        )
    else:
        print("🆕 Starting new wandb run")
        run = wandb.init(
            project="f1tenth_gym_sac_cloud",
            sync_tensorboard=True,
            save_code=True,
        )

    # Create environment
    print("🏁 Creating F1Tenth environment...")
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "num_beams": 36,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "rl"},
            "reset_config": {"type": "rl_random_static"},
        },
    )

    # Action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Load existing model or create new one
    if checkpoint and os.path.exists(checkpoint['model_path']):
        print(f"🔄 Resuming training from checkpoint: {checkpoint['model_path']}")
        print(f"   Timesteps completed: {checkpoint['timesteps_completed']:,}")

        model = SAC.load(
            checkpoint["model_path"],
            env=env,
            device=device,
            print_system_info=True
        )

        remaining_timesteps = total_timesteps - checkpoint["timesteps_completed"]
        print(f"   Remaining timesteps: {remaining_timesteps:,}")
    else:
        print("🆕 Starting new training from scratch")
        # SAC hyperparameters optimized for continuous control
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            device=device,
            seed=42,
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
        remaining_timesteps = total_timesteps

    # Setup callbacks
    cloud_save_callback = CloudSaveCallback(
        save_freq=save_frequency,
        s3_bucket=s3_bucket,
        save_path=models_dir,
        verbose=1
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=f"models/{run.id}",
        verbose=2
    )

    callbacks = [cloud_save_callback, wandb_callback]

    # Train the model
    if remaining_timesteps > 0:
        print(f"🏋️  Starting training for {remaining_timesteps:,} timesteps...")
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            reset_num_timesteps=False if checkpoint else True
        )
    else:
        print("✅ Training already completed!")

    # Save final model
    print("💾 Saving final model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_filename = f"sac_final_{run.id}_{timestamp}.zip"
    final_model_path = os.path.join(models_dir, final_model_filename)
    model.save(final_model_path)

    # Upload final model to S3
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(final_model_path, s3_bucket, f"models/sac/{final_model_filename}")
        print(f"✅ Final model uploaded to S3: s3://{s3_bucket}/models/sac/{final_model_filename}")
    except Exception as e:
        print(f"❌ Failed to upload final model to S3: {e}")

    # Log final training summary
    print("\n🎉 Training Summary:")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Wandb run ID: {run.id}")
    print(f"   Final model: {final_model_filename}")
    print(f"   S3 bucket: {s3_bucket}")

    run.finish()

    # Auto-shutdown instance
    print("\n🛑 Training completed. Shutting down instance in 60 seconds...")
    print("   (Cancel with Ctrl+C if you want to keep the instance running)")
    import time
    time.sleep(60)
    os.system("sudo shutdown now")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
