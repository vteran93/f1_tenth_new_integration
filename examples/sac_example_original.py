import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

# if using wandb (recommended):
from wandb.integration.sb3 import WandbCallback
import wandb
import numpy as np
import os
import json
import glob

import os
import json
import glob

# Custom callback for incremental saving
class IncrementalSaveCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, checkpoint_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.checkpoint_path = checkpoint_path

    def _init_callback(self) -> None:
        # Create save directory if it doesn't exist
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save model
            model_path = os.path.join(self.save_path, f"sac_checkpoint_{self.n_calls}.zip")
            self.model.save(model_path)
            
            # Save checkpoint info
            checkpoint_info = {
                "timesteps_completed": self.n_calls,
                "model_path": model_path,
                "run_id": getattr(self.locals.get('run'), 'id', None)
            }
            
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint_info, f)
            
            if self.verbose > 0:
                print(f"Checkpoint saved at step {self.n_calls}: {model_path}")
        
        return True

def find_latest_checkpoint(checkpoint_dir: str = "checkpoints"):
    """Find the latest checkpoint file"""
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None

def find_latest_model(models_dir: str = "models"):
    """Find the latest model file in case checkpoint file is missing"""
    if not os.path.exists(models_dir):
        return None
    
    pattern = os.path.join(models_dir, "sac_checkpoint_*.zip")
    model_files = glob.glob(pattern)
    
    if not model_files:
        return None
    
    # Sort by timestep number (extract from filename)
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_model = model_files[-1]
    
    # Extract timestep from filename
    timestep = int(latest_model.split('_')[-1].split('.')[0])
    
    return {
        "timesteps_completed": timestep,
        "model_path": latest_model,
        "run_id": None
    }

# toggle this to train or evaluate
train = True

if train:
    run = wandb.init(
        project="f1tenth_gym_sac",
        sync_tensorboard=True,
        save_code=True,
    )

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

    # Add action noise for better exploration (optional for SAC)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # SAC hyperparameters optimized for continuous control
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        device="cuda",
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

    model.learn(
        total_timesteps=1_000_000,
        callback=WandbCallback(
            gradient_save_freq=0, model_save_path=f"models/{run.id}", verbose=2
        ),
    )
    run.finish()

else:
    model_path = "models/3wlusg06/model.zip"
    model = SAC.load(model_path, print_system_info=True, device="cpu")
    eval_env = gym.make(
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
        render_mode="human",
    )
    obs, info = eval_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = eval_env.step(action)
        eval_env.render()

        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()
    eval_env.close()
