from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
import wandb
import numpy as np
import os
import json
import glob
import time
import torch
import gymnasium.wrappers
import f1tenth_gym

"""
This script provides training and evaluation routines for Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO), and Deep Deterministic Policy Gradient (DDPG) agents using the Stable Baselines3 library in a custom F1TENTH Gym environment. It includes support for incremental checkpoint saving, resuming training, logging with Weights & Biases (wandb), and optional video recording during evaluation.

Classes:
---------
IncrementalSaveCallback(BaseCallback):
    Callback for periodically saving model checkpoints and training progress.

    Parameters:
        save_freq (int): Frecuencia (en pasos de entrenamiento) con la que se guarda el modelo.
        save_path (str): Ruta del directorio donde se guardarán los modelos.
        checkpoint_path (str): Ruta del archivo JSON donde se almacena la información del checkpoint.
        verbose (int): Nivel de verbosidad para la impresión de mensajes (0 = silencioso).

Functions:
----------
find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> dict or None:
    Busca el archivo de checkpoint más reciente en el directorio especificado.

    Parámetros:
        checkpoint_dir (str): Directorio donde buscar el archivo de checkpoint.

    Retorna:
        dict: Información del checkpoint más reciente, o None si no existe.

find_all_models(models_dir: str = "models") -> list:
    Encuentra todos los archivos de modelo guardados en el directorio especificado y los ordena por número de timesteps.

    Parámetros:
        models_dir (str): Directorio donde buscar los modelos.

    Retorna:
        list: Lista de diccionarios con información de cada checkpoint encontrado.

find_latest_model(models_dir: str = "models") -> dict or None:
    Encuentra el archivo de modelo más reciente en el directorio especificado.

    Parámetros:
        models_dir (str): Directorio donde buscar los modelos.

    Retorna:
        dict: Información del modelo más reciente, o None si no existe.

print_system_info():
    Imprime información del sistema relevante para el entrenamiento, como GPU, memoria, núcleos de CPU y versión de Python.

Variables principales:
----------------------
gym_env (str): Nombre del entorno Gym a utilizar.
policy (str): Nombre de la política, extraído del ID del entorno.
train (bool): Si es True, ejecuta el entrenamiento; si es False, ejecuta la evaluación.
device (str): Dispositivo a utilizar para el entrenamiento ('cuda' o 'cpu').
record_video (bool): Si es True, graba videos durante la evaluación.
evaluate_all_checkpoints (bool): Si es True, evalúa todos los checkpoints encontrados.

Entrenamiento:
--------------
- Inicializa el entorno y el modelo SAC.
- Permite reanudar desde el último checkpoint.
- Guarda checkpoints periódicamente y registra métricas en wandb.
- Al finalizar, guarda el modelo final.

Evaluación:
-----------
- Permite evaluar un solo modelo o todos los checkpoints encontrados.
- Opción de grabar videos de las evaluaciones.
- Imprime resultados de cada evaluación (recompensa total, pasos, etc.).

"""

# Custom callback for incremental saving

import logging
import shutil
import wandb
from pathlib import Path
from wandb.integration.sb3 import WandbCallback


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO to see the logs in the console


class CustomWandbCallback(WandbCallback):
    def save_model(self):
        self.model.save(self.path)

        try:
            wandb.save(self.path, base_path=self.model_save_path)
            if self.verbose > 1:
                logger.info(f"🔗 Model symlinked via wandb.save to {self.path}")
        except Exception as e:
            # Fallback: copia el archivo manualmente si falla wandb.save
            try:
                model_file = Path(self.path)
                target_path = Path(wandb.run.dir) / model_file.name
                shutil.copy(str(model_file), str(target_path))
                if self.verbose > 0:
                    logger.info(
                        f"🧯 wandb.save failed, model copied to {target_path} instead.")
            except Exception as copy_error:
                logger.error(f"❌ Failed to save model: {copy_error}")


class IncrementalSaveCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, checkpoint_path: str, model_name: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name

    def _init_callback(self) -> None:
        # Create save directory if it doesn't exist
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save model
            model_path = os.path.join(
                self.save_path, f"{self.model_name}_checkpoint_{self.n_calls}.zip")
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


def find_all_models(models_dir: str = "models"):
    """Find all available model checkpoints sorted by timesteps"""
    if not os.path.exists(models_dir):
        return []

    pattern = os.path.join(models_dir, "*checkpoint_*.zip")
    model_files = glob.glob(pattern)

    print(model_files)
    if not model_files:
        return []

    # Sort by timestep number (extract from filename)
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    checkpoints = []
    for model_file in model_files:
        timestep = int(model_file.split('_')[-1].split('.')[0])
        checkpoints.append({
            "timesteps_completed": timestep,
            "model_path": model_file,
            "run_id": None
        })

    return checkpoints


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


def print_system_info(device=None):
    """Print system information for performance comparison"""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)

    # Check if GPU is available
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(
                0).total_memory / 1024**3
            print(f"🚀 GPU: {gpu_name}")
            print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
            print(f"🔧 CUDA Version: {torch.version.cuda}")
        else:
            print("❌ No GPU available or CUDA not installed")
    except ImportError:
        print("❌ PyTorch not available")

    print(f"🖥️  CPU Cores: {os.cpu_count()}")
    if device is not None:
        print(f"🐍 Python: Using device '{device}'")
    print("="*60 + "\n")


def make_env(gym_env, config=None):
    def _init():
        try:
            env = gym.make(gym_env, config=config)
            return env
        except Exception as e:
            import traceback
            print(f"\n\n[ERROR dentro de make_env - gym_env={gym_env}]\n")
            traceback.print_exc()
            raise e  # re-raise para que SubprocVecEnv también falle
    return _init


def setup_wandb(run_id=None, project="f1tenth_gym_multi_algo"):
    if run_id:
        run = wandb.init(
            project=project,
            id=run_id,
            resume="must",
            sync_tensorboard=True,
            save_code=True,
        )
    else:
        run = wandb.init(
            project=project,
            sync_tensorboard=True,
            save_code=True,
        )
    return run


def setup_tensorboard(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_video_recording(eval_env, videos_dir, video_filename):
    os.makedirs(videos_dir, exist_ok=True)
    return gymnasium.wrappers.RecordVideo(
        eval_env,
        video_folder=videos_dir,
        name_prefix=video_filename,
        episode_trigger=lambda x: True
    )


def train(model_name, model_cfg, env, total_timesteps, save_freq, models_dir, checkpoints_dir, run, device):
    checkpoint_path = os.path.join(checkpoints_dir, "latest_checkpoint.json")
    incremental_save_callback = IncrementalSaveCallback(
        save_freq=save_freq,
        save_path=models_dir,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        verbose=1
    )
    wandb_callback = CustomWandbCallback(
        gradient_save_freq=0,
        model_save_path=f"{models_dir}/{run.id}",
        verbose=2,
    )
    callbacks = [incremental_save_callback, wandb_callback]
    print(
        f"\n🚀 Starting training for {total_timesteps:,} timesteps on {device.upper()} with {model_name}...")
    start_time = time.time()
    model = model_cfg['constructor'](
        **model_cfg['params'],
        env=env, # parque
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        device=device
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=False,
        progress_bar=True,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n✅ Training completed for {model_name}!")
    print(
        f"⏱️  Total training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    final_model_path = os.path.join(
        models_dir, f"{model_name.lower()}_final_{run.id}.zip")
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")
    run.finish()
    return model


def evaluate(model_name, model_cfg, model_path, gym_env, env_config, device, record_video=False, videos_dir=None):
    print(f"\n{'='*60}\nEvaluating {model_name} model: {model_path}\n{'='*60}")
    try:
        model = model_cfg['constructor'].load(model_path, device=device)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return
    eval_env = gym.make(
        gym_env,
        config=env_config,
        render_mode="rgb_array" if record_video else "human",
    )
    if record_video and videos_dir:
        import re
        result_re = re.search(r'_checkpoint_(\d+)\.zip$', model_path)
        if result_re:
            checkpoint_number = result_re.group(1)
        else:
            checkpoint_number = "final"
        video_filename = f"{model_name}_eval_checkpoint_{checkpoint_number}_{int(time.time())}"
        eval_env = setup_video_recording(eval_env, videos_dir, video_filename)
        print(f"Recording video: {video_filename}")
    obs, info = eval_env.reset()
    done = False
    step_count = 0
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = eval_env.step(action)
        total_reward += reward
        step_count += 1
        if step_count > 5000:
            done = True
        if not record_video:
            eval_env.render()
    print(f"Results for {model_name}:")
    print(f"  - Total steps: {step_count}")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Average reward per step: {total_reward/step_count:.4f}")
    eval_env.close()
    if record_video:
        print(f"  - Video saved in {videos_dir}")


def main():
    train_mode = True  # Cambia a True para entrenar
    record_video = False
    evaluate_all_checkpoints = False  # Cambia a False para evaluar solo el último modelo
    gym_env = "f1tenth_gym:victor-multi-agent-v0"
    env_config = {
        "map": "Spielberg",
        "num_agents": 2,
        "timestep": 0.01,
        "num_beams": 36,
        "integrator": "rk4",
        "model": "st",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "multi_rl"},
        "reset_config": {"type": "rl_random_static"},
    }
    multithreading = False  # Cambia a True para usar SubprocVecEnv
    policy = gym_env.split(":")[-1] + "02_06_25_1_10K"
    # MlpPolicy para single-agent, `MultiInputPolicy` para multi-agent
    model_policy = "MultiInputPolicy"
    optimize_memory_usage = False  # No se puede optimizar memoria con `MultiInputPolicy`
    device = "cuda"
    total_timesteps = 10_000  # Total timesteps para entrenamiento
    save_freq = 1_000
    learning_rate = 5e-4  # Mismo learning rate para todos los modelos
    num_envs = 15
    buffer_size = 100_000  # Tamaño del buffer de replay para DDPG y SAC
    learning_starts = 10_000  # Mismo learning starts para todos los modelos
    batch_size = 512  # Tamaño del batch para PPO y SAC
    tau = 0.005  # Mismo tau para todos los modelos
    gamma = 0.98  # Mismo gamma para todos los modelos
    models_dir = f"models/{policy}"
    checkpoints_dir = f"checkpoints/{policy}"
    videos_dir = f"videos/{policy}"
    torch.backends.cudnn.benchmark = True

    # Diccionario de modelos a entrenar
    models_to_train = {
        "SAC": {
            "constructor": SAC,
            "params": {
                "policy": model_policy,
                "learning_rate": learning_rate,
                "buffer_size": buffer_size,
                "learning_starts": learning_starts,
                "batch_size": batch_size,
                "tau": tau,
                "gamma": gamma,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": 'auto',
                "target_update_interval": 1,
                "optimize_memory_usage": optimize_memory_usage,
                "policy_kwargs": dict(net_arch=[1024, 512, 256]),
                "replay_buffer_kwargs": dict(handle_timeout_termination=False)
            }
        },
        "PPO": {
            "constructor": PPO,
            "params": {
                "policy": model_policy,
                "learning_rate": learning_rate,                      # Igual que SAC
                "n_steps": 2048,                           # Queda como default razonable
                "batch_size": batch_size,                         # Igual que SAC
                "n_epochs": 10,
                "gamma": gamma,                             # Igual que SAC
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                # Para mantener equivalencia con SAC
                "ent_coef": - 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "policy_kwargs": dict(net_arch=[1024, 512, 256])
            }
        },
        "DDPG": {
            "constructor": DDPG,
            "params": {
                "policy": model_policy,
                "learning_rate": learning_rate,                     # Igual que SAC
                "buffer_size": buffer_size,                  # Igual que SAC
                "learning_starts": learning_starts,                 # Igual que SAC
                "batch_size": batch_size,                         # Igual que SAC
                "tau": tau,                              # Igual que SAC
                "gamma": gamma,                             # Igual que SAC
                "policy_kwargs": dict(net_arch=[1024, 512, 256]),
                "replay_buffer_kwargs": dict(handle_timeout_termination=False)
            }
        }
    }
    if multithreading:
        env_fns = [make_env(gym_env, config=env_config) for _ in range(num_envs)]
        env = SubprocVecEnv(env_fns)
    else:
        env = make_env(gym_env, config=env_config)()

    print("Train mode:", train_mode)
    if train_mode:
        print("*"*60)
        print("Entrenando el modelo")
        print("*"*60)
        for model_name, model_cfg in models_to_train.items():
            print(f"\n{'='*60}\nTraining {model_name}\n{'='*60}")
            run = setup_wandb()
            train(
                model_name,
                model_cfg,
                env,
                total_timesteps,
                save_freq,
                models_dir,
                checkpoints_dir,
                run,
                device
            )
    else:
        # Evaluar el modelo final de cada algoritmo
        for model_name, model_cfg in models_to_train.items():
            model_path = os.path.join(
                models_dir, f"{model_name.lower()}_final_*")
            # Buscar el modelo final más reciente
            model_files = glob.glob(model_path)
            if not model_files:
                print(f"No final model found for {model_name}")
                continue
            latest_model = sorted(model_files)[-1]
            if evaluate_all_checkpoints:
                models_to_evaluate = find_all_models(models_dir)
                for i, checkpoint in enumerate(models_to_evaluate):
                    model_path = checkpoint["model_path"]
                    timesteps = checkpoint["timesteps_completed"]

                    print(f"\n{'='*60}")
                    print(f"Evaluating checkpoint {i+1}/{len(models_to_evaluate)}")
                    print(f"Model: {model_path}")
                    print(f"Timesteps: {timesteps:,}")
                    print(f"{'='*60}")
                    evaluate(
                        model_name,
                        model_cfg,
                        model_path,
                        gym_env,
                        env_config,
                        device,
                        record_video=record_video,
                        videos_dir=videos_dir if record_video else None
                    )
            if not models_to_evaluate:
                print("No checkpoint models found in 'models' directory")
                exit(1)
            print(f"Found {len(models_to_evaluate)} checkpoints to evaluate")
            evaluate(
                model_name,
                model_cfg,
                latest_model,
                gym_env,
                env_config,
                device,
                record_video=record_video,
                videos_dir=videos_dir if record_video else None
            )


if __name__ == "__main__":
    main()
