from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
"""
This script provides training and evaluation routines for a Soft Actor-Critic (SAC) agent using the Stable Baselines3 library in a custom F1TENTH Gym environment. It includes support for incremental checkpoint saving, resuming training, logging with Weights & Biases (wandb), and optional video recording during evaluation.

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
import gymnasium.wrappers
from stable_baselines3 import SAC
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
import f1tenth_gym

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
            model_path = os.path.join(
                self.save_path, f"sac_checkpoint_{self.n_calls}.zip")
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

    pattern = os.path.join(models_dir, "sac_checkpoint_*.zip")
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


def main():
    # toggle this to train or evaluate
    # Use the new multi-agent environment
    gym_env = "f1tenth_gym:victor-multi-agent-v0"
    # Extract policy name from environment ID
    policy = gym_env.split(":")[-1] + "31005216_1"
    train = False
    device = "cuda"
    # Feature flags
    record_video = False  # Set to False to disable video recording during evaluation
    # Set to True to evaluate all available checkpoints
    evaluate_all_checkpoints = False

    torch.backends.cudnn.benchmark = True

    def make_env():
        return gym.make(
            gym_env,
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "timestep": 0.01,
                "num_beams": 36,
                "integrator": "rk4",
                "model": "st",
                "control_input": ["speed", "steering_angle"],
                "observation_config": {"type": "rl"},
                "reset_config": {"type": "rl_random_static"},
            },
        )

    num_envs = 16  # Ajusta según tu CPU/GPU
    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # Obtener los entornos subyacentes
    # base_envs = env.get_attr('envs')

    # # Aplicar set_attr a cada entorno
    # for base_env in base_envs:
    #     base_env.set_attr("handle_timeout_termination", False)

    print(f"✅ ENV shape: {env.num_envs}")
    total_timesteps = 3_000_000
    if train:
        # Print system information
        print_system_info()

        # Configuration
        total_timesteps = 3_000_000
        save_freq = 500_000  # Save every 100,000 steps
        models_dir = f"models/{policy}"
        checkpoints_dir = f"checkpoints/{policy}"
        print("Checkpoints directory:", checkpoints_dir)
        print("Models directory:", checkpoints_dir)

        # Check for existing checkpoint
        checkpoint = find_latest_checkpoint(checkpoints_dir)
        if not checkpoint:
            checkpoint = find_latest_model(models_dir)

        # Initialize wandb
        if checkpoint and checkpoint.get("run_id"):
            # Resume existing run
            run = wandb.init(
                project="f1tenth_gym_sac",
                id=checkpoint["run_id"],
                resume="must",
                sync_tensorboard=True,
                save_code=True,
            )
        else:
            # Start new run
            run = wandb.init(
                project="f1tenth_gym_sac",
                sync_tensorboard=True,
                save_code=True,
            )

        # env = make_env()  # Recreate the environment for the new run

        # Add action noise for better exploration (optional for SAC)
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(
            n_actions), sigma=0.1 * np.ones(n_actions))

        # Load existing model or create new one
        if checkpoint:
            print(
                f"Resuming training from checkpoint: {checkpoint['model_path']}")
            print(f"Timesteps completed: {checkpoint['timesteps_completed']}")
            model = SAC.load(
                checkpoint["model_path"],
                env=env,
                device=device,
                print_system_info=True
            )
            # Calculate remaining timesteps
            remaining_timesteps = total_timesteps - \
                checkpoint["timesteps_completed"]
            print(f"Remaining timesteps: {remaining_timesteps}")
        else:
            print("Starting new training from scratch")
            # model = SAC(
            #     # Tipo de política (red neuronal multilayer perceptron)
            #     "MlpPolicy",
            #     env,  # Entorno Gym
            #     # Nivel de verbosidad (1 = información, 0 = silencioso)
            #     verbose=1,
            #     # Carpeta para logs de TensorBoard
            #     tensorboard_log=f"runs/{run.id}",
            #     # Dispositivo para entrenamiento ('cuda' o 'cpu')
            #     device=device,
            #     seed=42,  # Semilla para reproducibilidad
            #     action_noise=action_noise,  # Ruido de acción para exploración
            #     learning_rate=3e-4,  # Tasa de aprendizaje del optimizador
            #     buffer_size=1_000_000,  # Tamaño del buffer de replay
            #     learning_starts=10000,  # Número de pasos antes de comenzar a aprender
            #     batch_size=4096*2,  # Tamaño del batch para entrenamiento
            #     tau=0.005,  # Factor de suavizado para actualización de la red objetivo
            #     gamma=0.99,  # Factor de descuento para recompensas futuras
            #     # Frecuencia de entrenamiento (cada paso)
            #     train_freq=(1, "step"),
            #     gradient_steps=4,  # Número de pasos de gradiente por actualización
            #     # Coeficiente de entropía ('auto' para ajuste automático)
            #     ent_coef='auto',
            #     target_update_interval=1,  # Frecuencia de actualización de la red objetivo
            #     # Optimiza el uso de memoria (puede ralentizar)
            #     optimize_memory_usage=False,
            #     # Arquitectura de la red neuronal
            #     policy_kwargs=dict(net_arch=[512, 256, 128])
            # )

            # model = SAC(
            #     "MlpPolicy",
            #     env,
            #     verbose=1,
            #     tensorboard_log=f"runs/{run.id}"
            # )

            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=f"runs/{run.id}",
                device="cuda",
                seed=42,
                # action_noise=action_noise,
                learning_rate=5e-4,            # Subimos un poquito la LR para acelerar early learning
                buffer_size=1_000_000,           # Más pequeño = más rápido, menos memoria
                learning_starts=10_000,          # Empieza a aprender rápido
                batch_size=512,                # Ideal para 3050
                tau=0.005,
                gamma=0.98,                    # Reduce un poco el horizonte
                train_freq=1,
                gradient_steps=1,              # Clave para velocidad: menos updates por step
                ent_coef='auto',
                target_update_interval=1,
                # Ahorro de RAM en GPU; parece que despues de cierta cantidad de pasos
                # no es compatible en true
                optimize_memory_usage=True,
                # Red compacta pero capaz
                policy_kwargs=dict(net_arch=[1024, 512, 256]),
                replay_buffer_kwargs=dict(
                    handle_timeout_termination=False)  # <--- CAMBIO AQUÍ


            )
            print(f"✅ Model sees {model.n_envs} envs")
            input("Waiting for user input to continue...")
            remaining_timesteps = total_timesteps

        # Setup callbacks
        checkpoint_path = os.path.join(
            checkpoints_dir, "latest_checkpoint.json")
        incremental_save_callback = IncrementalSaveCallback(
            save_freq=save_freq,
            save_path=models_dir,
            checkpoint_path=checkpoint_path,
            verbose=1
        )

        wandb_callback = WandbCallback(
            gradient_save_freq=0,
            model_save_path=f"models/{policy}/{run.id}",
            verbose=2
        )

        # Combine callbacks
        callbacks = [incremental_save_callback, wandb_callback]

        # Train the model
        if remaining_timesteps > 0:
            print(
                f"\n🚀 Starting training for {remaining_timesteps:,} timesteps on {device.upper()}...")
            start_time = time.time()  # Start timing
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                reset_num_timesteps=False,  # Don't reset timestep counter when resuming
                progress_bar=True,
            )
            end_time = time.time()  # End timing

            # Calculate and log training speed
            elapsed_time = end_time - start_time
            timesteps_per_second = remaining_timesteps / elapsed_time

            print("\n✅ Training completed!")
            print(
                f"⏱️  Total training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
            print(
                f"🏃 Training throughput: {timesteps_per_second:.2f} timesteps/second")
            print(
                f"📊 Average time per timestep: {elapsed_time/remaining_timesteps*1000:.2f} ms")
            print(f"🎯 Device used: {device.upper()}")

        # Save final model
        final_model_path = os.path.join(models_dir, f"sac_final_{run.id}.zip")
        model.save(final_model_path)
        print(f"Final model saved: {final_model_path}")

        run.finish()

    else:
        # Determine which models to evaluate
        if evaluate_all_checkpoints:
            models_to_evaluate = find_all_models("models")
            if not models_to_evaluate:
                print("No checkpoint models found in 'models' directory")
                exit(1)
            print(f"Found {len(models_to_evaluate)} checkpoints to evaluate")
        else:
            # Single model evaluation (original behavior)
            model_path = "models/victor-multi-agent-v031005216_1/ud9f4w6c/model"
            models_to_evaluate = [{
                "timesteps_completed": total_timesteps,
                "model_path": model_path,
                "run_id": None
            }]

        # Create videos directory if video recording is enabled
        if record_video:
            videos_dir = f"videos/{policy}"
            os.makedirs(videos_dir, exist_ok=True)

        # Evaluate each model
        for i, checkpoint in enumerate(models_to_evaluate):
            model_path = checkpoint["model_path"]
            timesteps = checkpoint["timesteps_completed"]

            print(f"\n{'='*60}")
            print(f"Evaluating checkpoint {i+1}/{len(models_to_evaluate)}")
            print(f"Model: {model_path}")
            print(f"Timesteps: {timesteps:,}")
            print(f"{'='*60}")

            # Load the model
            try:
                model = SAC.load(
                    model_path, print_system_info=(i == 0), device=device)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                continue

            eval_env = gym.make(
                gym_env,
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
                render_mode="rgb_array" if record_video else "human",
            )

            # Wrap environment for video recording only if feature flag is enabled
            if record_video:
                video_filename = f"sac_checkpoint_{gym_env}_{timesteps}_{int(time.time())}"
                eval_env = gymnasium.wrappers.RecordVideo(
                    eval_env,
                    video_folder=videos_dir,
                    name_prefix=video_filename,
                    episode_trigger=lambda x: True  # Record all episodes
                )
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

                # Optional: limit episode length to avoid very long videos/sessions
                if step_count > 5000:  # Adjust this limit as needed
                    done = True

                # Render only if not recording video (to avoid double rendering)
                if not record_video:
                    eval_env.render()

            print(f"Checkpoint {timesteps:,} results:")
            print(f"  - Total steps: {step_count}")
            print(f"  - Total reward: {total_reward:.2f}")
            print(
                f"  - Average reward per step: {total_reward/step_count:.4f}")

            eval_env.close()

            if record_video:
                print(f"  - Video saved: {video_filename}")

        print(f"\n{'='*60}")
        if evaluate_all_checkpoints:
            print(
                f"Completed evaluation of {len(models_to_evaluate)} checkpoints")
            if record_video:
                print(f"All videos saved in {videos_dir} directory")
        else:
            print("Evaluation completed!")
            if record_video:
                print(f"Video saved in {videos_dir} directory")
            else:
                print("No video was recorded (record_video=False)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
