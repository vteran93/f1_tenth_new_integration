# Script para encontrar la mejor configuración de it/s (iteraciones por segundo)
# para entrenamientos cortos en el entorno multi-agent de F1TENTH Gym

import time
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

# Configuración base del entorno y modelo
GYM_ENV = "f1tenth_gym:victor-multi-agent-v0"
ENV_CONFIG = {
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
MODEL_POLICY = "MultiInputPolicy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Rango de configuraciones a probar
envs_to_test = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
batch_sizes_to_test = [256, 512, 1024, 2048]
TOTAL_TIMESTEPS = 10_000
WARMUP_STEPS = 1_000

results = []

print(
    f"Probando combinaciones de num_envs y batch_size para {TOTAL_TIMESTEPS} pasos...")

for num_envs in envs_to_test:
    for batch_size in batch_sizes_to_test:
        print(f"\nTest: num_envs={num_envs}, batch_size={batch_size}")
        env_fns = [lambda: gym.make(GYM_ENV, config=ENV_CONFIG)
                   for _ in range(num_envs)]
        env = SubprocVecEnv(env_fns)
        model = SAC(
            policy=MODEL_POLICY,
            env=env,
            learning_rate=5e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=batch_size,
            tau=0.005,
            gamma=0.98,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            target_update_interval=1,
            optimize_memory_usage=False,
            policy_kwargs=dict(net_arch=[256, 128]),
            replay_buffer_kwargs=dict(handle_timeout_termination=False),
            verbose=0,
            device=DEVICE
        )
        start = time.time()
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            reset_num_timesteps=True,
            progress_bar=False
        )
        end = time.time()
        elapsed = end - start
        # it/s: pasos útiles (después de warmup) / tiempo total
        steps_measured = TOTAL_TIMESTEPS - WARMUP_STEPS
        its = steps_measured / elapsed
        print(f"  Tiempo total: {elapsed:.2f}s | it/s estimado: {its:.2f}")
        results.append({
            "num_envs": num_envs,
            "batch_size": batch_size,
            "its": its,
            "elapsed": elapsed
        })
        env.close()
        del model
        torch.cuda.empty_cache()


def get_best(results):
    best = max(results, key=lambda x: x["its"])
    return best


best = get_best(results)
print("\n==============================")
print("Mejor configuración encontrada:")
print(f"  num_envs   = {best['num_envs']}")
print(f"  batch_size = {best['batch_size']}")
print(f"  it/s       = {best['its']:.2f}")
print(f"  tiempo     = {best['elapsed']:.2f}s para {TOTAL_TIMESTEPS} pasos")
print("==============================")

# Criterio de parada: si la mejora es menor al 2% en dos pruebas consecutivas,
# se puede detener el script automáticamente (puedes implementar esto si lo deseas)
