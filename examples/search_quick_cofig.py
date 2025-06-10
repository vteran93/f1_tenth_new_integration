import itertools
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
import gymnasium as gym
import time
import platform
import os

# Config
ENV_NAME = "Pendulum-v1"
TRIAL_TIMESTEPS = 20000
MAX_TIME_PER_TRIAL = 60  # seconds

# Param grids
batch_sizes = [256, 512, 1024, 2048]
net_archs = [[256, 256], [512, 512]]
gradient_steps_options = [1, 2, 4]

# Detect device
if torch.cuda.is_available():
    device = "cuda"
elif platform.machine() == "arm64" and "Apple" in platform.platform():
    device = "cpu"
    print("🍏 Detected Apple Silicon. Falling back to CPU mode.")
else:
    device = "cpu"

print(f"🔧 Using device: {device}")

# Result log
results = []


def train_and_benchmark(batch_size, net_arch, gradient_steps):
    print("\n==============================")
    print(f"Testing batch_size={batch_size}, net_arch={net_arch}, gradient_steps={gradient_steps}")

    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: gym.make(ENV_NAME)])

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=torch.zeros(n_actions), sigma=0.1 * torch.ones(n_actions))

    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        device=device,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=batch_size,
        gradient_steps=gradient_steps,
        train_freq=1,
        learning_starts=1000,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=net_arch),
    )

    print(f"🔍 Model device: {model.device}, CUDA Available: {torch.cuda.is_available()}")
    if device == "cuda":
        print(f"🔍 GPU mem alloc before: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    start_time = time.time()
    try:
        model.learn(total_timesteps=TRIAL_TIMESTEPS, progress_bar=False)
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    elapsed = time.time() - start_time

    if elapsed > MAX_TIME_PER_TRIAL:
        print(f"⚠️ Trial exceeded time limit ({elapsed:.2f}s > {MAX_TIME_PER_TRIAL}s). Skipping.")
        return

    mem_alloc = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
    mem_reserved = torch.cuda.memory_reserved() / 1024**2 if device == "cuda" else 0

    print(f"✅ Time: {elapsed:.2f}s | GPU Mem Alloc: {mem_alloc:.1f} MB | Reserved: {mem_reserved:.1f} MB")
    results.append({
        "batch_size": batch_size,
        "net_arch": net_arch,
        "gradient_steps": gradient_steps,
        "time_sec": elapsed,
        "gpu_mem_alloc_MB": mem_alloc,
        "gpu_mem_reserved_MB": mem_reserved
    })


# Run all combinations
for batch_size, net_arch, gradient_steps in itertools.product(batch_sizes, net_archs, gradient_steps_options):
    train_and_benchmark(batch_size, net_arch, gradient_steps)

# Sort by speed
results.sort(key=lambda x: x["time_sec"])

print("\n===== BENCHMARK RESULTS (FASTEST FIRST) =====")
for r in results:
    print(f"⏱️ {r['time_sec']:.2f}s | Batch={r['batch_size']} | Arch={r['net_arch']} | GS={r['gradient_steps']} | Alloc={r['gpu_mem_alloc_MB']:.1f} MB")
