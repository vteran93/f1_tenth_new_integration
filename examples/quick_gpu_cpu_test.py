#!/usr/bin/env python3
"""
Quick test to compare GPU vs CPU training speed
This script runs a short training session on both devices and compares the results
"""

import gymnasium as gym
import numpy as np
import time
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise


def check_system_info():
    """Display system information"""
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)

    print(f"🖥️  CPU Cores: {torch.get_num_threads()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU: {gpu_name}")
        print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("❌ No GPU available")
        return False


def run_training_test(device, test_timesteps=5000):
    """Run a quick training test on specified device"""
    print(f"\n{'='*50}")
    print(f"Testing {device.upper()}")
    print(f"{'='*50}")

    # Create environment
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

    # Action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Create SAC model
    print(f"Creating SAC model on {device}...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        device=device,
        seed=42,
        action_noise=action_noise,
        learning_rate=3e-4,
        buffer_size=50_000,  # Smaller for quick test
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1,
        policy_kwargs=dict(net_arch=[256, 256])
    )

    print(f"Training {test_timesteps:,} timesteps...")
    start_time = time.time()

    model.learn(
        total_timesteps=test_timesteps,
        progress_bar=True
    )

    end_time = time.time()
    training_time = end_time - start_time
    throughput = test_timesteps / training_time

    print(f"\n📊 Results for {device.upper()}:")
    print(f"  ⏱️  Training time: {training_time:.2f} seconds")
    print(f"  🏃 Throughput: {throughput:.2f} timesteps/second")
    print(f"  📈 Time per timestep: {training_time/test_timesteps*1000:.2f} ms")

    env.close()
    return {
        'device': device,
        'time': training_time,
        'throughput': throughput,
        'timesteps': test_timesteps
    }


def main():
    print("🏎️  F1Tenth SAC GPU vs CPU Speed Test")

    # Check system
    gpu_available = check_system_info()

    # Test parameters
    test_timesteps = 10000  # Quick test - adjust as needed

    results = []

    # Test CPU
    print("\n🔄 Testing CPU performance...")
    cpu_result = run_training_test("cpu", test_timesteps)
    results.append(cpu_result)

    # Test GPU if available
    if gpu_available:
        print("\n🔄 Testing GPU performance...")
        gpu_result = run_training_test("cuda", test_timesteps)
        results.append(gpu_result)

        # Compare results
        speedup = gpu_result['throughput'] / cpu_result['throughput']
        time_saved = cpu_result['time'] - gpu_result['time']
        time_saved_percent = (time_saved / cpu_result['time']) * 100

        print(f"\n{'='*60}")
        print("🏆 PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"CPU Time:    {cpu_result['time']:.2f} seconds")
        print(f"GPU Time:    {gpu_result['time']:.2f} seconds")
        print(f"Time Saved:  {time_saved:.2f} seconds ({time_saved_percent:.1f}%)")
        print(f"Speedup:     {speedup:.2f}x")

        if speedup > 1.1:
            print(f"\n🚀 GPU is {speedup:.2f}x FASTER! Use device='cuda' for training")
        elif speedup < 0.9:
            print(f"\n🐌 CPU is {1/speedup:.2f}x faster. Use device='cpu' for training")
        else:
            print(f"\n⚖️  Performance is similar. Either device is fine.")

        print(f"\n💡 Recommendation:")
        if speedup > 1.2:
            print(f"   Use GPU (device='cuda') for best performance")
        elif speedup < 0.8:
            print(f"   Use CPU (device='cpu') for best performance")
        else:
            print(f"   Either device works well for this model size")

    else:
        print(f"\n📊 CPU-only results:")
        print(f"  Training time: {cpu_result['time']:.2f} seconds")
        print(f"  Throughput: {cpu_result['throughput']:.2f} timesteps/second")
        print(f"\n💡 To enable GPU comparison, install CUDA-enabled PyTorch")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
