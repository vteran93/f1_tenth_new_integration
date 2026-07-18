import gymnasium as gym
import numpy as np
import time
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
import matplotlib.pyplot as plt
import os


def benchmark_device(device_name, num_timesteps=10000):
    """
    Benchmark training on a specific device (CPU or GPU)
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking training on {device_name.upper()}")
    print(f"{'='*60}")

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

    # Action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,  # Disable verbose output for cleaner benchmark
        device=device_name,
        seed=42,  # Same seed for fair comparison
        action_noise=action_noise,
        learning_rate=3e-4,
        buffer_size=100_000,  # Smaller buffer for faster benchmark
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

    # Warmup - let the model do some initial steps
    print("Warming up...")
    model.learn(total_timesteps=1000, progress_bar=False)

    # Benchmark training
    print(f"Training {num_timesteps} timesteps on {device_name.upper()}...")
    start_time = time.time()

    model.learn(
        total_timesteps=num_timesteps,
        reset_num_timesteps=False,
        progress_bar=True
    )

    end_time = time.time()
    training_time = end_time - start_time

    # Calculate throughput
    throughput = num_timesteps / training_time

    print(f"\nResults for {device_name.upper()}:")
    print(f"  Total training time: {training_time:.2f} seconds")
    print(f"  Throughput: {throughput:.2f} timesteps/second")

    env.close()

    return {
        'device': device_name,
        'training_time': training_time,
        'throughput': throughput,
        'timesteps': num_timesteps
    }


def check_gpu_availability():
    """Check if CUDA GPU is available"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU detected: {gpu_name}")
        print(f"GPU memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("No GPU detected or CUDA not available")
        return False


def plot_comparison(results):
    """Plot benchmark results"""
    devices = [r['device'].upper() for r in results]
    times = [r['training_time'] for r in results]
    throughputs = [r['throughput'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training time comparison
    bars1 = ax1.bar(devices, times, color=['blue', 'orange'])
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.set_ylim(0, max(times) * 1.1)

    # Add value labels on bars
    for bar, time_val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                 f'{time_val:.1f}s', ha='center', va='bottom')

    # Throughput comparison
    bars2 = ax2.bar(devices, throughputs, color=['blue', 'orange'])
    ax2.set_ylabel('Throughput (timesteps/second)')
    ax2.set_title('Throughput Comparison')
    ax2.set_ylim(0, max(throughputs) * 1.1)

    # Add value labels on bars
    for bar, throughput_val in zip(bars2, throughputs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(throughputs)*0.01,
                 f'{throughput_val:.1f}', ha='center', va='bottom')

    plt.tight_layout()

    # Save plot
    plot_path = "gpu_vs_cpu_benchmark.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nBenchmark plot saved as: {plot_path}")
    plt.show()


def main():
    print("GPU vs CPU Training Benchmark for F1Tenth SAC")
    print("=" * 60)

    # Check system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU cores: {os.cpu_count()}")

    gpu_available = check_gpu_availability()

    # Benchmark parameters
    benchmark_timesteps = 20000  # Adjust this based on your patience level

    results = []

    # Always benchmark CPU
    print("\nStarting CPU benchmark...")
    cpu_result = benchmark_device("cpu", benchmark_timesteps)
    results.append(cpu_result)

    # Benchmark GPU if available
    if gpu_available:
        print("\nStarting GPU benchmark...")
        gpu_result = benchmark_device("cuda", benchmark_timesteps)
        results.append(gpu_result)

        # Calculate speedup
        speedup = gpu_result['throughput'] / cpu_result['throughput']
        time_reduction = (1 - gpu_result['training_time'] / cpu_result['training_time']) * 100

        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"CPU Training Time: {cpu_result['training_time']:.2f} seconds")
        print(f"GPU Training Time: {gpu_result['training_time']:.2f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Time reduction: {time_reduction:.1f}%")

        if speedup > 1:
            print(f"\n🚀 GPU is {speedup:.2f}x FASTER than CPU!")
        else:
            print(f"\n⚠️  CPU is {1/speedup:.2f}x faster than GPU for this workload")
            print("This might happen with small models or when GPU overhead exceeds benefits")

        # Plot results
        plot_comparison(results)

    else:
        print("\n❌ GPU not available. Only CPU benchmark completed.")
        print(f"CPU training time: {cpu_result['training_time']:.2f} seconds")
        print(f"CPU throughput: {cpu_result['throughput']:.2f} timesteps/second")


if __name__ == "__main__":
    main()
