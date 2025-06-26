import os
import numpy as np
import torch
import random
import glob
from ray.rllib.algorithms.algorithm import Algorithm

# Set environment variable for Python hash seed
os.environ["PYTHONHASHSEED"] = "42"

# Set seeds for determinism
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

# PyTorch deterministic settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from multiagent_ppo import MultiAgentF110, get_env_config, setup_policies_and_config
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec

# Path to the results directory
PATH_RESULTS = os.path.abspath("./ray_results")

# Visualize results
print("Training complete. Results saved to:", PATH_RESULTS)
print("You can visualize the results using TensorBoard or Ray Dashboard.")
print(f"Note: To visualize results, run `tensorboard --logdir={PATH_RESULTS}` in your terminal.")

# Register the environment
register_env("f1tenth_multi", lambda config: MultiAgentF110(config))

# --- 1. Find the latest checkpoint ---
# Construct the path to the experiment directory
exp_dir = os.path.join(PATH_RESULTS, "f1tenth_multiagent_ppo")

# Find the latest trial directory within the experiment
try:
    latest_trial_dir = "/home/sergio/repos/f1_tenth_new_integration/examples/ray_results/f1tenth_multiagent_ppo/PPO_f1tenth_multi_7a5a2_00000_0_2025-06-24_22-04-47" # sorted(glob.glob(os.path.join(exp_dir, "PPO_*")))[-1]
    print(f"Loading results from: {latest_trial_dir}")

    # Find the latest checkpoint in that trial directory
    latest_checkpoint_dir = sorted(glob.glob(os.path.join(latest_trial_dir, "checkpoint_*")))[-1]
    print(f"Loading checkpoint from: {latest_checkpoint_dir}")
except IndexError:
    print(f"Error: No training results found in '{exp_dir}'. Please run the training cell first.")
    # Exit gracefully if no checkpoint is found
    latest_checkpoint_dir = None

if latest_checkpoint_dir:
    # --- 2. Restore the trained algorithm ---
    algo = Algorithm.from_checkpoint(latest_checkpoint_dir)

    # --- 3. Create environment for visualization ---
    env_config = get_env_config()
    env_config["render_mode"] = "human"  # Enable human-readable rendering
    env = MultiAgentF110(env_config)

    # --- 4. Run 5 simulation episodes ---
    for eval_num in range(1, 6):
        print(f"\n=== Starting evaluation {eval_num}/5 ===")
        obs, info = env.reset()
        terminated = {"__all__": False}

        while not terminated["__all__"]:
            actions = {}
            for agent_id, agent_obs in obs.items():
                # Compute actions using the restored policy for each agent
                actions[agent_id] = algo.compute_single_action(
                    observation=agent_obs,
                    policy_id=agent_id,
                    explore=False  # Disable exploration for testing
                )
            # Step the environment with the computed actions
            obs, reward, terminated, truncated, info = env.step(actions)
            # Render the environment to visualize
            env.render()
        print(f"Evaluation {eval_num} finished.")

    # --- 5. Clean up ---
    env.close()
    print("All 5 simulations finished.")