import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
import gymnasium as gym

def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Configure SAC algorithm
    config = (
        SACConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=2,
            num_envs_per_env_runner=1,
        )
        .training(
            # SAC specific parameters
            train_batch_size=256,
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 50000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
            },
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            target_network_update_freq=1,
            tau=0.005,
        )
        .debugging(seed=42)
    )
    
    # Run training with Ray Tune
    tune.run(
        "SAC",
        config=config.to_dict(),
        stop={
            "timesteps_total": 50000,  # Stop after 50k timesteps
            "episode_reward_mean": 400,  # Stop if average reward reaches 400
        },
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_score_attribute="episode_reward_mean",
            checkpoint_score_order="max",
            num_to_keep=2,
            checkpoint_frequency=10
        ),
        storage_path="/tmp/ray_results",
        name="sac_cartpole_debug",
        verbose=1,
    )
    
    ray.shutdown()

if __name__ == "__main__":
    main()