
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig

def get_ppo_config(config):
    return (
        PPOConfig()
        .environment(env=config["environment"]["env_name"])
        .framework(config["framework"])
        .resources(num_gpus=config["resources"]["num_gpus"])
        .env_runners(num_env_runners=config["env_runners"]["num_env_runners"])
        .training(
            lr=config["algorithm"]["lr"],
            gamma=config["algorithm"]["gamma"],
            lambda_=config["algorithm"]["lambda_"],
            clip_param=config["algorithm"]["clip_param"],
            train_batch_size=config["algorithm"]["train_batch_size"],
        )
    )

def get_sac_config(config):
    return (
        SACConfig()
        .environment(env=config["environment"]["env_name"])
        .framework(config["framework"])
        .resources(num_gpus=config["resources"]["num_gpus"])
        .env_runners(num_env_runners=config["env_runners"]["num_env_runners"])
        .training(
            lr=config["algorithm"]["lr"],
            gamma=config["algorithm"]["gamma"],
            tau=config["algorithm"]["tau"],
            target_network_update_freq=config["algorithm"]["target_network_update_freq"],
            train_batch_size=config["algorithm"]["train_batch_size"],
            replay_buffer_config=config["algorithm"]["replay_buffer_config"],
        )
    )
