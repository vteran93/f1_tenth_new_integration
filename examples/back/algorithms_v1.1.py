
from typing import Dict
import ray
from ray.rllib.algorithms import ppo, sac
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import torch
import numpy as np

def setup_ppo(config: Dict, env: MultiAgentEnv, agent_id: str) -> ppo.PPO:
    ppo_config = ppo.PPOConfig().environment(
        env="f110_multi",
        env_config=config["environment"]
    ).rollouts(
        num_rollout_workers=2
    ).resources(
        num_gpus=0
    ).framework(
        framework="torch"
    ).training(
        lr=config["algorithm"]["agents"][agent_id]["lr"],
        gamma=config["algorithm"]["agents"][agent_id]["gamma"],
        lambda_=config["algorithm"]["agents"][agent_id]["lambda"],
        clip_param=config["algorithm"]["agents"][agent_id]["clip_param"],
        num_sgd_iter=config["algorithm"]["agents"][agent_id]["num_sgd_iter"],
        sgd_minibatch_size=config["algorithm"]["agents"][agent_id]["sgd_minibatch_size"],
        train_batch_size=config["algorithm"]["agents"][agent_id]["train_batch_size"],
    ).multi_agent(
        policies={
            agent_id: (None, env.observation_space, env.action_space, {})
            for agent_id in env.agents
        },
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id
    )
    return ppo_config.build()

def setup_sac(config: Dict, env: MultiAgentEnv, agent_id: str) -> sac.SAC:
    sac_config = sac.SACConfig().environment(
        env="f110_multi",
        env_config=config["environment"]
    ).rollouts(
        num_rollout_workers=2
    ).resources(
        num_gpus=0
    ).framework(
        framework="torch"
    ).training(
        lr=config["algorithm"]["agents"][agent_id]["lr"],
        gamma=config["algorithm"]["agents"][agent_id]["gamma"],
        tau=config["algorithm"]["agents"][agent_id]["tau"],
        target_network_update_freq=config["algorithm"]["agents"][agent_id]["target_network_update_freq"],
        train_batch_size=config["algorithm"]["agents"][agent_id]["train_batch_size"],
        replay_buffer_config=config["algorithm"]["agents"][agent_id]["replay_buffer_config"],
    ).multi_agent(
        policies={
            agent_id: (None, env.observation_space, env.action_space, {})
            for agent_id in env.agents
        },
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id
    )
    return sac_config.build()

def setup_algorithms(config: Dict, env: MultiAgentEnv) -> Dict[str, Algorithm]:
    algorithms = {}
    for agent_id in config["algorithm"]["agents"]:
        algo_type = config["algorithm"]["agents"][agent_id]["type"]
        if algo_type == "ppo":
            algorithms[agent_id] = setup_ppo(config, env, agent_id)
        elif algo_type == "sac":
            algorithms[agent_id] = setup_sac(config, env, agent_id)
    return algorithms