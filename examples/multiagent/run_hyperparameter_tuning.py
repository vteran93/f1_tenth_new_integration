import os
import numpy as np
import torch
import random
import click
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.rllib.algorithms.ppo import PPO as PPOTrainer, PPOConfig
from lib.utils import load_config, init_ray, get_logger, suppress_warnings
from examples.multiagent.lib.rewards import ProgressRewardEnv
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec

# --- Determinismo ---
os.environ["PYTHONHASHSEED"] = "42"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Registrar el entorno ---
register_env("f1tenth_multi", lambda config: MultiAgentF110(config))

# --- Espacio de búsqueda de hiperparámetros ---
search_space = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "gamma": tune.uniform(0.95, 0.999),
    "lambda_": tune.uniform(0.9, 1.0),
    "clip_param": tune.uniform(0.1, 0.4),
    "train_batch_size": tune.choice([256, 512, 1024, 2048, 4096]),
    "num_sgd_iter": tune.choice([5, 10, 20]),
    "sgd_minibatch_size": tune.choice([64, 128, 256]),
    "entropy_coeff": tune.uniform(0.0, 0.05),
    "model": {"fcnet_hiddens": tune.choice([(128, 128), (256, 256), (256, 128)])}
}


def policy_dict():
    temp_env = MultiAgentF110(get_env_config())
    policies = {agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
                for agent in temp_env.agents}
    temp_env.close()
    return policies


# --- Configuración base sin hiperparámetros ---
base_cfg = (
    PPOConfig()
    .environment("f1tenth_multi", env_config=get_env_config())
    .framework("torch")
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
    .multi_agent(
        policies=policy_dict(),
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id
    )
    .evaluation(evaluation_interval=10, evaluation_num_env_runners=1, evaluation_config={"seed": SEED + 1})
    .debugging(seed=SEED)
).to_dict()

# --- Integrar espacio de búsqueda en la configuración ---
config = base_cfg.copy()
config.update(search_space)

# --- Buscador Bayesiano ---
default_metric = "env_runners/episode_reward_mean"  # <-- Cambia aquí
search_alg = OptunaSearch(metric=default_metric, mode="max", seed=SEED)

# --- Ejecutar tuning ---
tune.run(
    PPOTrainer,
    config=config,
    num_samples=50,
    stop={"timesteps_total": 200_000},
    checkpoint_freq=10,
    storage_path=PATH_RESULTS,
    name="f1tenth_multiagent_ppo_tune",
    search_alg=search_alg,
    metric=default_metric,   # <-- Añade esto para ser explícito
    mode="max"
)


@click.command()
@click.option("--train", is_flag=True, help="Train the model")
@click.option("--resume", is_flag=True, help="Resume training from a checkpoint")
@click.option("--eval", is_flag=True, help="Evaluate the model")
@click.option("--config_path", default="examples/multiagent/configs/hyperparameter_tuning.yaml", help="Path to the hyperparameter tuning config file")
def main(train, resume, eval, config_path):
    config = load_config(config_path)
    init_ray()

    if train:
        run_training(config, resume)

    if eval:
        # Evaluation logic will be added here
        pass


def run_training(config, resume):
    env_config = load_config(f"examples/multiagent/configs/{config['env_config']}")
    temp_env = ProgressRewardEnv(env_config)
    policies = {agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
                for agent in temp_env.agents}
    temp_env.close()

    ppo_config_file = load_config(f"examples/multiagent/configs/{config['ppo_config']}")
    config_ppo = (PPOConfig()
                  .environment(ProgressRewardEnv, env_config=env_config)
                  .framework("torch")
                  .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                  .env_runners(
                  num_env_runners=0,
                  num_envs_per_env_runner=1,
                  )
                  .multi_agent(
                  policies=policies,
                  policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id
                  )
                  .training(
                  train_batch_size=tune.grid_search(config["hyperparameters"]["train_batch_size"]),
                  lr=tune.grid_search(config["hyperparameters"]["lr"])
                  )
                  .evaluation(
                  evaluation_interval=config["evaluation_interval"],
                  evaluation_num_env_runners=1,
                  evaluation_config={
                      "seed": 42
                  }
                  )
                  .debugging(
                  seed=42
                  ))

    tune.run(
        "PPO",
        config=config_ppo.to_dict(),
        stop={"timesteps_total": config["total_timesteps"]},
        checkpoint_freq=10,
        storage_path="ray_results",
        name=config["experiment_name"],
        resume=resume
    )


if __name__ == "__main__":
    main()
