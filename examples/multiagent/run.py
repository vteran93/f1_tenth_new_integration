import argparse
import importlib
from lib.utils import load_config, init_ray, get_logger, suppress_warnings, get_experiment_path, get_best_checkpoint
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.analysis import ExperimentAnalysis

suppress_warnings()
logger = get_logger(__name__)

ALGO_MAP = {
    "PPO": (PPOConfig, "ppo_config"),
    "SAC": (SACConfig, "sac_config"),
}


def get_reward_class(config):
    reward_function_name = config['training']['reward_function']
    reward_module = importlib.import_module('lib.rewards')
    return getattr(reward_module, reward_function_name)


def get_algorithm_config(config, env_config, policies):
    algorithm_name = config['training']['algorithm']

    if algorithm_name not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    AlgoConfigClass, config_key = ALGO_MAP[algorithm_name]

    algo_config_file = load_config(f"configs/{config[config_key]}")

    algo_config = (
        AlgoConfigClass()
        .environment(get_reward_class(config), env_config=env_config)
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=0,
            num_envs_per_env_runner=1,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .evaluation(
            evaluation_interval=config["training"]["eval_interval"],
            evaluation_num_env_runners=1,
            evaluation_config={"seed": 42},
        )
        .debugging(seed=42)
    )
    import pdb
    pdb.set_trace()
    algo_config.training(**algo_config_file)

    return algo_config


def create_env(config, render_mode=None):
    """Loads environment config and creates an environment instance."""
    env_config = load_config(f"configs/{config['env_config']}")
    if render_mode:
        env_config["render_mode"] = render_mode

    reward_class = get_reward_class(config)
    return reward_class(env_config=env_config), env_config


def run_training(config, resume):
    temp_env, env_config = create_env(config)
    policies = {agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
                for agent in temp_env.agents}
    temp_env.close()

    algorithm_name = config['training']['algorithm']
    config_algo = get_algorithm_config(config, env_config, policies)

    tune.run(
        algorithm_name,
        config=config_algo.to_dict(),
        stop={"timesteps_total": config["training"]["timesteps_total"]},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_score_attribute="episode_reward_mean",
            checkpoint_score_order="max",
            num_to_keep=3,
            checkpoint_at_end=True,
            checkpoint_frequency=10
        ),
        storage_path=config["storage_path"],
        name=config["experiment_name"],
        resume=resume
    )


def run_evaluation(config):
    logger.info("Starting evaluation...")
    experiment_path = get_experiment_path(config["experiment_name"], config["storage_path"])

    logger.info(f"Loading results from: {experiment_path}")
    analysis = ExperimentAnalysis(experiment_path)
    best_checkpoint = get_best_checkpoint(analysis)

    if not best_checkpoint:
        logger.error("No checkpoint found. Please train the model first.")
        return

    logger.info(f"Loading checkpoint from: {best_checkpoint}")

    algo = Algorithm.from_checkpoint(best_checkpoint)

    env, _ = create_env(config, render_mode="human")

    num_episodes = config.get("evaluation", {}).get("episodes", 5)
    for eval_num in range(1, num_episodes + 1):
        logger.info(f"=== Starting evaluation episode {eval_num}/{num_episodes} ===")
        obs, info = env.reset()
        terminated = {"__all__": False}

        while not terminated["__all__"]:
            actions = {}
            for agent_id, agent_obs in obs.items():
                actions[agent_id] = algo.compute_single_action(
                    observation=agent_obs,
                    policy_id=agent_id,
                    explore=False
                )
            obs, reward, terminated, truncated, info = env.step(actions)
            env.render()
        logger.info(f"Evaluation episode {eval_num} finished.")

    env.close()
    logger.info(f"All {num_episodes} evaluation episodes finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run F1TENTH multi-agent training or evaluation.")
    parser.add_argument("--config_path", default="configs/run.yaml", help="Path to the run config file")

    # Create subparsers for train and eval commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Create the parser for the "train" command
    parser_train = subparsers.add_parser("train", help="Train the model")
    parser_train.add_argument("--resume", action="store_true", help="Resume training from a checkpoint")

    # Create the parser for the "eval" command
    parser_eval = subparsers.add_parser("eval", help="Evaluate the model")

    args = parser.parse_args()

    config = load_config(args.config_path)
    init_ray()

    if args.command == "train":
        run_training(config, args.resume)
    elif args.command == "eval":
        run_evaluation(config)
