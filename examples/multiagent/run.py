import argparse
import importlib
import os
import shutil
from lib.utils import load_config, init_ray, get_logger, suppress_warnings, get_experiment_path, get_best_checkpoint
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.callback import Callback
import datetime

suppress_warnings()
logger = get_logger(__name__)

ALGO_MAP = {
    "PPO": (PPOConfig, "ppo_params"),
    "SAC": (SACConfig, "sac_params"),
}


def get_reward_class(config):
    reward_function_name = config['training']['reward_function']
    reward_module = importlib.import_module('lib.rewards')
    return getattr(reward_module, reward_function_name)


def get_algorithm_config(config, env_config, policies, policy_mapping_fn):
    algorithm_name = config['training']['algorithm']

    if algorithm_name not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    AlgoConfigClass, config_key = ALGO_MAP[algorithm_name]

    # Get algorithm config from the merged config instead of loading separate file
    algo_config_file = config.get(config_key, {}).copy()  # Make a copy to avoid mutations
    env_kwargs = algo_config_file.get('environment', {}) 
    
    # Clean up the config before passing to training
    if 'environment' in algo_config_file:
        del algo_config_file['environment']
        
    algo_config = (
        AlgoConfigClass()
        .environment(get_reward_class(config), env_config=env_config, **env_kwargs)
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
            policy_mapping_fn=policy_mapping_fn,
        )
        .evaluation(
            evaluation_interval=config["training"]["eval_interval"],
            evaluation_num_env_runners=1,
            evaluation_config={"seed": 42},
        ).env_runners(
            num_env_runners=14,            # 4 procesos en paralelo
            num_envs_per_env_runner=14,    # 4 entornos vectorizados por proceso
            gym_env_vectorize_mode="ASYNC"
        )
        .debugging(seed=42)
    )

    algo_config.training(**algo_config_file)
    return algo_config


def create_env(config, render_mode=None):
    """Loads environment config and creates an environment instance."""
    # Since environment is always included in the same config, use embedded env_config
    env_config = config['env'].copy()
    
    if render_mode:
        env_config["render_mode"] = render_mode

    reward_class = get_reward_class(config)
    return reward_class(env_config=env_config), env_config


def run_training(config):
    temp_env, env_config = create_env(config)
    
    # Check if we should use shared policy or individual policies per agent
    shared_policy = config['training'].get('shared_policy', True)  # Default to shared policy
    
    if shared_policy:
        # Compartimos la red neuronal para todos los agentes
        logger.info("Using shared policy for all agents")
        # Usamos un único nombre de política para que todos los agentes la compartan
        shared_policy_name = "shared_policy"
        policies = {
            shared_policy_name: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
        }
        policy_mapping_fn = lambda agent_id, *args, **kwargs: shared_policy_name
    else:
        # Redes neuronales por agente
        logger.info("Using individual policies per agent")
        policies = {
            agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {}) 
            for agent in temp_env.agents
        }
        policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id
    
    temp_env.close()

    algorithm_name = config['training']['algorithm']
    # Pasamos la función de mapeo correcta a la configuración del algoritmo
    config_algo = get_algorithm_config(config, env_config, policies, policy_mapping_fn)
    
    # Create callback to save resolved config
    save_config_callback = SaveConfigCallback(config)

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
        name=config["name"],
        resume="AUTO+ERRORED",
        callbacks=[save_config_callback],
    )


def run_evaluation(config, trial_name=None):
    logger.info("Starting evaluation...")
    experiment_path = get_experiment_path(config["name"], config["storage_path"])

    logger.info(f"Loading results from: {experiment_path}")
    analysis = ExperimentAnalysis(experiment_path)
    
    if trial_name:
        # Filter trials by trial_name pattern
        filtered_trials = []
        for trial in analysis.trials:
            trial_path = trial.local_dir if hasattr(trial, 'local_dir') else ''
            if (trial_name in trial.trial_id or 
                trial_name in trial_path):
                filtered_trials.append(trial)
        
        if not filtered_trials:
            logger.error(f"No trials found matching trial name: {trial_name}")
            available_names = [trial.trial_id for trial in analysis.trials]
            logger.info(f"Available trial IDs: {available_names}")
            return
        
        analysis.trials = filtered_trials
        logger.info(f"Found {len(filtered_trials)} trial(s) matching '{trial_name}'")
    else:
        logger.info("No specific trial specified, using best trial from experiment")
    
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


class SaveConfigCallback(Callback):
    def __init__(self, resolved_config):
        self.resolved_config = resolved_config

    def setup(self, stop=None, num_samples=None, total_num_samples=None, **info):
        """Called once at the very beginning of training."""
        import yaml
        
        # Save config to the experiment's storage path
        experiment_name = self.resolved_config["name"]
        storage_path = self.resolved_config["storage_path"]
        experiment_dir = os.path.join(storage_path, experiment_name)
        
        # Create experiment directory if it doesn't exist
        os.makedirs(experiment_dir, exist_ok=True)
        
        config_file_path = os.path.join(experiment_dir, f"{experiment_name}_config.yaml")
        
        with open(config_file_path, 'w') as f:
            yaml.dump(self.resolved_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved resolved experiment config to: {config_file_path}")

def setup_experiment_config(experiment, config_dir):
    """Setup experiment configuration with resolved paths."""
    config = experiment.copy()
    
    # Resolve relative paths
    if not os.path.isabs(config["storage_path"]):
        config["storage_path"] = os.path.abspath(os.path.join(config_dir, config["storage_path"]))
    
    return config


def find_experiment(experiments, experiment_name):
    """Find experiment by name and return it, or exit with error if not found."""
    experiment = next((e for e in experiments if e["name"] == experiment_name), None)
    if not experiment:
        available = [e["name"] for e in experiments]
        logger.error(f"Experiment '{experiment_name}' not found. Available: {available}")
        exit(1)
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run F1TENTH multi-agent training or evaluation.")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Path to experiments config file")

    # Create subparsers for train and eval commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Train command
    parser_train = subparsers.add_parser("train", help="Train the model")
    parser_train.add_argument("--experiment", type=str, help="Name of the experiment to run")
    parser_train.add_argument("--all", action="store_true", help="Run all experiments")

    # Eval command
    parser_eval = subparsers.add_parser("eval", help="Evaluate the model")
    parser_eval.add_argument("--experiment", type=str, required=True, help="Name of the experiment to evaluate")
    parser_eval.add_argument("--trial", type=str, help="Specific trial to evaluate (optional, uses best trial if not specified)")

    args = parser.parse_args()

    # Load experiments config
    config_dir = os.path.dirname(os.path.abspath(args.config))
    config_data = load_config(args.config)
    experiments = config_data["experiments"]
    
    init_ray()

    if args.command == "train":
        if args.all:
            # Run all experiments
            for experiment in experiments:
                config = setup_experiment_config(experiment, config_dir)
                logger.info(f"Running experiment: {experiment['name']}")
                run_training(config)
        elif args.experiment:
            # Run specific experiment
            experiment = find_experiment(experiments, args.experiment)
            config = setup_experiment_config(experiment, config_dir)
            run_training(config)
        else:
            logger.error("Please specify --experiment <name> or --all")
            exit(1)
            
    elif args.command == "eval":
        # Evaluate specific experiment
        experiment = find_experiment(experiments, args.experiment)
        config = setup_experiment_config(experiment, config_dir)
        run_evaluation(config, args.trial)
