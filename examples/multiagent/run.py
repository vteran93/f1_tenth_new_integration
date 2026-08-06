import argparse
import sys
from lib.utils import (
    load_config,
    init_ray,
    get_logger,
    suppress_warnings,
    get_experiment_path,
    get_best_checkpoint,
    get_reward_class,
    setup_experiment_config,
    find_experiment,
    validate_hyperparameter_config
)
from lib.callbacks import SaveConfig, MultipleAgentCallbacks
from lib.nan_detection import NaNProtectedTorchDiagGaussian
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.stopper import TrialPlateauStopper, CombinedStopper, Stopper
from ray.tune.search.optuna import OptunaSearch
from pathlib import Path

suppress_warnings()
logger = get_logger(__name__)

ALGO_MAP = {
    "PPO": (PPOConfig, "ppo_params"),
    "SAC": (SACConfig, "sac_params"),
}

SEED = 42

# Name under which the NaN-protected Gaussian action distribution is registered.
# Enabling `nan_protection: true` in a config injects this as the policy's
# custom_action_dist, which sanitises NaN/inf action means and log-stds before the
# Normal distribution is built. This prevents the hard crash observed on long PPO runs
# ("Expected parameter loc ... to satisfy the constraint Real()") that otherwise kills
# the trial mid-iteration. See lib/nan_detection.py.
NAN_PROTECTED_ACTION_DIST = "nan_protected_gaussian"


def register_nan_protected_action_dist():
    """Register the NaN-protected action distribution under its catalog name.

    Called once on the driver at import time. RLlib serialises the custom
    ModelCatalog registration into the algorithm config, so the tune trainable
    actor and every env_runner resolve the name without needing to re-register
    (verified: the distribution is constructed in the worker processes, not only
    the driver). Re-registering the same name is idempotent.
    """
    ModelCatalog.register_custom_action_dist(
        NAN_PROTECTED_ACTION_DIST, NaNProtectedTorchDiagGaussian
    )


# Register on the driver at import time.
register_nan_protected_action_dist()


class TimestepsStopper(Stopper):
    """Custom stopper that stops training after a certain number of timesteps."""

    def __init__(self, max_timesteps):
        self.max_timesteps = max_timesteps

    def __call__(self, trial_id, result):
        """Stop trial if timesteps_total exceeds max_timesteps."""
        return result.get("timesteps_total", 0) >= self.max_timesteps

    def stop_all(self):
        """Don't stop all experiments."""
        return False


def get_algorithm_config(config, env_config, policies, policy_mapping_fn, num_env_runners=8, num_envs_per_env_runner=2):
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

    # Opt-in NaN protection: route the policy through the NaN-protected action
    # distribution. Injected into the model config so it is serialised to every
    # env_runner/learner rather than relying on a driver-only monkey-patch.
    if config.get('nan_protection', False):
        model_cfg = dict(algo_config_file.get('model', {}))
        model_cfg['custom_action_dist'] = NAN_PROTECTED_ACTION_DIST
        algo_config_file['model'] = model_cfg
        logger.info(f"NaN protection enabled: using custom_action_dist='{NAN_PROTECTED_ACTION_DIST}'")

    algo_config = (
        AlgoConfigClass()
        .environment(get_reward_class(config), env_config=env_config, **env_kwargs)
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .callbacks(MultipleAgentCallbacks)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .evaluation(
            evaluation_interval=config["training"]["eval_interval"],
            evaluation_num_env_runners=1,
            evaluation_config={"seed": SEED},
        )
        .env_runners(
            num_env_runners=num_env_runners,             # Number of parallel processes (<= CPUs)
            num_envs_per_env_runner=num_envs_per_env_runner,     # Vectorized environments per process
            gym_env_vectorize_mode="ASYNC"
        )
        .debugging(seed=SEED)
    )

    algo_config.training(**algo_config_file)
    return algo_config


def create_env(config, render_mode=None):
    """Loads environment config and creates an environment instance."""
    suppress_warnings()  # Suppress warnings in worker processes
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
        logger.info("Using shared policy")
        shared_policy_name = "shared_policy"
        policies = {
            shared_policy_name: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
        }
        policy_mapping_fn = lambda agent_id, *args, **kwargs: shared_policy_name
    else:
        logger.info("Using individual policies")
        policies = {
            agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
            for agent in temp_env.agents
        }
        policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id

    temp_env.close()

    algorithm_name = config['training']['algorithm']
    
    # Get env_runners configuration from global config
    num_env_runners = config.get('num_env_runners', 8)
    num_envs_per_env_runner = config.get('num_envs_per_env_runner', 2)
    
    config_algo = get_algorithm_config(config, env_config, policies, policy_mapping_fn, 
                                     num_env_runners, num_envs_per_env_runner)

    # Define plateau stopper for convergence
    plateau_stopper = TrialPlateauStopper(
        metric="env_runners/episode_return_mean",  # Metric to monitor
        std=10.0,                # Standard deviation threshold change
        num_results=20,         # Number of results to consider for the standard deviation
        # grace_period=275,       # Minimum iterations before considering stopping, this allow initial exploration variability
        # grace_period is the number of iterations, we can calculate it based on steps = train_batch_size * iterations
        # so 275 iterations with a train_batch_size of 4000 is about 1_100_000 steps
        mode="max",
    )

    # Define timesteps stopper
    timesteps_stopper = TimestepsStopper(max_timesteps=config["training"]["timesteps_total"])

    # Combine both stoppers - trial will stop if ANY of the conditions is met
    combined_stopper = CombinedStopper(
        plateau_stopper,
        timesteps_stopper
    )

    # Setup search algorithm for hyperparameter tuning
    tune_kwargs = {}

    # Validate hyperparameter configuration
    validate_hyperparameter_config(config)

    hyperparameter_tuning = config.get("hyperparameter_tuning", False)

    if hyperparameter_tuning:
        default_metric = "env_runners/episode_reward_mean"
        search_alg = OptunaSearch(metric=default_metric, mode="max", seed=SEED)
        tune_kwargs["search_alg"] = search_alg
        tune_kwargs["num_samples"] = config["num_samples"]
        logger.info(f"Hyperparameter tuning enabled with {tune_kwargs['num_samples']} trials")
    else:
        tune_kwargs["num_samples"] = 1
        logger.info("Hyperparameter tuning disabled. Using num_samples=1 for single trial run.")

    # TODO, should we assign tune.run to a variable? like: results = tune.run(...)
    # So potentially we can access the results later or write them to a file, for ex.
    # best_trial = results.get_best_trial("env_runners/episode_return_mean", "max")
    # best_checkpoint_path = results.get_best_checkpoint(best_trial, "env_runners/episode_return_mean", "max")
    
    tune.run(
        algorithm_name,
        config=config_algo.to_dict(),
        stop=combined_stopper,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_score_attribute="env_runners/episode_return_mean",
            checkpoint_score_order="max",
            num_to_keep=3,
            checkpoint_at_end=True,
            checkpoint_frequency=10
        ),
        storage_path=config["storage_path"],
        name=config["name"],
        resume="AUTO+ERRORED",
        callbacks=[SaveConfig(config)],
        max_failures=5,  # Allow up to 3 failures before stopping the trial
        **tune_kwargs
    )


def run_evaluation(config, trial_name=None):
    logger.info("Starting evaluation")
    experiment_path = get_experiment_path(config["name"], config["storage_path"])

    logger.debug(f"Loading results from: {experiment_path}")
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
            logger.error(f"No trials found matching: {trial_name}")
            available_names = [trial.trial_id for trial in analysis.trials]
            logger.debug(f"Available trials: {available_names}")
            return

        analysis.trials = filtered_trials
        logger.info(f"Found {len(filtered_trials)} trial(s) matching '{trial_name}'")
    else:
        logger.debug("Using best trial from experiment")

    best_checkpoint = get_best_checkpoint(analysis)

    if not best_checkpoint:
        logger.error("No checkpoint found. Train model first")
        return

    logger.debug(f"Loading checkpoint: {best_checkpoint}")

    algo = Algorithm.from_checkpoint(best_checkpoint)

    env, _ = create_env(config, render_mode="human")

    # Determinar si el modelo se entrenó con una política compartida
    shared_policy = config['training'].get('shared_policy', True)

    num_episodes = config.get("evaluation", {}).get("episodes", 5)
    logger.info(f"Running {num_episodes} evaluation episodes")

    for eval_num in range(1, num_episodes + 1):
        logger.debug(f"Episode {eval_num}/{num_episodes}")
        obs, info = env.reset()
        terminated = {"__all__": False}

        while not terminated["__all__"]:
            actions = {}
            for agent_id, agent_obs in obs.items():
                # Usar el ID de política correcto según la configuración del experimento
                policy_id = "shared_policy" if shared_policy else agent_id
                actions[agent_id] = algo.compute_single_action(
                    observation=agent_obs,
                    policy_id=policy_id,
                    explore=False
                )
            obs, reward, terminated, truncated, info = env.step(actions)
            env.render()

    env.close()
    logger.info(f"All {num_episodes} evaluation episodes finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog=Path(__file__).stem,
        description='Run F1TENTH multi-agent training or evaluation.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
            Examples:
            python run.py train --experiment my_experiment
            python run.py train --all
            python run.py eval --experiment my_experiment
            python run.py eval --experiment my_experiment --trial trial_123
        """
    )
    parser.add_argument(
        '--config', type=Path, default=Path('configs/experiments.yaml'),
        help='Path to the experiments configuration file (default: configs/experiments.yaml)')

    subparsers = parser.add_subparsers(
        title='Commands',
        dest='command',
        required=True,
        help='Available commands:',
        description='Choose one of the following commands:'
    )

    # Train parser
    train_parser = subparsers.add_parser('train', help='Train one or more experiments')
    train_group = train_parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument('--experiment', type=str, help='Name of the experiment to train')
    train_group.add_argument('--all', action='store_true', help='Train all experiments from config file')

    # Eval parser
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained model')
    eval_parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment to evaluate')
    eval_parser.add_argument('--trial', type=str, default=None,
                             help='Specific trial to evaluate (optional, uses best trial if not specified)')

    if len(sys.argv) == 1:  # Si no se pasan argumentos
        parser.print_help()  # Mostrar el mensaje de ayuda
        sys.exit(0)  # Salir del programa

    args = parser.parse_args()

    config_path = args.config.resolve()
    config_dir = config_path.parent
    config_data = load_config(config_path)
    experiments = config_data.get('experiments', [])

    # Get num_cpus from config, default to 16 if not specified
    num_cpus = config_data.get('num_cpus', 16)
    # Get env_runners configuration from global config
    global_num_env_runners = config_data.get('num_env_runners', 8)
    global_num_envs_per_env_runner = config_data.get('num_envs_per_env_runner', 2)

    init_ray(num_cpus=num_cpus)

    if args.command == 'train':
        if args.all:
            experiments_to_run = experiments
        else:
            experiments_to_run = [find_experiment(experiments, args.experiment)]

        for experiment in experiments_to_run:
            cfg = setup_experiment_config(experiment, config_dir)
            # Add global env_runners config to each experiment
            cfg['num_env_runners'] = global_num_env_runners
            cfg['num_envs_per_env_runner'] = global_num_envs_per_env_runner
            logger.info(f"Training: {cfg['name']}")
            try:
                run_training(cfg)
            except Exception as e:
                logger.error(f"Error during training of {cfg['name']}: {e}")
                raise e
                continue
            # This way we can catch errors in training and continue with the next experiment

    elif args.command == 'eval':
        experiment = find_experiment(experiments, args.experiment)
        cfg = setup_experiment_config(experiment, config_dir)
        # Add global env_runners config (not needed for eval but for consistency)
        cfg['num_env_runners'] = global_num_env_runners
        cfg['num_envs_per_env_runner'] = global_num_envs_per_env_runner
        logger.info(f"Evaluating: {cfg['name']}" + (f" (trial={args.trial})" if args.trial else ""))
        run_evaluation(cfg, args.trial)
