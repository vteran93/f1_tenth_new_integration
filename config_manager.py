#!/usr/bin/env python3
"""
Advanced Training Configuration Manager for F1TENTH RL Experiments

This module provides a comprehensive system for managing training configurations,
including automatic hyperparameter search spaces, experiment templates,
and configuration validation.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Algorithm(Enum):
    """Supported RL algorithms."""
    PPO = "PPO"
    SAC = "SAC"
    IMPALA = "IMPALA"
    A3C = "A3C"
    APEX_DQN = "APEX_DQN"


class Environment(Enum):
    """Available F1TENTH environments."""
    OVAL_SMALL = "oval_small"
    OVAL_LARGE = "oval_large"
    RACETRACK = "racetrack"
    CUSTOM = "custom"


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    timesteps_total: int = 3_000_000
    eval_interval: int = 2000
    evaluation_episodes: int = 5
    checkpoint_interval: int = 1000
    reward_function: str = "ProgressRewardEnv"

    # Training resources
    num_workers: int = 4
    num_gpus: float = 1.0
    num_cpus_per_worker: int = 1

    # Advanced options
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 256
    num_sgd_iter: int = 10


@dataclass
class EnvironmentConfig:
    """Environment configuration dataclass."""
    map_name: str = "oval_small"
    num_agents: int = 2
    timestep: float = 0.01
    num_beams: int = 36
    integrator: str = "rk4"
    control_input: List[str] = field(default_factory=lambda: ["speed", "steering_angle"])
    observation_config: Dict[str, Any] = field(default_factory=lambda: {"type": "original"})
    reset_config: Dict[str, Any] = field(default_factory=lambda: {"type": "cl_grid_static"})
    render_mode: Optional[str] = None

    # Advanced environment options
    safety_mode: bool = False
    collision_penalty: float = -10.0
    progress_reward_scale: float = 1.0


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""
    gamma: float = 0.99
    lambda_: float = 0.95
    lr: float = 0.00005
    clip_param: float = 0.2
    value_function_clip_param: float = 10.0
    entropy_coeff: float = 0.01
    vf_loss_coeff: float = 1.0
    kl_coeff: float = 0.2
    normalize_actions: bool = True


@dataclass
class SACConfig:
    """SAC algorithm configuration."""
    tau: float = 0.005
    target_entropy: Union[str, float] = "auto"
    alpha_lr: float = 0.0003
    actor_lr: float = 0.0003
    critic_lr: float = 0.0003
    gamma: float = 0.99
    replay_buffer_size: int = 1_000_000
    normalize_actions: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    algorithm: Algorithm
    storage_path: str = "./models"

    # Sub-configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    ppo: Optional[PPOConfig] = None
    sac: Optional[SACConfig] = None

    # Experiment metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = ""

    def __post_init__(self):
        """Initialize algorithm-specific configs."""
        if self.algorithm == Algorithm.PPO and self.ppo is None:
            self.ppo = PPOConfig()
        elif self.algorithm == Algorithm.SAC and self.sac is None:
            self.sac = SACConfig()


class ConfigurationManager:
    """Manager for F1TENTH RL experiment configurations."""

    def __init__(self, config_dir: Path = Path("./configs")):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.config_dir / "experiments").mkdir(exist_ok=True)
        (self.config_dir / "templates").mkdir(exist_ok=True)
        (self.config_dir / "hyperparameter_search").mkdir(exist_ok=True)

    def create_experiment_config(self,
                                 name: str,
                                 algorithm: Algorithm,
                                 environment: Environment = Environment.OVAL_SMALL,
                                 **kwargs) -> ExperimentConfig:
        """Create a new experiment configuration."""

        # Create environment config
        env_config = EnvironmentConfig(map_name=environment.value)

        # Create training config
        training_config = TrainingConfig()

        # Create experiment config
        config = ExperimentConfig(
            name=name,
            algorithm=algorithm,
            environment=env_config,
            training=training_config,
            **kwargs
        )

        return config

    def save_config(self, config: ExperimentConfig, filename: Optional[str] = None) -> Path:
        """Save configuration to YAML file."""
        if filename is None:
            filename = f"{config.name}.yaml"

        filepath = self.config_dir / "experiments" / filename

        # Convert to dict and handle enums
        config_dict = asdict(config)
        config_dict["algorithm"] = config.algorithm.value

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {filepath}")
        return filepath

    def load_config(self, filepath: Union[str, Path]) -> ExperimentConfig:
        """Load configuration from YAML file."""
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Handle algorithm enum
        config_dict["algorithm"] = Algorithm(config_dict["algorithm"])

        # Reconstruct nested configs
        if "training" in config_dict:
            config_dict["training"] = TrainingConfig(**config_dict["training"])

        if "environment" in config_dict:
            config_dict["environment"] = EnvironmentConfig(**config_dict["environment"])

        if "ppo" in config_dict and config_dict["ppo"]:
            config_dict["ppo"] = PPOConfig(**config_dict["ppo"])

        if "sac" in config_dict and config_dict["sac"]:
            config_dict["sac"] = SACConfig(**config_dict["sac"])

        return ExperimentConfig(**config_dict)

    def create_hyperparameter_search_space(self,
                                           base_config: ExperimentConfig,
                                           search_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create hyperparameter search space for Ray Tune."""
        from ray import tune

        search_space = {}

        # Handle training parameters
        if "training" in search_params:
            for param, values in search_params["training"].items():
                if isinstance(values, dict) and "type" in values:
                    if values["type"] == "uniform":
                        search_space[f"training.{param}"] = tune.uniform(values["low"], values["high"])
                    elif values["type"] == "loguniform":
                        search_space[f"training.{param}"] = tune.loguniform(values["low"], values["high"])
                    elif values["type"] == "choice":
                        search_space[f"training.{param}"] = tune.choice(values["choices"])
                else:
                    search_space[f"training.{param}"] = tune.choice(values)

        # Handle algorithm parameters
        if base_config.algorithm == Algorithm.PPO and "ppo" in search_params:
            for param, values in search_params["ppo"].items():
                if isinstance(values, dict) and "type" in values:
                    if values["type"] == "uniform":
                        search_space[f"ppo.{param}"] = tune.uniform(values["low"], values["high"])
                    elif values["type"] == "loguniform":
                        search_space[f"ppo.{param}"] = tune.loguniform(values["low"], values["high"])
                    elif values["type"] == "choice":
                        search_space[f"ppo.{param}"] = tune.choice(values["choices"])
                else:
                    search_space[f"ppo.{param}"] = tune.choice(values)

        return search_space

    def generate_config_templates(self):
        """Generate predefined configuration templates."""

        # Template 1: Single Agent Racing
        single_agent_config = self.create_experiment_config(
            name="single_agent_racing",
            algorithm=Algorithm.PPO,
            environment=Environment.OVAL_SMALL,
            description="Single agent racing configuration for learning basic racing behavior",
            tags=["single_agent", "racing", "ppo"]
        )
        single_agent_config.environment.num_agents = 1
        single_agent_config.training.timesteps_total = 2_000_000

        # Template 2: Multi-Agent Competition
        multi_agent_config = self.create_experiment_config(
            name="multi_agent_competition",
            algorithm=Algorithm.PPO,
            environment=Environment.OVAL_SMALL,
            description="Multi-agent competitive racing configuration",
            tags=["multi_agent", "competition", "ppo"]
        )
        multi_agent_config.environment.num_agents = 4
        multi_agent_config.training.timesteps_total = 5_000_000
        multi_agent_config.training.num_workers = 8

        # Template 3: SAC Exploration
        sac_config = self.create_experiment_config(
            name="sac_exploration",
            algorithm=Algorithm.SAC,
            environment=Environment.OVAL_LARGE,
            description="SAC configuration for exploration and continuous control",
            tags=["sac", "exploration", "continuous_control"]
        )
        sac_config.training.timesteps_total = 3_000_000

        # Template 4: Safety-First Racing
        safety_config = self.create_experiment_config(
            name="safety_first_racing",
            algorithm=Algorithm.PPO,
            environment=Environment.RACETRACK,
            description="Safety-focused racing with collision avoidance",
            tags=["safety", "collision_avoidance", "ppo"]
        )
        safety_config.environment.safety_mode = True
        safety_config.environment.collision_penalty = -50.0
        safety_config.training.reward_function = "SafetyProgressRewardEnv"

        templates = [
            single_agent_config,
            multi_agent_config,
            sac_config,
            safety_config
        ]

        # Save templates
        template_dir = self.config_dir / "templates"
        for template in templates:
            # Create template file path directly
            template_file = template_dir / f"{template.name}.yaml"

            # Convert to dict and handle enums
            config_dict = asdict(template)
            config_dict["algorithm"] = template.algorithm.value

            with open(template_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            logger.info(f"Template saved to {template_file}")

        logger.info(f"Generated {len(templates)} configuration templates")
        return templates

    def validate_config(self, config: ExperimentConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate training config
        if config.training.timesteps_total <= 0:
            issues.append("Training timesteps must be positive")

        if config.training.num_workers < 1:
            issues.append("Number of workers must be at least 1")

        # Validate environment config
        if config.environment.num_agents < 1:
            issues.append("Number of agents must be at least 1")

        if config.environment.timestep <= 0:
            issues.append("Environment timestep must be positive")

        # Validate algorithm-specific configs
        if config.algorithm == Algorithm.PPO:
            if config.ppo is None:
                issues.append("PPO configuration is required for PPO algorithm")
            elif config.ppo.lr <= 0:
                issues.append("PPO learning rate must be positive")

        elif config.algorithm == Algorithm.SAC:
            if config.sac is None:
                issues.append("SAC configuration is required for SAC algorithm")
            elif config.sac.actor_lr <= 0:
                issues.append("SAC actor learning rate must be positive")

        return issues

    def list_available_configs(self) -> List[str]:
        """List all available configuration files."""
        config_files = []

        # List experiment configs
        exp_dir = self.config_dir / "experiments"
        if exp_dir.exists():
            config_files.extend([f.stem for f in exp_dir.glob("*.yaml")])

        # List template configs
        template_dir = self.config_dir / "templates"
        if template_dir.exists():
            config_files.extend([f"template:{f.stem}" for f in template_dir.glob("*.yaml")])

        return config_files

    def create_sweep_config(self,
                            base_config: ExperimentConfig,
                            sweep_params: Dict[str, List[Any]]) -> List[ExperimentConfig]:
        """Create a parameter sweep of configurations."""
        from itertools import product

        # Generate all combinations
        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())

        configs = []
        for i, combination in enumerate(product(*param_values)):
            # Create new config
            config = ExperimentConfig(**asdict(base_config))
            config.name = f"{base_config.name}_sweep_{i:03d}"

            # Apply parameters
            for param_name, value in zip(param_names, combination):
                if "." in param_name:
                    # Handle nested parameters
                    parts = param_name.split(".")
                    obj = config
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)
                else:
                    setattr(config, param_name, value)

            configs.append(config)

        return configs


def main():
    """Example usage of the configuration manager."""

    # Initialize manager
    manager = ConfigurationManager()

    # Generate templates
    templates = manager.generate_config_templates()
    print(f"Generated {len(templates)} templates")

    # Create a custom experiment
    custom_config = manager.create_experiment_config(
        name="my_custom_experiment",
        algorithm=Algorithm.PPO,
        environment=Environment.OVAL_SMALL,
        description="My custom racing experiment",
        tags=["custom", "testing"]
    )

    # Modify some parameters
    custom_config.training.timesteps_total = 1_000_000
    custom_config.environment.num_agents = 3
    custom_config.ppo.lr = 0.0001

    # Validate configuration
    issues = manager.validate_config(custom_config)
    if issues:
        print(f"Configuration issues: {issues}")
    else:
        print("Configuration is valid")

    # Save configuration
    config_path = manager.save_config(custom_config)
    print(f"Configuration saved to {config_path}")

    # Load and verify
    loaded_config = manager.load_config(config_path)
    print(f"Loaded configuration: {loaded_config.name}")

    # List available configs
    available = manager.list_available_configs()
    print(f"Available configurations: {available}")


if __name__ == "__main__":
    main()
