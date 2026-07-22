"""
Utility functions for F1TENTH benchmarking.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def load_experiment_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def save_experiment_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save experiment configuration to YAML file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def create_experiment_directory(base_dir: Path, experiment_name: str) -> Path:
    """Create standardized experiment directory structure."""

    exp_dir = base_dir / "Logs" / experiment_name

    # Create subdirectories
    (exp_dir / "RawData").mkdir(exist_ok=True, parents=True)
    (exp_dir / "Images").mkdir(exist_ok=True, parents=True)
    (exp_dir / "Analysis").mkdir(exist_ok=True, parents=True)

    return exp_dir


def get_available_experiments(base_dir: Path) -> List[str]:
    """Get list of available experiments."""

    logs_dir = base_dir / "Logs"

    if not logs_dir.exists():
        return []

    experiments = []
    for exp_dir in logs_dir.iterdir():
        if exp_dir.is_dir():
            experiments.append(exp_dir.name)

    return sorted(experiments)


def format_metrics_for_display(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary for human-readable display."""

    formatted = []

    for key, value in metrics.items():
        if isinstance(value, float):
            if 0.001 <= abs(value) < 1000:
                formatted.append(f"{key}: {value:.3f}")
            else:
                formatted.append(f"{key}: {value:.2e}")
        elif isinstance(value, int):
            formatted.append(f"{key}: {value}")
        elif isinstance(value, dict):
            formatted.append(f"{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    formatted.append(f"  {sub_key}: {sub_value:.3f}")
                else:
                    formatted.append(f"  {sub_key}: {sub_value}")
        else:
            formatted.append(f"{key}: {value}")

    return "\n".join(formatted)


class ExperimentLogger:
    """Logger for experiment data and metrics."""

    def __init__(self, experiment_name: str, base_dir: Path = Path("./Logs")):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.exp_dir = create_experiment_directory(self.base_dir, experiment_name)

        # Setup file logging
        log_file = self.exp_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger = logging.getLogger(f"experiment_{experiment_name}")
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics with optional step number."""

        step_info = f"Step {step}: " if step is not None else ""
        formatted_metrics = format_metrics_for_display(metrics)

        self.logger.info(f"{step_info}Metrics:\n{formatted_metrics}")

    def log_info(self, message: str) -> None:
        """Log general information."""
        self.logger.info(message)

    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)


def calculate_statistical_significance(data1: List[float],
                                       data2: List[float],
                                       alpha: float = 0.05) -> Dict[str, Any]:
    """Calculate statistical significance between two datasets."""

    try:
        from scipy import stats
    except ImportError:
        logger.warning("scipy not available, cannot calculate statistical significance")
        return {"error": "scipy not available"}

    if len(data1) < 2 or len(data2) < 2:
        return {"error": "Not enough data points"}

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(data1, data2)

    result = {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "alpha": alpha,
        "mean_diff": np.mean(data1) - np.mean(data2),
        "effect_size": abs(np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2)) / 2)
    }

    return result


def validate_experiment_data(data_dir: Path, experiment_name: str) -> Dict[str, Any]:
    """Validate experiment data integrity."""

    exp_dir = data_dir / "Logs" / experiment_name

    if not exp_dir.exists():
        return {"valid": False, "error": "Experiment directory not found"}

    validation_results = {
        "valid": True,
        "experiment_name": experiment_name,
        "episodes_found": 0,
        "missing_files": [],
        "corrupt_files": [],
        "warnings": []
    }

    # Find episode directories
    episode_dirs = [d for d in exp_dir.iterdir()
                    if d.is_dir() and "RawData_episode_" in d.name]

    validation_results["episodes_found"] = len(episode_dirs)

    for episode_dir in episode_dirs:
        episode_name = episode_dir.name

        # Check for required files
        required_files = ["states.npy", "actions.npy", "metrics.json"]

        for req_file in required_files:
            file_path = episode_dir / req_file
            if not file_path.exists():
                validation_results["missing_files"].append(f"{episode_name}/{req_file}")

        # Try to load files to check for corruption
        try:
            if (episode_dir / "states.npy").exists():
                np.load(episode_dir / "states.npy")
            if (episode_dir / "actions.npy").exists():
                np.load(episode_dir / "actions.npy")
        except Exception as e:
            validation_results["corrupt_files"].append(f"{episode_name}: {str(e)}")

    # Add warnings
    if validation_results["episodes_found"] == 0:
        validation_results["warnings"].append("No episode data found")

    if validation_results["missing_files"]:
        validation_results["warnings"].append(f"Missing {len(validation_results['missing_files'])} files")

    if validation_results["corrupt_files"]:
        validation_results["warnings"].append(f"Corrupt {len(validation_results['corrupt_files'])} files")
        validation_results["valid"] = False

    return validation_results
