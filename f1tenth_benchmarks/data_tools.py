"""
Data Tools for F1TENTH Benchmarking

Standalone data processing and analysis tools for F1TENTH RL experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and format F1TENTH training and evaluation data."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

    def save_episode_data(self,
                          episode_data: Dict[str, Any],
                          experiment_name: str,
                          episode_id: int) -> Path:
        """Save episode data in benchmark format."""

        # Create experiment directory structure
        exp_dir = self.data_dir / "Logs" / experiment_name / f"RawData_episode_{episode_id:04d}"
        exp_dir.mkdir(exist_ok=True, parents=True)

        # Save states as numpy array
        if 'states' in episode_data and episode_data['states']:
            states_array = np.array(episode_data['states'])
            np.save(exp_dir / "states.npy", states_array)

        # Save actions
        if 'actions' in episode_data and episode_data['actions']:
            actions_array = np.array(episode_data['actions'])
            np.save(exp_dir / "actions.npy", actions_array)

        # Save additional metrics as JSON
        metrics = {
            'episode_id': episode_id,
            'total_reward': episode_data.get('total_reward', 0.0),
            'completion': episode_data.get('completion', False),
            'total_progress': episode_data.get('total_progress', 0.0),
            'collisions': episode_data.get('collisions', 0),
            'lap_times': episode_data.get('lap_times', []),
            'avg_speed': episode_data.get('avg_speed', 0.0),
            'steering_smoothness': episode_data.get('steering_smoothness', 0.0),
            'path_efficiency': episode_data.get('path_efficiency', 0.0)
        }

        with open(exp_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved episode {episode_id} data to {exp_dir}")
        return exp_dir

    def load_episode_data(self, experiment_name: str, episode_id: int) -> Optional[Dict[str, Any]]:
        """Load episode data from files."""

        exp_dir = self.data_dir / "Logs" / experiment_name / f"RawData_episode_{episode_id:04d}"

        if not exp_dir.exists():
            return None

        data = {}

        # Load numpy arrays
        if (exp_dir / "states.npy").exists():
            data['states'] = np.load(exp_dir / "states.npy")

        if (exp_dir / "actions.npy").exists():
            data['actions'] = np.load(exp_dir / "actions.npy")

        # Load metrics
        if (exp_dir / "metrics.json").exists():
            with open(exp_dir / "metrics.json", 'r') as f:
                data['metrics'] = json.load(f)

        return data

    def get_experiment_summary(self, experiment_name: str) -> pd.DataFrame:
        """Get summary statistics for all episodes in an experiment."""

        exp_base_dir = self.data_dir / "Logs" / experiment_name

        if not exp_base_dir.exists():
            return pd.DataFrame()

        # Find all episode directories
        episode_dirs = [d for d in exp_base_dir.iterdir()
                        if d.is_dir() and d.name.startswith("RawData_episode_")]

        summary_data = []

        for episode_dir in episode_dirs:
            metrics_file = episode_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    summary_data.append(metrics)

        if summary_data:
            return pd.DataFrame(summary_data)
        else:
            return pd.DataFrame()


class BenchmarkAnalyzer:
    """Analyze benchmark data and generate reports."""

    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor

    def analyze_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Comprehensive analysis of an experiment."""

        summary_df = self.data_processor.get_experiment_summary(experiment_name)

        if summary_df.empty:
            return {"error": "No data found for experiment"}

        analysis = {
            'experiment_name': experiment_name,
            'total_episodes': len(summary_df),
            'completion_rate': summary_df['completion'].mean() if 'completion' in summary_df else 0,
            'average_reward': summary_df['total_reward'].mean() if 'total_reward' in summary_df else 0,
            'average_progress': summary_df['total_progress'].mean() if 'total_progress' in summary_df else 0,
            'total_collisions': summary_df['collisions'].sum() if 'collisions' in summary_df else 0,
            'collision_rate': summary_df['collisions'].mean() if 'collisions' in summary_df else 0,
        }

        # Lap time analysis
        if 'lap_times' in summary_df.columns:
            all_lap_times = []
            for lap_times in summary_df['lap_times']:
                if isinstance(lap_times, list):
                    all_lap_times.extend(lap_times)

            if all_lap_times:
                analysis['lap_times'] = {
                    'mean': np.mean(all_lap_times),
                    'std': np.std(all_lap_times),
                    'min': np.min(all_lap_times),
                    'max': np.max(all_lap_times),
                    'count': len(all_lap_times)
                }

        # Performance metrics
        if 'avg_speed' in summary_df.columns:
            analysis['avg_speed'] = summary_df['avg_speed'].mean()

        if 'steering_smoothness' in summary_df.columns:
            analysis['steering_smoothness'] = summary_df['steering_smoothness'].mean()

        if 'path_efficiency' in summary_df.columns:
            analysis['path_efficiency'] = summary_df['path_efficiency'].mean()

        return analysis

    def compare_experiments(self, experiment_names: List[str]) -> pd.DataFrame:
        """Compare multiple experiments."""

        comparison_data = []

        for exp_name in experiment_names:
            analysis = self.analyze_experiment(exp_name)
            if 'error' not in analysis:
                comparison_data.append(analysis)

        return pd.DataFrame(comparison_data)

    def generate_report(self, experiment_name: str, output_path: Optional[Path] = None) -> str:
        """Generate a comprehensive analysis report."""

        analysis = self.analyze_experiment(experiment_name)

        if 'error' in analysis:
            return f"Error: {analysis['error']}"

        report = f"""
# F1TENTH Benchmark Analysis Report

**Experiment:** {analysis['experiment_name']}
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

- **Total Episodes:** {analysis['total_episodes']}
- **Completion Rate:** {analysis.get('completion_rate', 0):.2%}
- **Average Reward:** {analysis.get('average_reward', 0):.2f}
- **Average Progress:** {analysis.get('average_progress', 0):.2%}
- **Total Collisions:** {analysis.get('total_collisions', 0)}
- **Collision Rate:** {analysis.get('collision_rate', 0):.2f} per episode

## Performance Metrics

"""

        if 'lap_times' in analysis:
            lap_stats = analysis['lap_times']
            report += f"""
### Lap Times
- **Best Lap:** {lap_stats['min']:.3f}s
- **Average Lap:** {lap_stats['mean']:.3f}s (±{lap_stats['std']:.3f}s)
- **Worst Lap:** {lap_stats['max']:.3f}s
- **Total Laps:** {lap_stats['count']}
"""

        if 'avg_speed' in analysis:
            report += f"""
### Driving Behavior
- **Average Speed:** {analysis['avg_speed']:.2f} m/s
- **Steering Smoothness:** {analysis.get('steering_smoothness', 0):.3f}
- **Path Efficiency:** {analysis.get('path_efficiency', 0):.3f}
"""

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report


class MetricsCalculator:
    """Calculate detailed performance metrics from trajectory data."""

    @staticmethod
    def calculate_lap_time(states: np.ndarray, timestep: float = 0.01) -> float:
        """Calculate lap time from states array."""
        if len(states) == 0:
            return 0.0
        return len(states) * timestep

    @staticmethod
    def calculate_path_length(states: np.ndarray) -> float:
        """Calculate total path length."""
        if len(states) < 2:
            return 0.0

        # Extract x, y positions (assuming first two columns)
        positions = states[:, :2]
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return np.sum(distances)

    @staticmethod
    def calculate_average_speed(states: np.ndarray) -> float:
        """Calculate average speed from states."""
        if len(states) == 0:
            return 0.0

        # Assuming speed is in column 3
        if states.shape[1] > 3:
            speeds = states[:, 3]
            return np.mean(np.abs(speeds))

        return 0.0

    @staticmethod
    def calculate_steering_smoothness(actions: np.ndarray) -> float:
        """Calculate steering smoothness metric."""
        if len(actions) < 2:
            return 0.0

        # Assuming steering is the second action (index 1)
        if actions.shape[1] > 1:
            steering_angles = actions[:, 1]
            steering_changes = np.diff(steering_angles)
            smoothness = 1.0 / (1.0 + np.var(steering_changes))
            return smoothness

        return 0.0

    @staticmethod
    def calculate_progress_rate(states: np.ndarray, track_length: float = 100.0) -> float:
        """Calculate progress rate along the track."""
        if len(states) == 0:
            return 0.0

        # This is a simplified calculation
        # In practice, you'd need track-specific progress calculation
        path_length = MetricsCalculator.calculate_path_length(states)
        return min(path_length / track_length, 1.0)


class TrajectoryAnalyzer:
    """Analyze vehicle trajectories and racing lines."""

    def __init__(self):
        self.metrics_calc = MetricsCalculator()

    def analyze_trajectory(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
        """Comprehensive trajectory analysis."""

        if len(states) == 0:
            return {}

        analysis = {
            'lap_time': self.metrics_calc.calculate_lap_time(states),
            'path_length': self.metrics_calc.calculate_path_length(states),
            'average_speed': self.metrics_calc.calculate_average_speed(states),
            'progress_rate': self.metrics_calc.calculate_progress_rate(states),
        }

        if len(actions) > 0:
            analysis['steering_smoothness'] = self.metrics_calc.calculate_steering_smoothness(actions)

        # Calculate additional metrics
        if states.shape[1] >= 3:  # x, y, theta
            # Calculate direction changes (curvature approximation)
            if len(states) > 2:
                angles = states[:, 2]  # theta
                angle_changes = np.abs(np.diff(angles))
                analysis['path_curvature'] = np.mean(angle_changes)

        return analysis

    def plot_trajectory(self, states: np.ndarray, save_path: Optional[Path] = None) -> None:
        """Plot vehicle trajectory."""

        if len(states) < 2:
            logger.warning("Not enough data points to plot trajectory")
            return

        plt.figure(figsize=(10, 8))

        # Extract x, y positions
        x = states[:, 0]
        y = states[:, 1]

        # Plot trajectory with color gradient
        plt.plot(x, y, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        plt.scatter(x[0], y[0], color='green', s=100, label='Start', zorder=5)
        plt.scatter(x[-1], y[-1], color='red', s=100, label='End', zorder=5)

        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Vehicle Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trajectory plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
