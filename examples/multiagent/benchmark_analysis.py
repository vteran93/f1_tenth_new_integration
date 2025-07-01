#!/usr/bin/env python3
"""
Script para ejecutar análisis de benchmark en experimentos de f1tenth_gym
usando las herramientas de f1tenth_benchmarks.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import subprocess
import importlib.util

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ruta al repositorio f1tenth_benchmarks
# Path to the local f1tenth_benchmarks module (now integrated)
# BENCHMARKS_PATH = Path(__file__).parent.parent / "f1tenth_benchmarks"


def add_benchmarks_to_path():
    """Use the local f1tenth_benchmarks module."""
    try:
        # Use the local integrated module
        import f1tenth_benchmarks
        logger.info("Successfully imported local f1tenth_benchmarks module")
        return True
    except ImportError as e:
        logger.error(f"Could not import local f1tenth_benchmarks: {e}")
        return False


def check_dependencies():
    """Verificar que las dependencias necesarias estén disponibles."""
    required_modules = [
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'trajectory_planning_helpers'
    ]

    missing_modules = []
    for module in required_modules:
        spec = importlib.util.find_spec(module)
        if spec is None:
            missing_modules.append(module)

    if missing_modules:
        logger.error(f"Missing required modules: {missing_modules}")
        logger.info("Install with: pip install matplotlib numpy pandas scipy trajectory-planning-helpers")
        return False

    return True


def run_trajectory_analysis(experiment_name: str, trial_id: str, storage_path: str = "./ray_results"):
    """Ejecutar análisis de trayectoria para un experimento específico."""
    try:
        from f1tenth_benchmarks.data_tools.plot_trajectory_analysis import plot_trajectory_analysis

        logger.info(f"Running trajectory analysis for {experiment_name}, trial {trial_id}")

        # Cambiar al directorio de logs para que las rutas funcionen correctamente
        original_cwd = os.getcwd()
        logs_dir = Path(storage_path) / "Logs"

        if logs_dir.exists():
            os.chdir(logs_dir)
            plot_trajectory_analysis(experiment_name, trial_id)
            logger.info(f"Trajectory analysis completed for {trial_id}")
        else:
            logger.warning(f"Logs directory not found: {logs_dir}")

        os.chdir(original_cwd)

    except Exception as e:
        logger.error(f"Error running trajectory analysis: {e}")
        return False

    return True


def run_tracking_accuracy(experiment_name: str, trial_id: str, storage_path: str = "./ray_results"):
    """Calcular precisión del seguimiento de trayectoria."""
    try:
        from f1tenth_benchmarks.data_tools.plot_raceline_tracking import calculate_tracking_accuracy, plot_raceline_tracking

        logger.info(f"Calculating tracking accuracy for {experiment_name}, trial {trial_id}")

        original_cwd = os.getcwd()
        logs_dir = Path(storage_path) / "Logs"

        if logs_dir.exists():
            os.chdir(logs_dir)

            # Calcular precisión del seguimiento
            calculate_tracking_accuracy(experiment_name, trial_id, centerline=True)

            # Generar plots de seguimiento
            plot_raceline_tracking(experiment_name, trial_id)

            logger.info(f"Tracking accuracy analysis completed for {trial_id}")
        else:
            logger.warning(f"Logs directory not found: {logs_dir}")

        os.chdir(original_cwd)

    except Exception as e:
        logger.error(f"Error calculating tracking accuracy: {e}")
        return False

    return True


def run_drl_training_plots(experiment_name: str, trial_id: str, storage_path: str = "./ray_results"):
    """Generar plots de entrenamiento DRL."""
    try:
        from f1tenth_benchmarks.data_tools.specific_plotting.plot_drl_training import plot_drl_training

        logger.info(f"Generating DRL training plots for {experiment_name}, trial {trial_id}")

        original_cwd = os.getcwd()
        logs_dir = Path(storage_path) / "Logs"

        if logs_dir.exists():
            os.chdir(logs_dir)
            plot_drl_training(experiment_name, trial_id)
            logger.info(f"DRL training plots completed for {trial_id}")
        else:
            logger.warning(f"Logs directory not found: {logs_dir}")

        os.chdir(original_cwd)

    except Exception as e:
        logger.error(f"Error generating DRL training plots: {e}")
        return False

    return True


def generate_benchmark_comparison(storage_path: str = "./ray_results"):
    """Generar comparación de benchmark usando los notebooks."""
    try:
        logger.info("Generating benchmark comparison plots")

        # Buscar notebooks en f1tenth_benchmarks
        # For now, disable notebook execution as the external notebooks are not available
        # notebook_dir = Path(__file__).parent.parent / "f1tenth_benchmarks" / "notebooks"
        logger.info("Notebook execution disabled - using local f1tenth_benchmarks module")

        notebooks = [
            "benchmark_result_plots.ipynb",
            "drl_result_plots.ipynb"
        ]

        results_generated = False

        for notebook in notebooks:
            notebook_path = notebook_dir / notebook
            if notebook_path.exists():
                try:
                    # Ejecutar notebook usando jupyter
                    cmd = [
                        "jupyter", "nbconvert",
                        "--to", "notebook",
                        "--execute",
                        "--inplace",
                        str(notebook_path)
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(notebook_dir))

                    if result.returncode == 0:
                        logger.info(f"Successfully executed {notebook}")
                        results_generated = True
                    else:
                        logger.warning(f"Failed to execute {notebook}: {result.stderr}")

                except Exception as e:
                    logger.warning(f"Error executing {notebook}: {e}")
            else:
                logger.warning(f"Notebook not found: {notebook_path}")

        if results_generated:
            logger.info("Benchmark comparison plots generated successfully")
        else:
            logger.warning("No benchmark plots were generated")

    except Exception as e:
        logger.error(f"Error generating benchmark comparison: {e}")
        return False

    return True


def run_full_analysis(experiment_name: str, storage_path: str = "./ray_results"):
    """Ejecutar análisis completo para un experimento."""
    if not add_benchmarks_to_path():
        return False

    if not check_dependencies():
        return False

    logger.info(f"Starting full analysis for experiment: {experiment_name}")

    # Buscar todos los trials en el experimento
    logs_dir = Path(storage_path) / "Logs" / experiment_name

    if not logs_dir.exists():
        logger.error(f"Experiment logs not found: {logs_dir}")
        return False

    # Buscar directorios RawData_*
    trial_dirs = list(logs_dir.glob("RawData_*"))

    if not trial_dirs:
        logger.warning(f"No trial data found in {logs_dir}")
        return False

    logger.info(f"Found {len(trial_dirs)} trials to analyze")

    success_count = 0
    total_trials = len(trial_dirs)

    for trial_dir in trial_dirs:
        trial_id = trial_dir.name.replace("RawData_", "")
        logger.info(f"Analyzing trial: {trial_id}")

        trial_success = True

        # Ejecutar análisis de trayectoria
        if not run_trajectory_analysis(experiment_name, trial_id, storage_path):
            trial_success = False

        # Ejecutar análisis de seguimiento
        if not run_tracking_accuracy(experiment_name, trial_id, storage_path):
            trial_success = False

        # Generar plots de entrenamiento DRL
        if not run_drl_training_plots(experiment_name, trial_id, storage_path):
            trial_success = False

        if trial_success:
            success_count += 1
            logger.info(f"Trial {trial_id} analyzed successfully")
        else:
            logger.warning(f"Some analyses failed for trial {trial_id}")

    logger.info(f"Analysis completed: {success_count}/{total_trials} trials processed successfully")

    # Generar comparación de benchmark
    generate_benchmark_comparison(storage_path)

    return success_count > 0


def main():
    parser = argparse.ArgumentParser(description="Run f1tenth_benchmarks analysis on f1tenth_gym experiments")
    parser.add_argument("experiment_name", help="Name of the experiment to analyze")
    parser.add_argument("--storage-path", default="./ray_results", help="Path to ray results storage")
    parser.add_argument("--trial-id", help="Specific trial to analyze (analyze all if not specified)")
    parser.add_argument("--analysis-type", choices=["full", "trajectory", "tracking", "training"],
                        default="full", help="Type of analysis to run")

    args = parser.parse_args()

    if not add_benchmarks_to_path():
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    if args.trial_id:
        # Analizar un trial específico
        logger.info(f"Analyzing specific trial: {args.trial_id}")

        success = True
        if args.analysis_type in ["full", "trajectory"]:
            success &= run_trajectory_analysis(args.experiment_name, args.trial_id, args.storage_path)

        if args.analysis_type in ["full", "tracking"]:
            success &= run_tracking_accuracy(args.experiment_name, args.trial_id, args.storage_path)

        if args.analysis_type in ["full", "training"]:
            success &= run_drl_training_plots(args.experiment_name, args.trial_id, args.storage_path)

        if not success:
            sys.exit(1)
    else:
        # Analizar todo el experimento
        if not run_full_analysis(args.experiment_name, args.storage_path):
            sys.exit(1)

    logger.info("Analysis completed successfully!")


if __name__ == "__main__":
    main()
