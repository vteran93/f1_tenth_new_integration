"""
Script para renderizar y visualizar el mejor episodio encontrado en la evaluación masiva.
"""

import json
import sys
from pathlib import Path

# Import original functionality
from run import (
    load_config, init_ray, get_logger, suppress_warnings,
    get_experiment_path, get_best_checkpoint, create_env,
    setup_experiment_config, find_experiment
)
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.analysis import ExperimentAnalysis

suppress_warnings()
logger = get_logger(__name__)


def render_best_episode(config, trial_name=None):
    """
    Renderiza el mejor episodio encontrado en la evaluación masiva.
    """
    logger.info("Starting BEST EPISODE rendering")
    experiment_path = get_experiment_path(config["name"], config["storage_path"])
    
    # Buscar el archivo de resumen de evaluación masiva
    eval_runs_dir = Path(experiment_path) / "eval_runs"
    summary_file = eval_runs_dir / "mass_eval_summary.json"
    
    if not summary_file.exists():
        logger.error(f"No mass evaluation summary found at {summary_file}")
        logger.info("Please run mass evaluation first with: python run_extended.py mass-eval --experiment <name>")
        return
    
    # Cargar el resumen y obtener la mejor seed
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    best_episode = summary.get('best', {})
    best_seed = best_episode.get('seed')
    best_return = best_episode.get('return')
    best_length = best_episode.get('length')
    best_collisions = best_episode.get('collisions')
    best_max_lap_progress = best_episode.get('max_lap_progress', 0.0)
    
    if best_seed is None:
        logger.error("No best episode seed found in summary")
        return
    
    logger.info(f"Rendering best episode:")
    logger.info(f"  Seed: {best_seed}")
    logger.info(f"  Max Lap Progress: {best_max_lap_progress:.3f}")
    logger.info(f"  Return: {best_return:.2f}")
    logger.info(f"  Length: {best_length} steps")
    logger.info(f"  Collisions: {best_collisions}")
    
    # Cargar el checkpoint
    analysis = ExperimentAnalysis(experiment_path)
    if trial_name:
        filtered = []
        for t in analysis.trials:
            path = getattr(t, "local_dir", "")
            if (trial_name in getattr(t, "trial_id", "")) or (trial_name in path):
                filtered.append(t)
        if not filtered:
            logger.error(f"No trials found matching: {trial_name}")
            return
        analysis.trials = filtered
    
    best_checkpoint = get_best_checkpoint(analysis)
    if not best_checkpoint:
        logger.error("No checkpoint found")
        return
    
    logger.info(f"Loading checkpoint: {best_checkpoint}")
    algo = Algorithm.from_checkpoint(best_checkpoint)
    
    # Crear entorno con renderizado
    env, _ = create_env(config, render_mode="human")
    
    # Determinar configuración de política
    shared_policy_cfg = config['training'].get('shared_policy', True)
    
    try:
        # Reset con la mejor seed
        try:
            obs, info = env.reset(seed=best_seed)
        except TypeError:
            obs, info = env.reset()
            logger.warning(f"Environment doesn't support seed parameter. Cannot guarantee reproduction of seed {best_seed}")
        
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        step_count = 0
        total_reward = 0
        
        logger.info("Starting best episode visualization...")
        logger.info("Press Ctrl+C to stop early")
        
        while not (terminated.get("__all__", False) or truncated.get("__all__", False)):
            # Renderizar
            env.render()
            
            # Ejecutar acciones
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = "shared_policy" if shared_policy_cfg else agent_id
                action = algo.compute_single_action(
                    observation=agent_obs,
                    policy_id=policy_id,
                    explore=False
                )
                actions[agent_id] = action
            
            obs, reward, terminated, truncated, info = env.step(actions)
            
            # Acumular métricas
            if isinstance(reward, dict):
                step_reward = sum(reward.values())
            else:
                step_reward = reward
            total_reward += step_reward
            step_count += 1
            
            # Mostrar progreso cada 50 pasos
            if step_count % 50 == 0:
                logger.info(f"Step {step_count}, Total reward: {total_reward:.2f}")
        
        logger.info(f"Episode completed!")
        logger.info(f"  Final step count: {step_count}")
        logger.info(f"  Final total reward: {total_reward:.2f}")
        logger.info(f"  Expected reward: {best_return:.2f}")
        
        if abs(total_reward - best_return) > 1.0:
            logger.warning(f"Reward mismatch! Expected {best_return:.2f}, got {total_reward:.2f}")
            logger.warning("This might be due to environment seed not being supported")
    
    except KeyboardInterrupt:
        logger.info("Rendering stopped by user")
    
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            algo.stop()
        except Exception:
            pass


def create_video_best_episode(config, trial_name=None):
    """
    Crea un video MP4 del mejor episodio.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        logger.error("imageio not available for video creation")
        logger.info("Install with: pip install imageio[ffmpeg]")
        return
    
    logger.info("Creating VIDEO of best episode")
    experiment_path = get_experiment_path(config["name"], config["storage_path"])
    
    # Buscar el archivo de resumen
    eval_runs_dir = Path(experiment_path) / "eval_runs"
    summary_file = eval_runs_dir / "mass_eval_summary.json"
    
    if not summary_file.exists():
        logger.error(f"No mass evaluation summary found at {summary_file}")
        return
    
    # Cargar el resumen y obtener la mejor seed
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    best_episode = summary.get('best', {})
    best_seed = best_episode.get('seed')
    
    if best_seed is None:
        logger.error("No best episode seed found in summary")
        return
    
    # Cargar el checkpoint
    analysis = ExperimentAnalysis(experiment_path)
    if trial_name:
        filtered = []
        for t in analysis.trials:
            path = getattr(t, "local_dir", "")
            if (trial_name in getattr(t, "trial_id", "")) or (trial_name in path):
                filtered.append(t)
        if not filtered:
            logger.error(f"No trials found matching: {trial_name}")
            return
        analysis.trials = filtered
    
    best_checkpoint = get_best_checkpoint(analysis)
    if not best_checkpoint:
        logger.error("No checkpoint found")
        return
    
    logger.info(f"Loading checkpoint: {best_checkpoint}")
    algo = Algorithm.from_checkpoint(best_checkpoint)
    
    # Crear entorno con renderizado RGB
    env, _ = create_env(config, render_mode="rgb_array")
    
    shared_policy_cfg = config['training'].get('shared_policy', True)
    frames = []
    
    try:
        # Reset con la mejor seed
        try:
            obs, info = env.reset(seed=best_seed)
        except TypeError:
            obs, info = env.reset()
        
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        step_count = 0
        
        logger.info(f"Recording video of best episode (seed {best_seed})...")
        
        while not (terminated.get("__all__", False) or truncated.get("__all__", False)):
            # Capturar frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Ejecutar acciones
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = "shared_policy" if shared_policy_cfg else agent_id
                action = algo.compute_single_action(
                    observation=agent_obs,
                    policy_id=policy_id,
                    explore=False
                )
                actions[agent_id] = action
            
            obs, reward, terminated, truncated, info = env.step(actions)
            step_count += 1
            
            if step_count % 50 == 0:
                logger.info(f"Recorded {step_count} frames...")
        
        # Guardar video
        if frames:
            video_path = eval_runs_dir / "best_episode.mp4"
            logger.info(f"Saving video with {len(frames)} frames...")
            imageio.mimsave(str(video_path), frames, fps=30)
            logger.info(f"Video saved to: {video_path}")
            logger.info(f"Video duration: {len(frames)/30:.1f} seconds")
        else:
            logger.warning("No frames captured for video")
    
    except Exception as e:
        logger.error(f"Error creating video: {e}")
    
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            algo.stop()
        except Exception:
            pass


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Render or create video of the best episode from mass evaluation'
    )
    parser.add_argument('--config', type=Path, default=Path('configs/experiments.yaml'))
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--trial', type=str, default=None)
    parser.add_argument('--mode', choices=['render', 'video'], default='render',
                        help='render: show live visualization, video: create MP4 file')
    
    args = parser.parse_args()
    
    config_path = args.config.resolve()
    config_dir = config_path.parent
    config_data = load_config(config_path)
    experiments = config_data.get('experiments', [])
    
    num_cpus = config_data.get('num_cpus', 16)
    init_ray(num_cpus=num_cpus)
    
    experiment = find_experiment(experiments, args.experiment)
    cfg = setup_experiment_config(experiment, config_dir)
    
    if args.mode == 'render':
        render_best_episode(cfg, args.trial)
    else:
        create_video_best_episode(cfg, args.trial)