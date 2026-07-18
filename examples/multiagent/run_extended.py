"""
Extended evaluation module for F1TENTH multi-agent training.
Implements mass evaluation functionality following SOLID Open/Closed principle.
"""

import argparse
import sys
import json
import time
import numpy as np
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


def _make_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-safe format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        # Try to convert to string as fallback
        return str(obj)


def _episode_better(a, b):
    """
    Devuelve True si episodio a es mejor que b (multi-criterio).
    Prioridad 1: mayor max_lap_progress (progreso máximo en la pista)
    Prioridad 2: mayor retorno
    Prioridad 3: menos colisiones
    Prioridad 4: menor longitud
    """
    if b is None:
        return True
    # Prioridad 1: mayor max_lap_progress
    if a["max_lap_progress"] != b["max_lap_progress"]:
        return a["max_lap_progress"] > b["max_lap_progress"]
    # Prioridad 2: mayor retorno
    if a["return"] != b["return"]:
        return a["return"] > b["return"]
    # Prioridad 3: menos colisiones
    if a["collisions"] != b["collisions"]:
        return a["collisions"] < b["collisions"]
    # Prioridad 4: menor longitud
    return a["length"] < b["length"]


def run_mass_evaluation(config, trial_name=None):
    """
    Evaluación masiva: corre N episodios (p.ej. 10_000), guarda métricas agregadas
    y la mejor trayectoria (JSON y MP4 si es posible).
    Config espera:
      config["evaluation"]["mass_episodes"] = 10000
      config["evaluation"]["record_best"] = True
      (opcional) config["evaluation"]["seed_base"] = 0
    """
    logger.info("Starting MASS evaluation")
    experiment_path = get_experiment_path(config["name"], config["storage_path"])
    analysis = ExperimentAnalysis(experiment_path)

    # Selección de checkpoint
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
        logger.info(f"Found {len(filtered)} trial(s) matching '{trial_name}'")
    
    best_checkpoint = get_best_checkpoint(analysis)
    if not best_checkpoint:
        logger.error("No checkpoint found. Train model first")
        return
    logger.info(f"Loading checkpoint: {best_checkpoint}")
    algo = Algorithm.from_checkpoint(best_checkpoint)

    # Determinar si el modelo se entrenó con una política compartida
    shared_policy_cfg = config['training'].get('shared_policy', True)
    candidate_policy = "shared_policy" if shared_policy_cfg else "agent_0"
    logger.info(f"Using policy configuration: shared_policy={shared_policy_cfg}")

    # Parámetros de evaluación
    eval_conf = config.get("evaluation", {})
    total_target_episodes = int(eval_conf.get("mass_episodes", 10000))
    record_best = bool(eval_conf.get("record_best", True))
    seed_base = int(eval_conf.get("seed_base", 0))
    render_best = bool(eval_conf.get("render_best", False))  # solo para re-ejecutar la mejor

    # Crear carpeta de eval
    out_dir = Path(experiment_path) / "eval_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(seed_base, seed_base + total_target_episodes))
    (out_dir / "seeds.json").write_text(json.dumps(seeds, indent=2))

    # Crear env sin render para velocidad
    env, _ = create_env(config, render_mode=None)

    # Acumuladores
    returns, lengths, max_lap_progresses = [], [], []
    success_count, collision_count = 0, 0
    total_eps = 0
    best_ep = None  # dict con resumen
    best_traj = None  # lista de pasos

    try:
        # Bucle principal
        for seed in seeds:
            # reset con seed (si tu env no soporta seed, caemos a reset() normal)
            try:
                obs, info = env.reset(seed=seed)
            except TypeError:
                obs, info = env.reset()

            ep_return = 0.0
            ep_len = 0
            ep_max_lap_progress = 0.0  # Track maximum lap progress achieved
            terminated = {"__all__": False}
            truncated = {"__all__": False}
            traj = [] if record_best else None

            while not (terminated.get("__all__", False) or truncated.get("__all__", False)):
                actions = {}
                for agent_id, agent_obs in obs.items():
                    # Usar el ID de política correcto según la configuración del experimento
                    policy_id = "shared_policy" if shared_policy_cfg else agent_id
                    
                    action = algo.compute_single_action(
                        observation=agent_obs,
                        policy_id=policy_id,
                        explore=False
                    )
                    actions[agent_id] = action

                next_obs, reward, terminated, truncated, info = env.step(actions)

                # suma de reward multiagente
                if isinstance(reward, dict):
                    step_rew = float(np.sum(list(reward.values())))
                else:
                    step_rew = float(reward)
                ep_return += step_rew
                ep_len += 1

                # Extract and track maximum lap progress - use same method as callbacks
                try:
                    # Access the underlying F110 environment to calculate lap progress (same as MultipleAgentCallbacks)
                    f110_env = getattr(env, 'env', env)  # Unwrap if wrapped
                    if hasattr(f110_env, 'num_agents') and hasattr(f110_env, 'track'):
                        for i in range(f110_env.num_agents):
                            current_s, _ = f110_env.track.centerline.spline.calc_arclength_inaccurate(
                                f110_env.poses_x[i], f110_env.poses_y[i]
                            )
                            current_progress = current_s / f110_env.track.centerline.spline.s[-1]
                            ep_max_lap_progress = max(ep_max_lap_progress, current_progress)
                except Exception:
                    # If lap progress calculation fails, continue without it
                    pass

                if record_best and traj is not None:
                    traj.append({
                        "obs": _make_json_serializable(obs),
                        "act": _make_json_serializable(actions),
                        "rew": float(step_rew),
                        "info": _make_json_serializable(info),
                    })

                obs = next_obs

                # Sanidad numérica
                if not np.isfinite(ep_return):
                    logger.error("Non-finite return detected. Aborting episode.")
                    break

            # Éxito/colisión desde info (si tu env lo expone)
            ep_success = False
            ep_collision = False
            if isinstance(info, dict) and len(info) > 0:
                # success = True si todos marcan success en el último paso
                ep_success = all(isinstance(v, dict) and v.get("success", False) for v in info.values() if isinstance(v, dict))
                # collision = True si cualquiera marcó collision alguna vez; intentamos recogerlo de info final
                ep_collision = any(isinstance(v, dict) and v.get("collision", False) for v in info.values() if isinstance(v, dict))

            returns.append(ep_return)
            lengths.append(ep_len)
            max_lap_progresses.append(ep_max_lap_progress)
            success_count += int(ep_success)
            collision_count += int(ep_collision)
            total_eps += 1

            # Actualizar mejor episodio
            candidate = {
                "return": ep_return,
                "length": ep_len,
                "collisions": int(ep_collision),
                "max_lap_progress": ep_max_lap_progress,
                "seed": seed
            }
            if _episode_better(candidate, best_ep):
                best_ep = candidate
                best_traj = traj

            if total_eps % 500 == 0:
                logger.info(f"[Eval] {total_eps}/{total_target_episodes} episodes | "
                            f"R_mean={np.mean(returns):.2f}±{np.std(returns):.2f} | "
                            f"len_mean={np.mean(lengths):.1f} | "
                            f"max_lap_prog={np.mean(max_lap_progresses):.3f}±{np.std(max_lap_progresses):.3f} | "
                            f"succ={success_count} coll={collision_count}")

            if total_eps >= total_target_episodes:
                break

    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            algo.stop()
        except Exception:
            pass

    # Resumen y persistencia
    ret = np.array(returns, dtype=float)
    lng = np.array(lengths, dtype=int)
    mlp = np.array(max_lap_progresses, dtype=float)
    summary = {
        "episodes": int(total_eps),
        "return_mean": float(ret.mean()) if ret.size else None,
        "return_std": float(ret.std(ddof=1)) if ret.size > 1 else 0.0,
        "length_mean": float(lng.mean()) if lng.size else None,
        "length_std": float(lng.std(ddof=1)) if lng.size > 1 else 0.0,
        "max_lap_progress_mean": float(mlp.mean()) if mlp.size else None,
        "max_lap_progress_std": float(mlp.std(ddof=1)) if mlp.size > 1 else 0.0,
        "successes": int(success_count),
        "collisions": int(collision_count),
        "best": best_ep,
        "timestamp": int(time.time()),
    }
    (out_dir / "mass_eval_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Mass eval summary saved to: {out_dir/'mass_eval_summary.json'}")

    if record_best and best_traj is not None and best_ep is not None:
        # Guarda trayectoria cruda
        safe_episode_data = {
            "meta": _make_json_serializable(best_ep),
            "steps": _make_json_serializable(best_traj)
        }
        (out_dir / "best_episode.json").write_text(json.dumps(safe_episode_data, indent=2))
        logger.info(f"Best episode JSON saved (seed={best_ep['seed']})")
        logger.info(f"Best episode stats: Max_Lap_Progress={best_ep['max_lap_progress']:.3f}, Return={best_ep['return']:.2f}, Length={best_ep['length']}, Collisions={best_ep['collisions']}")

        # Intentar render MP4 de la mejor seed si está habilitado
        if render_best:
            try:
                logger.info("Rendering best episode video...")
                _render_best_episode(config, best_checkpoint, best_ep['seed'],
                                     candidate_policy, shared_policy_cfg, out_dir)
            except Exception as e:
                logger.warning(f"Could not render video of best episode: {e}")

    return summary


def _render_best_episode(config, checkpoint_path, best_seed, policy_id, shared_policy, out_dir):
    """Renderiza el mejor episodio en video MP4."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        logger.warning("imageio not available for video rendering")
        return

    # Crear nuevo algoritmo y env con render
    algo = Algorithm.from_checkpoint(checkpoint_path)
    env, _ = create_env(config, render_mode="rgb_array")
    
    frames = []
    
    try:
        obs, info = env.reset(seed=best_seed)
    except TypeError:
        obs, info = env.reset()
    
    terminated = {"__all__": False}
    truncated = {"__all__": False}
    
    try:
        while not (terminated.get("__all__", False) or truncated.get("__all__", False)):
            # Capturar frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Ejecutar acción
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_used = "shared_policy" if shared_policy else agent_id
                action = algo.compute_single_action(
                    observation=agent_obs,
                    policy_id=policy_used,
                    explore=False
                )
                actions[agent_id] = action
            
            obs, reward, terminated, truncated, info = env.step(actions)
        
        # Guardar video
        if frames:
            video_path = out_dir / "best_episode.mp4"
            imageio.mimsave(str(video_path), frames, fps=30)
            logger.info(f"Best episode video saved to: {video_path}")
    
    finally:
        env.close()
        algo.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog=Path(__file__).stem,
        description='Extended F1TENTH multi-agent evaluation with mass evaluation capability.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
            Examples:
            python run_extended.py mass-eval --experiment oval_small_PPO_Individual_Policy_ProgressRewardAdvanced
            python run_extended.py mass-eval --experiment oval_small_PPO_Individual_Policy_ProgressRewardAdvanced --trial trial_123
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

    # Mass eval parser
    mass_eval_parser = subparsers.add_parser('mass-eval', help='Run mass evaluation (10,000 episodes)')
    mass_eval_parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment to evaluate')
    mass_eval_parser.add_argument('--trial', type=str, default=None,
                                  help='Specific trial to evaluate (optional, uses best trial if not specified)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    config_path = args.config.resolve()
    config_dir = config_path.parent
    config_data = load_config(config_path)
    experiments = config_data.get('experiments', [])

    # Get num_cpus from config, default to 16 if not specified
    num_cpus = config_data.get('num_cpus', 16)
    init_ray(num_cpus=num_cpus)

    if args.command == 'mass-eval':
        experiment = find_experiment(experiments, args.experiment)
        cfg = setup_experiment_config(experiment, config_dir)
        logger.info(f"Mass evaluating: {cfg['name']}" + (f" (trial={args.trial})" if args.trial else ""))
        run_mass_evaluation(cfg, args.trial)
