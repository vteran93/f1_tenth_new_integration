# Reinforcement Learning Evaluation Instructions

Este documento consolida las directrices para pruebas y evaluación de agentes en entornos de **Reinforcement Learning (RL)**, junto con la extensión de evaluación masiva (10.000 episodios) y grabación de la mejor ejecución.

---

## 1. Objetivo

Asegurar que las pruebas de los agentes RL sean **reproducibles**, **medibles**, y **automatizables**. Este estándar aplica tanto para entornos tipo Gym/PettingZoo como para configuraciones multiagente en RLlib o Stable-Baselines3.

---

## 2. Tipos de Pruebas en RL

### **a) Unitarias (Infraestructura y Entorno)**

* Verifican API: `reset()`, `step()`, `reward()`, `done`, `observation_space`, `action_space`.
* Asegura tipos de datos válidos y compatibilidad con `gym.Env`.

```python
def test_step_output_shape(env):
    obs, reward, done, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
```

### **b) Integración (Entorno ↔ Agente)**

* Verifica interacción correcta: no fugas de memoria ni `NaN` en rewards.
* Ejecuta un “smoke test” de 1000 pasos.

### **c) Regresión de Desempeño**

* Carga un checkpoint previo y compara métricas clave:

  * `mean_reward`, `reward_std`, `collision_rate`, `success_rate`.
* Falla la prueba si el nuevo modelo empeora significativamente.

### **d) Estabilidad Numérica**

* Detecta explosiones de gradientes o `NaN`.
* Implementa callbacks o hooks para abortar runs inestables.

### **e) Robustez y Transferencia**

* Evalúa bajo distintas semillas, ruido en sensores o variaciones ambientales.
* Mide el *performance drop* entre entornos de train y test.

---

## 3. Estrategia de Pruebas

| Nivel         | Herramienta        | Propósito                      |
| ------------- | ------------------ | ------------------------------ |
| Unit          | Pytest             | Validar entorno                |
| Integración   | SB3/RLlib DummyEnv | Verificar ciclo agent-env      |
| Regresión     | W&B / scripts eval | Comparar performance histórico |
| Estabilidad   | Callbacks          | Detectar NaN/divergencias      |
| Robustez      | Random seeds/noise | Asegurar consistencia          |
| Transferencia | Multi-env          | Validar generalización         |

---

## 4. Evaluación Estándar (`run_evaluation` mejorado)

Incluye:

* Semillas reproducibles (`seeds=[0,1,2]`).
* Parada correcta (`terminated` o `truncated`).
* Métricas agregadas: retorno, longitud, éxitos, colisiones.
* Política detectada automáticamente.
* Persistencia: `eval_summary.json` dentro del experimento.
* Limpieza de recursos (`env.close()`, `algo.stop()`).

```python
def run_evaluation(config, trial_name=None):
    # ... (código completo disponible en patch)
    summary = {
        "episodes": total_episodes,
        "return_mean": np.mean(returns),
        "return_std": np.std(returns),
        "successes": successes,
        "collisions": collisions,
    }
    Path(experiment_path, 'eval_summary.json').write_text(json.dumps(summary, indent=2))
```

### 📈 Resultados esperados

* `eval_summary.json` con métricas promedio ± desviación estándar.
* Evaluación reproducible, sin `NaN`, sin loops infinitos.

---

## 5. Evaluación Masiva (`run_mass_evaluation`)

Ejecuta **10.000 episodios** distribuidos por semillas, sin render, grabando la **mejor carrera**.

### **Características:**

* Paraleliza evaluaciones por semillas.
* Guarda métricas globales (`mass_eval_summary.json`).
* Criterio de mejor episodio: máximo retorno, mínima colisión, menor longitud.
* Guarda trayectoria detallada (`best_episode.json`) y video opcional (`best_episode.mp4`).

```yaml
# Config YAML
 evaluation:
   episodes: 10000
   record_best: true
   seed_base: 0
   render_best: true  # opcional (requiere soporte rgb_array)
```

```python
def run_mass_evaluation(config, trial_name=None):
    # Bucle masivo con seeds, sin render
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        # ejecutar política
        while not (terminated['__all__'] or truncated['__all__']):
            action = algo.compute_single_action(...)
            obs, reward, terminated, truncated, info = env.step(action)
            # guardar trayectoria si record_best=True
```

### **Output Files:**

```
<storage>/<name>/<date>/eval_runs/
├─ seeds.json
├─ mass_eval_summary.json
├─ best_episode.json
└─ best_episode.mp4 (opcional)
```

---

## 6. Causas comunes de divergencia entre train y eval

* `explore=True` durante training y `explore=False` en test (política diferente).
* Seeds distintas o no fijadas.
* Ignorar `truncated` → métricas sesgadas.
* Políticas multi-agent mal asignadas (`shared_policy` ausente).
* Estadísticas de normalización no sincronizadas.
* Config del entorno distinta (parámetros, ruido, física).
* Render activo (altera tiempo de simulación).

---

## 7. Extensiones recomendadas

* **Paralelización:** usar `evaluation_num_env_runners` y `algo.evaluate()` de RLlib.
* **Tests automáticos:** comparar `return_mean` con baseline, fallar si cae > Xσ.
* **Filtros personalizados:** definir "mejor" como sin colisión + máximo retorno.

---

## 8. Estructura de CI recomendada

```bash
pytest tests/test_env.py
python run.py eval
python run.py mass-eval
```

---

### ✅ Resultado esperado general

* Evaluaciones reproducibles con seeds.
* Métricas consolidadas.
* Mejor episodio grabado.
* Scripts integrados en CI/CD para regresión de performance.


#### 

Ejemplo de run mass evaluation


```python
def _episode_better(a, b):
    """Devuelve True si episodio a es mejor que b (multi-criterio)."""
    if b is None:
        return True
    # Prioridad 1: mayor retorno
    if a["return"] != b["return"]:
        return a["return"] > b["return"]
    # Prioridad 2: menos colisiones
    if a["collisions"] != b["collisions"]:
        return a["collisions"] < b["collisions"]
    # Prioridad 3: menor longitud
    return a["length"] < b["length"]


def run_mass_evaluation(config, trial_name=None):
    """
    Evaluación masiva: corre N episodios (p.ej. 10_000), guarda métricas agregadas
    y la mejor trayectoria (JSON y MP4 si es posible).
    Config espera:
      config["evaluation"]["episodes"] = 10000
      config["evaluation"]["record_best"] = True
      (opcional) config["evaluation"]["seed_base"] = 0
    """
    import json, time, numpy as np
    from pathlib import Path

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

    # Política a usar
    policy_map = list(algo.workers.local_worker().policy_map.keys())
    shared_policy_cfg = config['training'].get('shared_policy', True)
    candidate_policy = "shared_policy" if shared_policy_cfg and "shared_policy" in policy_map else policy_map[0]
    logger.info(f"Using policy '{candidate_policy}' for evaluation.")

    # Parámetros de evaluación
    eval_conf = config.get("evaluation", {})
    total_target_episodes = int(eval_conf.get("episodes", 10000))
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
    returns, lengths = [], []
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
            terminated = {"__all__": False}
            truncated = {"__all__": False}
            traj = [] if record_best else None
            rnn_state = {}

            while not (terminated.get("__all__", False) or truncated.get("__all__", False)):
                actions = {}
                for agent_id, agent_obs in obs.items():
                    policy_id = candidate_policy if shared_policy_cfg else (agent_id if agent_id in policy_map else candidate_policy)
                    state = rnn_state.get(agent_id, None)
                    action, state_out, _ = algo.compute_single_action(
                        observation=agent_obs,
                        policy_id=policy_id,
                        explore=False,
                        state=state
                    )
                    actions[agent_id] = action
                    if state_out is not None:
                        rnn_state[agent_id] = state_out

                next_obs, reward, terminated, truncated, info = env.step(actions)

                # suma de reward multiagente
                if isinstance(reward, dict):
                    step_rew = float(np.sum(list(reward.values())))
                else:
                    step_rew = float(reward)
                ep_return += step_rew
                ep_len += 1

                if record_best:
                    traj.append({
                        "obs": {k: np.asarray(v).tolist() for k, v in obs.items()},
                        "act": {k: (np.asarray(v).tolist() if hasattr(v, "shape") else v) for k, v in actions.items()},
                        "rew": step_rew,
                        "info": info,
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
            success_count += int(ep_success)
            collision_count += int(ep_collision)
            total_eps += 1

            # Actualizar mejor episodio
            candidate = {"return": ep_return, "length": ep_len, "collisions": int(ep_collision), "seed": seed}
            if _episode_better(candidate, best_ep):
                best_ep = candidate
                best_traj = traj

            if total_eps % 500 == 0:
                logger.info(f"[Eval] {total_eps}/{total_target_episodes} episodes | "
                            f"R_mean={np.mean(returns):.2f}±{np.std(returns):.2f} | "
                            f"len_mean={np.mean(lengths):.1f} | succ={success_count} coll={collision_count}")

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
    import numpy as np, json, time
    ret = np.array(returns, dtype=float)
    lng = np.array(lengths, dtype=int)
    summary = {
        "episodes": int(total_eps),
        "return_mean": float(ret.mean()) if ret.size else None,
        "return_std": float(ret.std(ddof=1)) if ret.size > 1 else 0.0,
        "length_mean": float(lng.mean()) if lng.size else None,
        "length_std": float(lng.std(ddof=1)) if lng.size > 1 else 0.0,
        "successes": int(success_count),
        "collisions": int(collision_count),
        "best": best_ep,
        "timestamp": int(time.time()),
    }
    (out_dir / "mass_eval_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Mass eval summary saved to: {out_dir/'mass_eval_summary.json'}")

    if record_best and best_traj is not None:
        # Guarda trayectoria cruda
        (out_dir / "best_episode.json").write_text(json.dumps({
            "meta": best_ep, "steps": best_traj
        }, indent=2))
        logger.info(f"Best episode JSON saved (seed={best_ep['seed']})")

        # Intentar render MP4 de la mejor seed
        if render_best:
            try:
                import imageio.v2 as imageio
                # Reejecutar SOLO la best seed con render 'rgb_array'
                env2, _ = create_env(config, render_mode="rgb_array")
                try:
                    try:
                        obs, info = env2.reset(seed=best_ep["seed"])
                    except TypeError:
                        obs, info = env2.reset()
                    frames = []
                    terminated = {"__all__": False}
                    truncated = {"__all__": False}
                    rnn_state = {}
                    while not (terminated.get("__all__", False) or truncated.get("__all__", False)):
                        actions = {}
                        for agent_id, agent_obs in obs.items():
                            policy_id = candidate_policy if shared_policy_cfg else (agent_id if agent_id in policy_map else candidate_policy)
                            state = rnn_state.get(agent_id, None)
                            action, state_out, _ = algo.compute_single_action(
                                observation=agent_obs,
                                policy_id=policy_id,
                                explore=False,
                                state=state
                            )
                            actions[agent_id] = action
                            if state_out is not None:
                                rnn_state[agent_id] = state_out
                        obs, reward, terminated, truncated, info = env2.step(actions)
                        frame = env2.render()
                        if frame is not None:
                            frames.append(frame)
                    if frames:
                        imageio.mimsave(out_dir / "best_episode.mp4", frames, fps=30)
                        logger.info(f"Best episode video saved to: {out_dir/'best_episode.mp4'}")
                finally:
                    try: env2.close()
                    except Exception: pass
            except Exception as e:
                logger.warning(f"Could not render video of best episode: {e}")
```
## Mejoras de parada para run_evaluation


```python
def run_evaluation(config, trial_name=None):
    logger.info("Starting evaluation")
    experiment_path = get_experiment_path(config["name"], config["storage_path"])
    analysis = ExperimentAnalysis(experiment_path)

    # --- seleccionar checkpoint ---
    if trial_name:
        filtered = []
        for t in analysis.trials:
            path = getattr(t, "local_dir", "")
            if (trial_name in t.trial_id) or (trial_name in path):
                filtered.append(t)
        if not filtered:
            logger.error(f"No trials found matching: {trial_name}")
            logger.debug(f"Available trials: {[t.trial_id for t in analysis.trials]}")
            return
        analysis.trials = filtered
        logger.info(f"Found {len(filtered)} trial(s) matching '{trial_name}'")
    best_checkpoint = get_best_checkpoint(analysis)
    if not best_checkpoint:
        logger.error("No checkpoint found. Train model first")
        return
    logger.debug(f"Loading checkpoint: {best_checkpoint}")

    algo = Algorithm.from_checkpoint(best_checkpoint)

    # --- elegir policy id válida ---
    shared_policy_cfg = config['training'].get('shared_policy', True)
    candidate_policy = "shared_policy" if shared_policy_cfg else None
    available = list(algo.workers.local_worker().policy_map.keys())
    if candidate_policy not in available:
        # si no hay shared_policy, toma la primera
        candidate_policy = available[0]
        logger.warning(f"'shared_policy' not found. Using policy '{candidate_policy}' from {available}")

    # --- seeds y episodios ---
    eval_conf = config.get("evaluation", {})
    num_episodes = int(eval_conf.get("episodes", 5))
    seeds = eval_conf.get("seeds", [0, 1, 2])  # reproducible por defecto
    render = bool(eval_conf.get("render", True))

    # --- métricas acumuladas ---
    import numpy as np, json, time
    episode_returns = []
    episode_lengths = []
    collisions = 0
    successes = 0
    total_episodes = 0

    # --- crear env de evaluación con render opcional ---
    env, _ = create_env(config, render_mode="human" if render else None)

    try:
        for seed in seeds:
            # Gymnasium: reset(seed=seed) cuando aplique
            try:
                obs, info = env.reset(seed=seed)
            except TypeError:
                # si tu env no soporta seed en reset, setéalo por config interna si procede
                obs, info = env.reset()
            episodes_this_seed = 0

            while episodes_this_seed < max(1, num_episodes // max(1, len(seeds))):
                total_episodes += 1
                episodes_this_seed += 1
                ep_return = 0.0
                ep_len = 0
                terminated = {"__all__": False}
                truncated = {"__all__": False}

                # soporte (opcional) para políticas recurrentes
                rnn_state_per_agent = {}

                while not (terminated.get("__all__", False) or truncated.get("__all__", False)):
                    actions = {}
                    for agent_id, agent_obs in obs.items():
                        policy_id = candidate_policy if shared_policy_cfg else agent_id if agent_id in available else candidate_policy
                        state = rnn_state_per_agent.get(agent_id, None)
                        try:
                            action, state_out, _ = algo.compute_single_action(
                                observation=agent_obs,
                                policy_id=policy_id,
                                explore=False,
                                state=state
                            )
                        except Exception as e:
                            logger.error(f"compute_single_action failed for {agent_id}: {e}")
                            raise
                        actions[agent_id] = action
                        if state_out is not None:
                            rnn_state_per_agent[agent_id] = state_out

                    obs, reward, terminated, truncated, info = env.step(actions)
                    # reward puede ser dict multiagente o escalar agregado
                    if isinstance(reward, dict):
                        ep_return += float(np.sum(list(reward.values())))
                    else:
                        ep_return += float(reward)
                    ep_len += 1

                    if render:
                        try:
                            env.render()
                        except Exception:
                            pass

                    # señales de éxito/colisión si tu env las expone en info por agente
                    if isinstance(info, dict):
                        # marca éxito si TODOS reportan success True alguna vez
                        if all(isinstance(v, dict) and v.get("success", False) for v in info.values() if isinstance(v, dict)):
                            successes += 1
                        # cuenta colisiones si ALGUNO reporta collision True
                        if any(isinstance(v, dict) and v.get("collision", False) for v in info.values() if isinstance(v, dict)):
                            collisions += 1

                episode_returns.append(ep_return)
                episode_lengths.append(ep_len)

                # reset siguiente episodio con misma seed para consistencia intra-seed
                try:
                    obs, info = env.reset(seed=seed)
                except TypeError:
                    obs, info = env.reset()

    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            algo.stop()  # libera workers
        except Exception:
            pass

    # --- resumen y persistencia ---
    ret = np.array(episode_returns, dtype=float)
    lng = np.array(episode_lengths, dtype=int)
    summary = {
        "trials_path": str(experiment_path),
        "checkpoint": str(best_checkpoint),
        "policy_used": candidate_policy,
        "episodes": int(total_episodes),
        "seeds": list(seeds),
        "return_mean": float(ret.mean()) if ret.size else None,
        "return_std": float(ret.std(ddof=1)) if ret.size > 1 else 0.0,
        "length_mean": float(lng.mean()) if lng.size else None,
        "length_std": float(lng.std(ddof=1)) if lng.size > 1 else 0.0,
        "successes": int(successes),
        "collisions": int(collisions),
        "timestamp": int(time.time())
    }
    out = Path(experiment_path) / "eval_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info(f"Eval summary saved to: {out}")
    logger.info(f"Return mean±std: {summary['return_mean']:.2f} ± {summary['return_std']:.2f} "
                f"(n={summary['episodes']}) | successes={successes} collisions={collisions}")
```