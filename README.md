![Python 3.8 3.9](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym/actions/workflows/docker.yml/badge.svg)
![Code Style](https://github.com/f1tenth/f1tenth_gym/actions/workflows/lint.yml/badge.svg)

# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

This project is still under heavy developement.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

---

# Fork de investigación — RL multi-agente (SAC / PPO sobre F1TENTH)

> Este repositorio es un **fork de investigación** de `f1tenth_gym`. Sobre el
> simulador base construimos un entorno multi-agente (RLlib) para comparar
> algoritmos de RL (SAC vs PPO) y funciones de recompensa en carreras autónomas.
> El trabajo vive principalmente en `examples/multiagent/`.

## Changelog / Historia de los experimentos

Esta sección **reconstruye la intención** detrás de los commits y ramas de
2025 (equipo: **Victor**, **Sergio** y **Pepe**). No busca culpables ni
responsables: es un mapa para entender *qué exploraba y qué quería lograr cada
uno* cuando todos experimentábamos en paralelo, antes de poner orden.

> Convención: cada entrada enlaza la **intención** con la evidencia (ramas/commits).
> La rama consolidada actual es `publicacion-paper` (ver
> [`docs/CONSOLIDATION_publicacion-paper.md`](docs/CONSOLIDATION_publicacion-paper.md)).

### Fase 0 — Base heredada (upstream, 2023)
Fork de `f1tenth_gym` (Zheng, Amine, Berducci, Tumu et al.): simulador
single-agent con Gymnasium, LiDAR, modelos dinámicos y tracks. Punto de partida:
`cd56335` (jul-2023).

### Fase 1 — "Hacer que el multi-agente arranque" (jun 2025)
*Intención: pasar de single-agent a varios coches y poder entrenar una política.*
- **Victor** — puesta a punto multi-agente y ejecución de su propia política, con
  foco en entorno Windows/WSL y **entrenamiento en la nube** (terraform + `train_cloud/`).
  Ramas: `tfm_implement_multi_agent`, `mmarllib-sergio` (setup + requirements Windows + devcontainer).
  Commits: *"Code to setup multiagent"*, *"Cambios para ejecutar mi politica"*, *"test in cloud"*.
- **Sergio** — apuesta por **Ray/RLlib** para PPO + TensorBoard como camino "serio"
  de entrenamiento. Ramas: `ray-new-api`, `ray-sergio`.
  Commits: *"Add multi-agent training and evaluation example using PPO with Ray"*, *"Add tensorboard script"*.
- **Victor** — *"Implement sac with ray like Sergio"*: alinear SAC al enfoque Ray de Sergio.

**Aprendizaje clave de esta fase (documentado en el código):**
`f1tenth_gym/kohonda/kohonda_multi_agent_F110env.py` (`VictorMultiAgentEnv.step`)
deja constancia de que **stable-baselines3 no soporta multi-agente** (su
`collect_rollouts`/`monitor.py` espera un único escalar de recompensa). Esa fue la
razón real de migrar de SB3 → **RLlib**. Ese módulo Kohonda se conserva a propósito
como memoria de esa decisión (ver [`examples/multiagent/docs/INTEGRATION_evans_kohonda.md`](examples/multiagent/docs/INTEGRATION_evans_kohonda.md)).

### Fase 2 — Exploración de recompensas y de líneas paralelas (mediados de jun 2025)
*Intención: probar distintas señales de recompensa y controladores; cada quien por su vía.*
- **Sergio** — recompensa por progreso + penalización de colisión + logging TensorBoard;
  primera reorganización del repo.
- **Pepe** — integración de los **benchmarks de Benjamin Evans** (`BDEvan5/f1tenth_benchmarks`):
  recompensas `RacePerformanceReward` y `CrossTrackHeadReward` (CTH) + `PurePursuitPlanner`
  (baseline clásico). Contexto: TFM. **Nota de autoría:** este trabajo es de Pepe, pero como
  Pepe no usaba git, fue **Victor quien lo commiteó** en su nombre; por eso figura bajo la
  cuenta de Victor como *"Archivos pepe"* (ramas huérfanas `evans_integration_pepe`,
  `evans_integration_show_simulation_pepe`). La autoría git ≠ autoría intelectual aquí.
- **Victor** — búsqueda de hiperparámetros SAC y **tracks propios** (oval_small, figure8,
  complex_circuit, semi_rectangular). Ramas: `sac_experimental`, `tensorboard_validation`,
  `training_tracks`. También `benchmark_integrations` (f1tenth_benchmarks standalone).

### Fase 3 — Endurecer el framework y consolidar (fin jun – inicio jul 2025)
*Intención: dejar de pelearse con el framework y tener un pipeline reproducible.*
- **Sergio** — gran refactor: CLI con subcomandos (`train`/`eval`), consolidación de
  configs PPO/SAC en YAML, `SaveConfig`, encadenado de experimentos, `TrialPlateauStopper`
  y métricas custom. Commits: *"First draft for full organization…"*, *"Run.py now allows chaining experiments"*.
- **Ambos** — robustez frente a **NaN** (ruido de LiDAR → errores de torch), ajuste de
  memoria/paralelismo (*"It happened already to both of us"* al quedarse sin RAM) y
  **`TimestepsStopper`** que arregla el bug de entrenamiento infinito.
- **Victor** — callbacks de métricas (LapProgress, LapTime, CollisionStats, AverageSpeed),
  *"Refactor consolidado"*.
- Nuevas familias de recompensa: Speed / Waypoint / CompetitiveOvertaking (Sergio) y
  **Kohonda** (Victor).

### Fase 4 — Ciencia de la función de recompensa (jul 2025)
*Intención: comparar recompensas de forma ordenada y afinar hiperparámetros.*
- **Kohonda** integrada y simplificada (PRs #10, #11); comparación conceptual en `lib/REWARDS.md`.
  Ramas: `kohonda_rewards`, `kohonda_simplify`.
- **Sergio** — **tuning de hiperparámetros** con Optuna en `run.py`; *"Clean up of the whole repo"*
  (retira scripts top-level legacy).
- **Victor** — *"kohonda simplify"*, retirada de tests obsoletos, *"ray worker watchdog"*.

### Fase 5 — Evaluación y ajustes finales (20 jul 2025)
- **Sergio** — `fcnet_activation=relu` para PPO (evita gradiente que se desvanece),
  configuración dinámica de env-runners, `.yml`→`.yaml`.
- **Victor** — *"Add plots to evaluate behaviour"* (gráficas de evaluación).

### Fase 6 — Conciliación para el paper (2026)
*Intención: recuperar todo lo aprovechable disperso en ramas y trabajar sobre una sola base.*
- Rama **`publicacion-paper`** (derivada de `main`): consolida la mainline `multiagent`
  + rescate del working tree + ramas con trabajo único, excluyendo basura, outputs y secretos.
- Documentación asociada: reporte de bugs de las recompensas
  ([`examples/multiagent/docs/BUG_REPORT.md`](examples/multiagent/docs/BUG_REPORT.md)),
  detalle de la consolidación y ruta de integración de los algoritmos de Evans/Pepe.
- Reconciliado también el repo standalone de Sergio (`SergioSV96/multiagent`,
  snapshot "Todo" del 2025-07-20): 22/23 archivos ya idénticos; solo difería
  `hyperparameter_tuning.yaml` (se mantuvo la nuestra, con nombres de parámetro
  SAC válidos de RLlib). Detalle en `docs/CONSOLIDATION_publicacion-paper.md`.

## Mapa de ramas (qué era cada una)

| Rama | Intención reconstruida | Estado en `publicacion-paper` |
|---|---|---|
| `multiagent` | Mainline consolidada (RLlib, recompensas, callbacks, configs) | Base (fusionada) |
| `ray-new-api`, `ray-sergio` | Migración de Sergio a RLlib (PPO) | Contenida / edición divergente descartada |
| `tfm_implement_multi_agent`, `mmarllib-sergio` | Setup multi-agente, Windows, devcontainer, nube (terraform) | Fusionadas |
| `evans_integration_pepe`, `evans_integration_show_simulation_pepe` | TFM de Pepe: recompensas de Evans (CTH, RacePerformance) + Pure Pursuit (autoría de Pepe; commiteado por Victor porque Pepe no usaba git) | Fuera (huérfanas); **portar archivos** (ver INTEGRATION doc) |
| `pepe-TFM`, `tfm-pepe` | Trabajo de TFM asociado | Contenida / referencia |
| `kohonda_rewards`, `kohonda_simplify` | Port de recompensa Kohonda (origen del aprendizaje SB3→RLlib) | Fusionadas |
| `sac_experimental`, `tensorboard_validation`, `hyperparameter-tuning` | Búsqueda de hiperparámetros SAC + validación TensorBoard | Fusionadas / contenida |
| `training_tracks` | Generación de tracks propios (oval, figure8, etc.) | Fusionada (sin outputs) |
| `benchmark_integrations` | Integración de `f1tenth_benchmarks` | Fusionada |
| `multi_agent_debug_ray`, `bugfix_error` | Depuración/estabilidad de Ray | Contenidas |
| `deprecated`, `revert_multi_agent` | Código obsoleto / vías muertas | Fuera |

---

## Quickstart
We recommend installing the simulation inside a virtualenv. You can install the environment by running:

```bash
virtualenv gym_env
source gym_env/bin/activate
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym
pip install -e .
```

Then you can run a quick waypoint follow example by:
```bash
cd examples
python3 waypoint_follow.py
```

A Dockerfile is also provided with support for the GUI with nvidia-docker (nvidia GPU required):
```bash
docker build -t f1tenth_gym_container -f Dockerfile .
docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym_container
````
Then the same example can be ran.

## Known issues
- Library support issues on Windows. You must use Python 3.8 as of 10-2021
- On MacOS Big Sur and above, when rendering is turned on, you might encounter the error:
```
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```
You can fix the error by installing a newer version of pyglet:
```bash
$ pip3 install pyglet==1.5.20
```
And you might see an error similar to
```
f110-gym 0.2.1 requires pyglet<1.5, but you have pyglet 1.5.20 which is incompatible.
```
which could be ignored. The environment should still work without error.

## Citing
If you find this Gym environment useful, please consider citing:

```
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```
