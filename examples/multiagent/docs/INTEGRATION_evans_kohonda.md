# Integración a mediano plazo: algoritmos `evans_*_pepe` + estado de Kohonda

**Fecha:** 2026-07-18
**Rama:** `publicacion-paper`
**Contexto:** análisis de las ramas huérfanas `evans_integration_pepe` /
`evans_integration_show_simulation_pepe` (integración de Pepe de los benchmarks
de Benjamin Evans, `BDEvan5/f1tenth_benchmarks`) y localización del trabajo Kohonda.

> Estas ramas quedaron **fuera** de la consolidación (historia huérfana, sin
> merge-base, y volcados con basura `*:Zone.Identifier`). Este documento extrae
> lo valioso y define cómo portarlo, sin arrastrar su historia.

---

## 1. ¿Dónde está el trabajo Kohonda? (sí, está preservado)

En `publicacion-paper` hay **dos** implementaciones Kohonda, más sus configs:

| Ubicación | Qué es | Stack |
|---|---|---|
| `examples/multiagent/lib/rewards.py::KohondaMultiAgentF110Env` (L499) | Versión **consolidada**, solo-reward, sobre nuestro `MultiAgentF110` | RLlib |
| `f1tenth_gym/kohonda/kohonda_multi_agent_F110env.py` (470 L) | Implementación **standalone** original: `KohondaMultiAgentF110Env(F110Env)` + `VictorMultiAgentEnv`, con `custom_vector_observation.py` y `utils.py` | SB3 (era previa) |
| `configs/Kohonda_experiments.yml`, `configs/experiments_kohonda_sac.yaml` | Experimentos SAC/Kohonda | — |

### Por qué Kohonda importa (aprendizaje del framework)
El módulo `f1tenth_gym/kohonda/` conserva el hallazgo clave del proyecto, documentado
en `VictorMultiAgentEnv.step()` (L394-399):

> `_get_reward` devuelve la puntuación total de todos los agentes, no la de cada
> uno; si devuelves la de cada uno, **SAC de SB3 falla en `monitor.py`** en
> `self.rewards.append(float(reward))` de `collect_rollouts` — **stable-baselines3
> no sirve para multi-agente**. Hay que devolver la recompensa de un solo agente y
> el otro sería un dummy.

**Esto es exactamente la justificación de la migración SB3 → RLlib** (nuestro stack
actual `MultiAgentF110`). Vale la pena citarlo en la sección de método del paper:
SB3 asume un único escalar de recompensa por step, por lo que el verdadero
multi-agente requería RLlib. **Recomendación: NO borrar `f1tenth_gym/kohonda/`**;
es documentación viva de una decisión de arquitectura.

---

## 2. Algoritmos en `evans_*_pepe`

### 2.1 Recompensas (`examples/rewards.py`)
Dos estrategias, ambas como **objeto callable** `__call__(state, agent_id) -> (total, components)`:

- **`RacePerformanceReward`**: `progress + speed + collision + overtake + jerk`, con
  `stall_penalty` bajo `min_speed_threshold`. Multi-componente y competitivo.
- **`CrossTrackHeadReward` (CTH)**: `-cte_scale·CTE − he_scale·HE − collision`, el
  clásico de Evans (error de traza + error de rumbo contra la raceline). Baseline
  principista muy citado en la literatura F1TENTH.

### 2.2 Planner clásico (`examples/planners.py`, `purepursuit_multiagent.py`)
- **`PurePursuitPlanner`**: control geométrico puro (lookahead + `atan2`). Sirve como
  **baseline no-RL** — crítico para una tesis (RL vs control clásico).

### 2.3 Factoría de algoritmos (`examples/algorithms.py`)
- `get_ppo_config` / `get_sac_config`: factorías single-agent parametrizadas por YAML.
  Redundante con nuestro `ALGO_MAP` en `run.py`; **no aporta**.

### 2.4 Patrón de arquitectura (lo más valioso)
El env de entrenamiento (`multiagent_simplified_tensorboard.py`) inyecta la recompensa
**por-agente desde YAML** y expone componentes:
```python
self.reward_fn[agent_id] = RewardClass(**reward_params)   # estrategia por agente
state = {poses_x, poses_y, poses_theta, linear_vels_x/y, ang_vels_z,
         collisions, lap_counts, lap_times, actions, prev_actions}
reward, components = self.reward_fn[agent_id](state, agent_id)
self.last_components[agent_id] = components                # → logging por componente
```
Contraste con lo nuestro: nosotros **heredamos** la reward (una subclase de
`MultiAgentF110` por función, elegida por nombre de clase). El patrón de Pepe la
**compone/inyecta** y devuelve el desglose de componentes.

---

## 3. Qué vale la pena integrar (y qué no)

| Elemento | Valor | Decisión |
|---|---|---|
| Retorno `(total, components)` + logging por componente | **Alto** — diagnostica reward hacking (ver `BUG_REPORT.md`) | ✅ Adoptar |
| Reward CTH (CTE + HE) | **Alto** — baseline principista para el paper | ✅ Portar como `CrossTrackHeadRewardEnv` |
| `PurePursuitPlanner` | **Alto** — baseline clásico vs RL | ✅ Portar a `examples/multiagent/baselines/` |
| Jerk penalty + stall penalty | **Medio** — suavidad/realismo | ✅ Añadir como componentes opcionales |
| Overtake bonus | **Medio** | ⚠️ Reimplementar con arc-length (ver 4) |
| Reward inyectable desde YAML (vs herencia) | **Medio** — refactor de diseño | 🕓 Mediano plazo (evaluar coste/beneficio) |
| `algorithms.py`, `back/*`, scripts sueltos | Bajo/nulo — redundante o experimental | ❌ Descartar |

---

## 4. Bugs/limitaciones a corregir al portar (no copiar tal cual)

Las recompensas de Pepe tienen atajos que **no** debemos heredar:

1. **Progreso mal proxy** — `RacePerformanceReward` usa
   `progress = lap_counts + poses_x/100`. `poses_x` es coordenada mundo, no
   arc-length → progreso incorrecto y hackeable. **Usar** nuestro
   `track.centerline.spline.calc_arclength_inaccurate(...)` con manejo de wrap con
   signo (mismo patrón correcto de `ProgressRewardAdvancedEnv`; ojo con el bug
   inverso de `ProgressRewardEnv` descrito en `BUG_REPORT.md` §1).
2. **Velocidad = acción comandada** — usa `state['actions'][0]` como velocidad. El
   `state` ya trae `linear_vels_x`; **usar la velocidad medida**.
3. **Overtake solo por `lap_counts`** — no detecta adelantamientos dentro de la misma
   vuelta. **Usar** comparación de arc-length `s` entre agentes (cf. la clase
   comentada `CompetitiveOvertakingRewardEnv` en `lib/rewards.py`).
4. **CTE por `argmin` fuerza bruta** cada step (O(N)). **Reusar**
   `nearest_point_on_trajectory` (njit, ya en `lib/utils.py`) o
   `cubic_spline.find_closest_point` (añadido en el merge de `tensorboard_validation`).
5. **`print(...)` por step** en el loop de reward → **quitar**; usar los callbacks de
   métricas (`lib/callbacks.py`) para TensorBoard.

---

## 5. Ruta de integración propuesta (mediano plazo)

**Fase A — Baseline clásico (rápido, alto valor para tesis)**
1. Portar `PurePursuitPlanner` a `examples/multiagent/baselines/pure_pursuit.py`,
   adaptado a la API multi-agente (`action_dict`) y a nuestra raceline.
2. Script de evaluación que corra el planner en las mismas tracks/seeds que el RL →
   tabla comparativa (lap time, colisiones) RL vs clásico.

**Fase B — Reward CTH como clase de primera clase**
3. Crear `CrossTrackHeadRewardEnv(MultiAgentF110)` en `lib/rewards.py`, reusando
   spline/arclength y `nearest_point_on_trajectory`, corrigiendo los puntos de §4.
4. Añadirla a los configs de comparación (junto a Progress/Speed/Waypoint/Kohonda).

**Fase C — Logging por componente (transversal, ataca reward hacking)**
5. Estandarizar que cada `_compute_reward` pueda devolver `(reward, components)` y
   que un callback (`RewardComponents`) los escriba a TensorBoard por-agente. Esto
   hace visible el hacking de línea de meta descrito en `BUG_REPORT.md`.

**Fase D — (opcional) Reward inyectable desde YAML**
6. Evaluar migrar de "una subclase por reward" a "estrategia de reward inyectada"
   (patrón de Pepe). Beneficio: reward distinta por agente y composición; coste:
   refactor de `get_reward_class`/configs. Decidir según necesidad del paper.

---

## 6. Nota de proceso

Las ramas `evans_*_pepe` **no** se mergean (historia huérfana + basura). El plan es
**portar archivos concretos** (cherry-pick manual de `rewards.py`, `planners.py`) a
nuestra estructura `examples/multiagent/`, no fusionar ramas. Los originales siguen
intactos en `origin` como referencia.
