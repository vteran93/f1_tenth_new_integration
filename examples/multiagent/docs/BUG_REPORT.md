# Reporte de bugs — `examples/multiagent`

**Fecha:** 2026-07-18
**Rama:** `multiagent`
**Alcance:** escaneo estático del código fuente del ejemplo multiagente
(`lib/rewards.py`, `lib/multiagent_env.py`, `lib/callbacks.py`, `lib/utils.py`, `run.py`).

> Nota: `gh` (GitHub CLI) no está instalado en este entorno, por lo que los
> hallazgos se documentan aquí en lugar de abrirse como issues. Cada hallazgo
> está redactado para poder copiarse tal cual a un issue de GitHub.

## Resumen

| # | Severidad | Archivo | Título |
|---|-----------|---------|--------|
| 1 | 🔴 Alta | `lib/rewards.py:39` | `ProgressRewardEnv`: corrección de vuelta invertida → recompensa negativa gigante al completar una vuelta |
| 2 | 🔴 Alta | `lib/multiagent_env.py:31` | `_last_s` se reinicia a 0 en vez del arclength inicial → pico de recompensa espurio en el primer paso |
| 3 | 🟠 Media | `lib/rewards.py:554` | `KohondaMultiAgentF110Env`: distancia sin signo + `prev_waypoints` en (0,0) sin reset → reward hacking y pico inicial |
| 4 | 🟠 Media | `lib/rewards.py:184` | `SpeedRewardEnv`: `position_speed` sin dirección → oscilar farmea recompensa (reward hacking) |
| 5 | 🟠 Media | `run.py:344` | Bucle `--all`: `raise e` aborta el resto de experimentos; `continue` es código muerto |
| 6 | 🟠 Media | `run.py:172` | Tuning de hiperparámetros optimiza una métrica inexistente (`episode_reward_mean` vs `episode_return_mean`) |
| 7 | 🟡 Baja | `run.py:84` | Semilla de evaluación no reproducible (`random.randint` sin sembrar) |

---

## 1. 🔴 `ProgressRewardEnv`: la corrección de cierre de vuelta está invertida

**Archivo:** `examples/multiagent/lib/rewards.py:36-44`

```python
prog = current_s - self._last_s[i]

# Handle lap completion (when current_s wraps around to beginning)
if prog > 0.9 * self.env.track.centerline.spline.s[-1]:
    prog = (self.env.track.centerline.spline.s[-1] - self._last_s[i]) + current_s

reward = prog
```

**Problema:** cuando el coche cruza la línea de meta avanzando, `current_s` "envuelve"
de ~`L` (longitud de la pista) a ~`0`, por lo que:

```
prog = current_s - last_s ≈ 0 - L = -L   (negativo grande)
```

La condición `prog > 0.9*L` es **falsa** en ese caso, así que la corrección nunca se
aplica y el agente recibe una recompensa ≈ `-L` **justo al completar una vuelta**.
La rama de corrección solo se dispara con `prog` **positivo** grande (es decir, cuando
el coche retrocede cruzando la meta: `last_s≈0`, `current_s≈L`), y encima infla aún más
esa recompensa. La lógica de wrap está invertida respecto de la dirección real.

Las demás clases de recompensa (`ProgressRewardAdvancedEnv`, `SafetyRewardEnv`, etc.)
usan la lógica correcta:

```python
if prog < -0.5 * track_length:
    prog += track_length   # cruce de meta hacia adelante
elif prog > 0.5 * track_length:
    prog -= track_length   # movimiento hacia atrás
```

**Impacto:** `ProgressRewardEnv` es la función de recompensa **por defecto** en casi todos
los configs (`experiments.yaml`, `experiments_pepe.yaml`, `experiments_sergio.yaml`,
`hyperparameter_tuning.yaml`, `testing_rewards.yaml`, …). El agente es penalizado
fuertemente exactamente cuando hace lo correcto (completar la vuelta), lo que sesga toda
la señal de aprendizaje y cualquier comparación SAC vs PPO basada en esta recompensa.

**Fix sugerido:** sustituir la corrección por la misma que usan las demás clases:

```python
prog = current_s - self._last_s[i]
track_length = self.env.track.centerline.spline.s[-1]
if prog < -0.5 * track_length:
    prog += track_length
elif prog > 0.5 * track_length:
    prog -= track_length
reward = prog
```

---

## 2. 🔴 `_last_s` se reinicia a 0 en lugar del arclength inicial → pico espurio en el primer paso

**Archivo:** `examples/multiagent/lib/multiagent_env.py:26-33`

```python
def reset(self, *, seed=None, options=None):
    obs, info = self.env.reset(seed=seed, options=options)
    self._last_positions = [(self.env.poses_x[i], self.env.poses_y[i]) for i in range(self.env.num_agents)]
    self._crashed_agents = set()
    self._last_s = [0.0] * self.env.num_agents  # <-- Reset progress tracker
```

**Problema:** `_last_s` se fija a `0.0`, pero la posición de arranque del coche casi nunca
está en el arclength 0 del spline. En el primer `step`:

```
prog = current_s - _last_s = current_s - 0 = current_s   (p. ej. 5 m)
```

Esto genera una recompensa de progreso positiva grande y espuria en el **primer paso de
cada episodio**, proporcional al arclength de la posición de salida. Afecta a **todas** las
clases de recompensa basadas en progreso (`ProgressRewardEnv`, `ProgressRewardAdvancedEnv`,
`SpeedRewardEnv`, `WaypointRewardEnv`, `SafetyRewardEnv`).

**Impacto:** contamina el retorno del episodio con un offset dependiente de la posición de
salida; es también un vector de reward hacking (reiniciar/chocar rápido para recolectar el
bono del primer paso).

**Fix sugerido:** inicializar `_last_s` con el arclength real de arranque en `reset()`:

```python
self._last_s = []
for i in range(self.env.num_agents):
    s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
        self.env.poses_x[i].item(), self.env.poses_y[i].item())
    self._last_s.append(s)
```

(Ojo: `SpeedRewardEnv` usa `raceline` en vez de `centerline`; ver que la inicialización
use el mismo spline que el cómputo de la recompensa, o inicializar por-clase.)

---

## 3. 🟠 `KohondaMultiAgentF110Env`: distancia sin signo + `prev_waypoints` sin reset

**Archivo:** `examples/multiagent/lib/rewards.py:553-578` y `521-526`

```python
dist = np.linalg.norm(self._current_waypoints[i] - self.prev_waypoints[i])
...
reward = dist + collision_penalty
```

**Problema A (reward hacking):** `dist` es la distancia euclídea entre los waypoints más
cercanos de dos pasos consecutivos, **siempre no negativa** independientemente de la
dirección. Un agente que retrocede u oscila igualmente cobra recompensa positiva
proporcional al desplazamiento — exactamente el patrón de "ir y venir cerca de la meta"
que la metodología del proyecto advierte evitar. La implementación original de Kohonda usa
progreso **con signo** a lo largo del índice de waypoint.

**Problema B (pico inicial):** `self.prev_waypoints` se inicializa a `(0,0)` en `__init__`
(`rewards.py:523`) y **no se reinicia** en `reset()` (el `reset()` base solo toca `_last_s`,
`_crashed_agents` y `_last_positions`). En el primer paso del primer episodio,
`prev_waypoints=(0,0)` mientras el waypoint real está lejos → `dist` enorme → recompensa
espuria gigante. En episodios posteriores el estado de Kohonda (`prev_waypoints`,
`prev_vels`, `prev_yaw`, `_current_indices`) arrastra valores del episodio anterior.

**Fix sugerido:** usar progreso con signo por índice de waypoint (diferencia de índices con
manejo de wrap) en vez de la norma; y sobreescribir `reset()` en la clase para reinicializar
`prev_waypoints`/`prev_vels`/`prev_yaw` a la pose inicial de cada agente.

---

## 4. 🟠 `SpeedRewardEnv`: `position_speed` sin dirección permite farmear recompensa

**Archivo:** `examples/multiagent/lib/rewards.py:172-198`

```python
track_speed = max(0.0, progress / self.timestep)          # correctamente recorta hacia atrás
...
position_speed = distance_moved / self.timestep           # euclídea → SIEMPRE ≥ 0
speed = max(track_speed, position_speed)                   # <-- gana la sin dirección
```

**Problema:** aunque `track_speed` recorta el movimiento hacia atrás a 0, `position_speed`
es la distancia euclídea (sin dirección) y `speed = max(track_speed, position_speed)`. Por
tanto, un coche que se sacude adelante-atrás o gira sobre sí mismo obtiene `speed > 0` y
cobra `speed_reward` (y hasta `speed_bonus`). Es un canal de reward hacking que contradice el
objetivo de la recompensa (avanzar rápido por la pista).

**Fix sugerido:** usar únicamente el progreso con signo a lo largo de la pista para la
velocidad (`track_speed`), o proyectar el desplazamiento sobre la tangente de la línea de
carrera antes de usarlo; no mezclar con una magnitud sin dirección.

---

## 5. 🟠 Bucle `--all`: `raise e` aborta los experimentos restantes (`continue` es código muerto)

**Archivo:** `examples/multiagent/run.py:341-347`

```python
try:
    run_training(cfg)
except Exception as e:
    logger.error(f"Error during training of {cfg['name']}: {e}")
    raise e
    continue
# This way we can catch errors in training and continue with the next experiment
```

**Problema:** el comentario declara la intención de "continuar con el siguiente experimento",
pero `raise e` vuelve a lanzar la excepción y aborta todo el bucle `--all`. El `continue`
posterior es **inalcanzable** (código muerto). El comportamiento contradice la intención.

**Fix sugerido:** decidir el contrato. Si se quiere continuar ante fallos, quitar el
`raise e` (y el `continue` sobra porque ya está al final del bucle); si se quiere abortar,
quitar el comentario y el `continue` engañosos.

---

## 6. 🟠 Tuning de hiperparámetros optimiza una métrica inexistente

**Archivo:** `examples/multiagent/run.py:172` (comparar con `145`, `173`, `191`, y
`lib/utils.py:83-85`)

```python
default_metric = "env_runners/episode_reward_mean"      # <-- OptunaSearch
search_alg = OptunaSearch(metric=default_metric, mode="max", seed=SEED)
```

**Problema:** OptunaSearch usa `env_runners/episode_reward_mean`, pero el resto del pipeline
usa `env_runners/episode_return_mean` (el `TrialPlateauStopper`, `checkpoint_score_attribute`
y `get_best_checkpoint`). En el API nuevo de RLlib la métrica es `episode_return_mean`;
`episode_reward_mean` es el nombre del API viejo. Si la clave no existe, Optuna no recibe la
señal de objetivo y el tuning no optimiza nada útil (o falla).

**Fix sugerido:** unificar a `env_runners/episode_return_mean` en todo el archivo.

---

## 7. 🟡 Semilla de evaluación no reproducible

**Archivo:** `examples/multiagent/run.py:82-85`

```python
.evaluation(
    evaluation_interval=config["training"]["eval_interval"],
    evaluation_num_env_runners=1,
    evaluation_config={"seed": random.randint(0, 10000)},
)
```

**Problema:** la semilla de evaluación se genera con `random.randint` sin sembrar el RNG del
módulo `random`, por lo que cambia en cada ejecución. Esto rompe la reproducibilidad de las
métricas de evaluación (tiempo de vuelta, tasa de colisión), que es precisamente el criterio
de comparación defendible para SAC vs PPO. Relacionado: `.debugging(seed=SEED)` fija `SEED=42`
para **todos** los trials, incluido el tuning, lo que impide una comparación multi-semilla.

**Fix sugerido:** derivar la semilla de evaluación de forma determinista a partir de `SEED`
(o de la semilla del trial) y, para comparaciones estadísticas, parametrizar la semilla por
trial para poder correr ≥5 semillas por algoritmo.

---

## Observaciones secundarias (menor confianza / a revisar)

- **`WaypointRewardEnv`** (`rewards.py:256-277`): la mezcla del bono de vuelta (+10 con
  `last_s_threshold=0`) y el conteo de umbrales puede doble-contar o descontar cerca de la
  meta; conviene añadir un test que verifique el reward acumulado en una vuelta limpia.
- **`AverageSpeed`** (`callbacks.py:410`): asume el layout `agents[i].state[3]` para la
  velocidad; frágil si cambia el modelo dinámico del simulador.
- **`multiagent_env.step`** (`multiagent_env.py:42`): a los agentes chocados se les envía
  acción cero; si el espacio de acción es `[steering, acceleration]`, "cero" no es una acción
  neutra (el coche sigue rodando). Verificar contra la versión instalada de `f1tenth_gym`.
- **`utils.py:45-57`** (`_parse_tune_function`): usa `eval()` sobre contenido del YAML. No es
  un bug funcional, pero es una superficie de ejecución de código arbitrario si el YAML no es
  de confianza.

## Cómo reproducir / verificar

Los hallazgos 1–4 se pueden confirmar con un test unitario que instancie cada `*RewardEnv`,
haga `reset()` y compare la recompensa del primer paso y la del cruce de meta contra el
progreso esperado con signo. Los tests actuales (`tests/test_rewards.py`) no cubren el cruce
de meta ni el primer paso.
