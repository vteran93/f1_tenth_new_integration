# Diseño: entrenamiento por currículum + observación "minimapa" (occupancy grid)

Documento de diseño para la tesis. Fecha: 2026-08-06.
Motivación empírica y decisiones de diseño, con referencias `archivo:línea` al código real
para que sea implementable y defendible.

---

## 1. Motivación (por qué currículum, y por qué ahora)

Los tres entrenamientos en `oval_small` (2 agentes, política compartida) muestran una
progresión clara de fallos — ver `examples/multiagent/eval_videos/README.md`:

| Recompensa | Comportamiento aprendido | Vueltas |
|---|---|---|
| `ProgressRewardEnv` | se arrastra (~1.3 m/s), no arriesga | 0 |
| `SpeedRewardEnv` | se arrastra aún más (sobrevivir paga) | 0 |
| `ProgressTimePenaltyEnv` | por fin corre, pero **se estampa en la curva** | 0 |

La competencia que falta es concreta: **trazar a velocidad**. Entrenar directamente sobre la
pista completa a máxima dificultad exige aprender acelerar + frenar + girar + no chocar a la vez,
con recompensa densa pero `gamma` miope (0.9787 → horizonte efectivo ~47 pasos; una vuelta son
~700–5400 pasos, así que el agente **no "ve" la curva que viene**). El currículum descompone eso
en competencias que se adquieren en orden.

---

## 2. Currículum: diseño

### 2.1 Ejes de dificultad (todos ya tienen soporte en el código)

| Eje | Fácil → Difícil | Knob en el código |
|---|---|---|
| Geometría de pista | recta → curva suave → curva cerrada → circuito real | mapa (`update_map`) o generador |
| Velocidad máxima | v_max baja → alta | cap sobre la acción de velocidad (env/wrapper) |
| Anchura de pista | ancha → estrecha | `WIDTH` del generador (`random_trackgen.py:71`) |
| Posición de salida | grid fija → aleatoria + offset lateral | `reset_config` (`reset/__init__.py:8`) |
| Oponentes (multiagente) | solo → con rival | `num_agents` en `env_config` |
| Resolución del sensor | menos → más rayos | `num_beams` (`testing_rewards.yaml:18`) |

### 2.2 Etapas propuestas (hechas a mano, depurables)

Empezar con etapas fijas y explícitas (no auto-currículum): menos superficie para hacks.

- **E0 — Recta**: `training_tracks/create_tracks.py:91` (`create_straight_track`) o pista recta
  generada. v_max capada baja. Objetivo: acelerar y mantener carril.
- **E1 — Óvalo ancho, lento**: `oval_small`, salida `cl_grid_static`, v_max media. Objetivo:
  primera curva suave.
- **E2 — Óvalo, salida aleatoria, v_max alta**: `cl_random_static` / `move_laterally`. Objetivo:
  generalizar la curva y correr.
- **E3 — Circuito con curvas variadas**: `figure8_track` / `complex_circuit`
  (`create_tracks.py:234,409`) o pista generada con `TRACK_TURN_RATE` alto. Objetivo: encadenar
  curvas de distinto signo.
- **E4 — Circuito real**: `Spielberg` / `Catalunya` / `Monza` / `Silverstone` (`maps/`). Objetivo:
  transferencia final.

### 2.3 Mecanismo de promoción — **por métrica de evaluación, no por recompensa**

La skill es tajante (`references/evaluation.md`): la recompensa de entrenamiento engaña (ya lo
vimos tres veces). Promoción por **tasa de vueltas limpias / % de completado** sobre un set fijo
de evaluación:

> Regla: si `success_rate(etapa N) ≥ 0.8` durante K evaluaciones consecutivas → pasar a N+1.

### 2.4 Cableado en RLlib (Ray 2.46) — **no existe nada, hay que conectarlo**

No hay scaffolding de currículum (`grep curriculum` solo devuelve TODOs en
`examples/analyze_sac_collision_behavior.py:258,318`). Dos opciones:

**Opción A (recomendada, mínima):** callback `on_train_result` que ya se sobreescribe en el repo
para NaN (`run_nan_protected.py:60`, `lib/nan_detection.py:157`). Ahí se lee la métrica de eval y,
al cruzar el umbral, se re-configura la dificultad en todos los workers:

```python
# dentro de on_train_result(algorithm, result):
if promote(result):
    stage = self.stage + 1
    algorithm.env_runner_group.foreach_env(lambda env: env.set_stage(stage))
```

**Opción B (más idiomática):** hacer que `MultiAgentF110` (`multiagent_env.py:10`) herede de
`TaskSettableEnv` e implemente `get_task`/`set_task(stage)`, donde `set_task` llama a
`F110Env.update_map(map_de_la_etapa)` (`f110_env.py:618`) y ajusta v_max/reset_config.

**Cuidado técnico** (del mapa del código):
- Tras `update_map` hay que **reconstruir `self.reset_fn`** (`f110_env.py:169-171`), que queda
  ligada a la `track` anterior.
- `ScanSimulator2D` es **atributo de clase compartido** (`base_classes.py:58,128`): swaps de mapa
  por-worker OK, pero cuidado con múltiples envs vectorizados en el mismo proceso.
- Las pistas de `training_tracks/tracks/` **no están en el path del loader**; hay que copiarlas a
  `maps/` (el loader solo busca ahí, `track/utils.py:29`).

### 2.5 Defendibilidad (tesis)
- **Confound**: si se entrena con currículum, hacerlo **idéntico** para PPO y SAC y documentar el
  schedule. Mantener un **baseline sin currículum** (ablación).
- **Olvido catastrófico**: no eliminar del todo etapas fáciles; mezclar un % de pistas fáciles.
- **Eval fijo**: el set de evaluación final (pistas+seeds) no cambia entre currículum y baseline.

---

## 3. Observación "minimapa" (occupancy grid egocéntrico)

### 3.1 Decisión de fuente: **LiDAR (sensor-honesto)** — elegido por Victor

La rejilla se construye **rasterizando el scan del LiDAR**, no recortando la imagen del mapa. El
coche solo "ve" lo que el sensor real vería. Defendible: sin información privilegiada del mapa
global.

### 3.2 Especificación

- Rejilla **egocéntrica** N×N (p. ej. **64×64**), resolución `r` m/celda (p. ej. **0.15 m** →
  ~9.6 m de visión). Coche en el borde inferior-centro, **morro hacia arriba** (rotada a `theta`).
- Valores por celda (empatan con tu ASCII `_`/`X`): `0.0` libre, `1.0` ocupado (muro), y un tercer
  estado **desconocido** (`0.5`) para lo que ningún rayo tocó.
- **Rasterización** desde los rayos: para cada rayo `(θ_i, d_i)` del scan (`observation.py:36-39`),
  marcar libres las celdas a lo largo del rayo hasta `min(d_i, alcance_rejilla)` y ocupada la
  celda del impacto si `d_i < max_range`.

### 3.3 Limitación honesta de tu elección (y su mitigación)

- El config usa **`num_beams: 36`** y **FOV 4.7 rad (~270°)** (`base_classes.py:72-73`,
  `testing_rewards.yaml:18`). 36 rayos → rejilla **dispersa** (huecos entre rayos a distancia) y
  **ciega detrás del coche** (~90° traseros sin cobertura).
- Mitigación: subir `num_beams` **solo para la observación de rejilla** (p. ej. 108–180) — es un
  knob barato; y usar el estado "desconocido" explícito para no confundir "sin datos" con "libre".
  El nº de rayos puede incluso ser un eje del currículum (§2.1).

### 3.4 Dónde se implementa

Observación nueva registrada en la fábrica, **no** un post-proceso:

1. Subclase `OccupancyGridObservation(Observation)` en `f1tenth_gym/envs/observation.py`
   (junto a `OriginalObservation:30`), con `observation_space()` = `Box(0,1,(N,N,1))` (o `Dict`,
   ver §3.5) y `observe()` que rasteriza el scan.
2. Registrar rama nueva en `observation_factory` (`observation.py:343`), p. ej.
   `type: "occupancy_grid"`.
3. Propagar por el wrapper multiagente: `MultiAgentF110._make_single_agent_obs_space` y
   `_convert_obs` (`multiagent_env.py:13-33`) deben pasar la rejilla al espacio del agente.

### 3.5 El coche necesita saber su velocidad → observación híbrida

Una imagen sola no lleva la velocidad, que es esencial para frenar antes de la curva. Recomendado:

```
Dict{
  "grid":  Box(0,1,(N,N,1)),        # minimapa
  "state": Box(-inf,inf,(k,))       # [vel_x, vel_y, vel_ang, ...]
}
```

→ requiere **modelo custom** (CNN para `grid` + MLP para `state`, concatenados), registrado con
`ModelCatalog.register_custom_model` y referenciado como `model.custom_model` en `ppo_params`
(mismo mecanismo que ya usamos para `custom_action_dist` de `nan_protection`).

### 3.6 Coste (recordatorio de la skill)

CNN sobre 64×64 en **CPU** es más lento que el MLP `[256,256]` actual. La skill avisa del límite
de cómputo en hardware de consumo. Mantener la CNN pequeña (2–3 conv) y N moderado (48–64).

---

## 4. Secuenciación recomendada

1. **Currículum primero.** Es barato, reusa el env y el generador, y ataca el fallo que ya vemos
   (trazar a velocidad). No cambia la representación → comparable con lo actual.
2. **Minimapa después**, como mejora de representación (nuevo tipo de obs + CNN). Es un lift mayor
   (código + cómputo).
3. **No mezclar ambos en el mismo experimento** o no se sabrá cuál ayudó (confound).

---

## 5. Lista concreta de ficheros a tocar

**Currículum (Opción A):**
- `examples/multiagent/lib/callbacks.py` — callback `on_train_result` que promueve de etapa.
- `examples/multiagent/lib/multiagent_env.py` — `set_stage(k)` que llama `update_map` + reconstruye
  `reset_fn` + ajusta v_max/reset_config.
- `examples/multiagent/configs/experiments_*.yaml` — definición de etapas y umbrales.
- (opcional) copiar `training_tracks/tracks/*` a `maps/`; hoistear knobs de `random_trackgen.py:65`
  a parámetros de función.

**Minimapa:**
- `f1tenth_gym/envs/observation.py` — `OccupancyGridObservation` + rama en `observation_factory:343`.
- `examples/multiagent/lib/multiagent_env.py` — propagar la rejilla en `_make_single_agent_obs_space`
  y `_convert_obs`.
- `examples/multiagent/lib/models.py` (nuevo) — modelo CNN+MLP custom.
- `examples/multiagent/run.py` — registrar el modelo custom (junto al registro de `nan_protection`).
- config: `observation_config: {type: occupancy_grid, grid_size: 64, resolution: 0.15}` +
  `num_beams` subido.

---

## 6. Cómo medirlo (defendible)

Por cada configuración (baseline vs currículum; MLP vs minimapa): ≥5 seeds, set de evaluación fijo,
métricas **lap time / % completado / tasa de colisión / éxito**, agregadas con media±std (o
mediana+IQR si no-normal), test de Mann-Whitney U o Welch según normalidad
(`references/evaluation.md`).
