# Currículum "kata": 100 pistas de Taikyoku Shodan a Bassai Dai

Fecha: 2026-09-11. Implementa la parte de currículum de
`docs/CURRICULUM_AND_MINIMAP_design.md` (§2) con un generador de pistas propio y el cableado
mínimo en RLlib. Motivación: las 6 recompensas probadas en `oval_small` no completan ni una
vuelta bajo la misma configuración (ver `examples/multiagent/eval_videos/README.md`); el cuello
de botella es el setup de aprendizaje, no la recompensa.

## 1. Analogía con los kata Shotokan

Cada kata añade **una** técnica nueva sobre las anteriores. Cada etapa del currículum añade
**una** competencia de conducción nueva y la geometría de la pista es la mínima que la exige.

| Etapa | Kata | Competencia nueva | Familia geométrica | Anchura (m) | R mín (m) | Longitud (m) | v máx (m/s) |
|---|---|---|---|---|---|---|---|
| 1 | Taikyoku Shodan | acelerar, mantener carril, una curva suave, un solo sentido de giro | estadio (rectángulo con semicírculos) | 4.0–4.4 | 6–9 | 55–80 | 5 |
| 2 | Taikyoku Nidan | primer cambio de sentido | estadio + una S suave en la recta | 3.8–4.2 | 4.5–6 | 70–100 | 5 |
| 3 | Taikyoku Sandan | encadenar cambios de sentido a velocidad | estadio + chicane o doble S | 3.6–4.0 | 4–5.5 | 80–115 | 6 |
| 4 | Heian Shodan | curvas de ~90° con rectas desiguales (frenar antes del vértice) | polígono convexo redondeado (3–5 vértices) | 3.4–3.8 | 3–4.5 | 70–110 | 6 |
| 5 | Heian Nidan | girar hacia el otro lado (curva cóncava) | polígono con un vértice reflejo ("riñón") | 3.2–3.6 | 2.8–4 | 80–120 | 7 |
| 6 | Heian Sandan | secuencias en S, curvas alternas | polígono con 2–3 vértices reflejos | 3.0–3.4 | 2.5–3.5 | 90–130 | 7 |
| 7 | Heian Yondan | horquilla: frenada fuerte y 180° | polígono + un "dedo" de horquilla | 3.0–3.2 | 2.2–3 (punta 1.9–2.0) | 90–140 | 8 |
| 8 | Heian Godan | combinación a velocidad | recta larga + 1–2 horquillas + S | 2.8–3.2 | 2–2.8 (punta 1.8–2.0) | 100–150 | 8 |
| 9 | Tekki Shodan | precisión lateral en pasillo estrecho | eslalon de 3–4 desplazamientos, estrecho | 2.4–2.8 | 2.4–3.2 | 90–140 | 8 |
| 10 | Bassai Dai | todo, trazado desconocido | circuito aleatorio (8–11 vértices, reflejos, horquillas) | 2.6–3.2 | 1.8–2.6 | 140–220 | 8 |

Referencias de escala: `oval_small` mide 70 m y 3.0 m de ancho; Spielberg 2.2 m de ancho; el
coche gira con R ≥ 0.74 m. La anchura máxima se limita a 4.4 m porque el contador de vueltas
de `F110Env._check_done` tolera ±2 m laterales respecto a la pose de salida.

Métricas reales del set generado (media sobre las 10 pistas de entrenamiento de cada etapa):

| Etapa | Long. (m) | Ancho (m) | R mín (m) | Cambios de sentido | % recta | Índice dificultad |
|---|---|---|---|---|---|---|
| 1 Taikyoku Shodan | 63.7 | 4.23 | 7.21 | 0.0 | 21 | 0.27 |
| 2 Taikyoku Nidan | 92.5 | 4.00 | 5.49 | 4.0 | 39 | 0.37 |
| 3 Taikyoku Sandan | 105.4 | 3.82 | 4.97 | 7.2 | 35 | 0.45 |
| 4 Heian Shodan | 89.8 | 3.57 | 4.49 | 0.0 | 56 | 0.36 |
| 5 Heian Nidan | 103.6 | 3.39 | 3.67 | 2.0 | 63 | 0.43 |
| 6 Heian Sandan | 115.0 | 3.21 | 3.23 | 5.0 | 62 | 0.51 |
| 7 Heian Yondan | 111.1 | 3.12 | 1.96 | 5.2 | 55 | 0.68 |
| 8 Heian Godan | 123.9 | 2.99 | 1.90 | 6.2 | 55 | 0.71 |
| 9 Tekki Shodan | 123.4 | 2.58 | 2.96 | 11.6 | 44 | 0.68 |
| 10 Bassai Dai | 169.0 | 2.90 | 1.85 | 10.2 | 57 | 0.77 |

Índice de dificultad = `0.35·min(1, 2/Rmín) + 0.25·min(1, 2.2/ancho) + 0.25·min(1, giro_total/(2π)/6) + 0.15·min(1, cambios/12)`.
Es orientativo: el orden pedagógico (qué competencia se introduce) manda sobre el índice; por
eso Heian Shodan (0.36) queda por debajo de Taikyoku Sandan (0.45) aunque va después.

Ficheros: `maps/kata_XX_<slug>_NN/` (100 de entrenamiento), `maps/kata_XX_<slug>_e1/` (10 de
evaluación, semillas distintas, nunca se entrena con ellas), manifiesto
`maps/kata_curriculum.yaml`, métricas `docs/kata_tracks_metrics.csv`, galería
`docs/kata_tracks_gallery.png`. Regenerar: `python training_tracks/kata_trackgen.py`.

## 2. Generador (`training_tracks/kata_trackgen.py`)

Motor único: **polígono + radio de fillet por vértice**. Las familias solo construyen el
polígono; el motor lo convierte en una línea central C1 de rectas y arcos, así el radio mínimo
se controla analíticamente. Un candidato se rechaza si la línea central se autointerseca, si el
pasillo (buffer de ancho/2 + 0.3 m de pared) tiene más de un agujero o su área no cuadra con
`longitud × ancho` (solape), si algún fillet no cabe sin bajar del suelo de radio de la etapa
(la punta de horquilla tiene su propio suelo, `ancho/2 + 0.4`), o si longitud o caja no están en
rango. Convención de imagen: se dibuja en coordenadas mundo y se guarda volteada
verticalmente porque `Track.from_track_name` hace `FLIP_TOP_BOTTOM` al cargar. El punto 0 de la
línea central es el centro de la recta más larga; el sentido de giro se sortea por pista.

## 3. Cableado de entrenamiento

* `examples/multiagent/lib/multiagent_env.py`: opciones `curriculum`, `action_repeat`,
  `min_speed`/`max_speed`, `episode_timeout`, `reward_params`; `set_stage(k)`; cambio de mapa en
  caliente (`sim.set_map(Track)` + `env.track` + reconstruir `reset_fn`); progreso acumulado
  monótono por agente (`laps_completed`) independiente de la recompensa.
* `f1tenth_gym/envs/base_classes.py`: `ScanSimulator2D` pasa a ser **por instancia** de
  `RaceCar` (antes era atributo de clase). Con varios envs por proceso en mapas distintos, el
  simulador compartido devolvía scans del mapa equivocado.
* `examples/multiagent/lib/callbacks.py`: `CurriculumStats` (métricas `lap_success`,
  `laps_completed`, `curriculum_stage`, `episode_timed_out`) y la promoción en
  `MultipleAgentCallbacks.on_train_result`: se pasa a la etapa k+1 cuando
  `lap_success_mean ≥ 0.8` durante 3 resultados consecutivos tras al menos `min_steps` en la
  etapa, o al agotar `max_steps` (presupuesto de seguridad para que la noche recorra las 10
  etapas). El estado se persiste en `curriculum_state.json` del trial. Métricas en TensorBoard
  bajo `curriculum/*`.
* `examples/multiagent/run.py`: `evaluation.env` (env de evaluación con las 10 pistas `e1`
  fijas, salida fija, sin tope de velocidad), `evaluation.explore`, `training.plateau_stop`
  y bloque `reporting` en los parámetros del algoritmo.
* Config: `examples/multiagent/configs/experiments_kata.yaml`. Lanzador:
  `examples/multiagent/run_kata_overnight.sh [start|status|stop]`.

### Escala de la recompensa (`ProgressTimePenaltyEnv` con `action_repeat: 5`)

Un paso de RL son 0.05 s. A 3 m/s el progreso por paso es 0.15 m. Con `time_penalty 0.1` el
neto es positivo por encima de 2 m/s; congelarse cuesta −0.1/paso; con γ = 0.99 (horizonte
100 pasos = 5 s) el valor de conducir rápido es ≈ +15, el de congelarse ≈ −10 y el choque es
−20 terminal. Orden correcto: rápido > lento > congelarse > chocar. Con la escala anterior
(0.01 s por paso, penalti 0.02, choque −50, γ = 0.9787) el choque valía 50 veces más que una
conducción perfecta, lo que explica el colapso a "congelarse" observado en agosto.

### Confundidores controlados

PPO y SAC comparten env, recompensa, currículum, presupuesto por etapa (en pasos de RL) y set
de evaluación. Solo difiere el bloque de hiperparámetros del algoritmo. Falta el baseline sin
currículum (misma recompensa, `action_repeat 5`, solo `Bassai Dai`) para la ablación del
paper: es una entrada más en el YAML.

## 4. Monitorizar la noche

```bash
cd examples/multiagent && ./run_kata_overnight.sh status
```

```bash
cd examples/multiagent && ../../venv/bin/tensorboard --logdir models_kata
```

Señales a mirar: `curriculum/stage` (etapa actual), `custom_metrics/lap_success_mean` (la
métrica de promoción), `custom_metrics/average_speed_mean`, `evaluation/.../lap_success_mean`
(set fijo, comparable entre algoritmos y a lo largo del tiempo).

## 5. Limitaciones conocidas

* El render (`render_mode: human`) no sigue el cambio de mapa en caliente; los vídeos se
  generan con el script de evaluación externo.
* Con `num_envs_per_env_runner > 1` cada env carga sus propias pistas (caché LRU de 12
  `Track` por env, ~2 MB cada una).
* El contador de vueltas nativo de `F110Env` (`lap_counts`) sigue funcionando pero la métrica
  de promoción usa el progreso monótono del wrapper, que no depende de la puerta de salida.

## 6. Resultados de la primera noche (2026-09-11 → 12) y correcciones v2

Config: `experiments_kata.yaml`, PPO y SAC concurrentes (5 runners × 2 envs cada uno).
Curvas: `docs/kata_overnight_20260912_curves.png`. Evaluación por pista del checkpoint final de
PPO (determinista, sin tope de velocidad, 2 episodios × 2 agentes por pista):
`docs/kata_eval_20260912_ppo_final.csv` (script `examples/multiagent/kata_eval_tracks.py`).

| | PPO | SAC |
|---|---|---|
| Pasos de RL en la noche | 3.0 M (2.5 h, ~340 pasos/s) | 0.58 M (8.1 h, ~20 pasos/s) |
| Etapa alcanzada | 10 (Bassai Dai), 8 de 9 promociones por presupuesto | 5, todas por presupuesto |
| Éxito de vuelta, evaluación reservada (final) | **0.57–0.60**, 2.7–2.8 vueltas/episodio | 0.00 (congelado desde ~0.4 M pasos) |

Éxito por pista reservada, PPO final (fracción de agente-episodios con ≥ 1 vuelta; velocidad
media; tiempo de la primera vuelta):

| Kata | Éxito | Vueltas | Vel. (m/s) | 1ª vuelta (s) | Nota |
|---|---|---|---|---|---|
| 01 Taikyoku Shodan | 1.00 | 8.6 | 6.4 | 8.9 | sin choques, 59 m |
| 02 Taikyoku Nidan | 0.75 | 4.2 | 4.1 | 16.6 | |
| 03 Taikyoku Sandan | 1.00 | 5.0 | 3.5 | 17.3 | sin choques |
| 04 Heian Shodan | 1.00 | 2.3 | 3.9 | 25.7 | choca tras 2 vueltas |
| 05 Heian Nidan | 1.00 | 4.5 | 3.1 | 19.5 | sin choques |
| 06 Heian Sandan | 1.00 | 1.7 | 3.1 | 27.6 | choca tras 1–2 vueltas |
| 07 Heian Yondan | 0.00 | 0.1 | 0.1 | – | se para ante la horquilla |
| 08 Heian Godan | 0.00 | 0.4 | 0.7 | – | se para / choca |
| 09 Tekki Shodan | 0.00 | 0.3 | 0.8 | – | choca en el eslalon estrecho |
| 10 Bassai Dai | 0.00 | 0.2 | 4.0 | – | choca rápido |

El checkpoint de la época Taikyoku (iter 50) iba a 7–10 m/s y chocaba en todas (éxito 0.03):
el currículum convirtió "rápido y choca" en "vuelta tras vuelta a 3–6 m/s" en las seis
primeras katas. Las cuatro últimas (horquillas, pasillo estrecho, circuito aleatorio) no se
aprendieron: la política se detiene ante la horquilla en vez de tomarla.

### Diagnóstico

1. **Espacio de acción de velocidad.** El agente actúa sobre el rango del vehículo,
   [−5, 20] m/s. Con `normalize_actions`, la salida de la política en [−1, 1] se mapea a ese
   rango, pero solo [0, 8] es útil: ~70 % del rango es zona muerta (≤ 0 → parado) o
   inalcanzable (> tope de etapa). Consecuencias medidas: en PPO la media de la política se
   arrastra mientras el ruido de exploración recortado por los topes finge velocidad (a la
   iteración 50 la velocidad estocástica era 3.7 m/s y la determinista 7–10 m/s) y la entropía
   sube de 2.0 a 3.9 nats; en SAC la gaussiana comprimida cae en la zona muerta, `alpha`
   colapsa a 5·10⁻⁴ y la política se congela (`mean_q` ≈ −12, el valor de estar parado).
2. **Presupuesto por etapa.** 200k pasos no bastaron desde la etapa 3: 8 de 9 promociones
   fueron por presupuesto, así que las katas 7–10 se vieron con éxito de entrenamiento ≈ 0.05
   y sin señal útil.
3. **Rendimiento de SAC.** ~1 paso de gradiente por segundo con el buffer priorizado de 1 M
   (`num_agent_steps_trained` 14.7 M frente a 1.16 M muestreados): 20 pasos de entorno/s.

### Cambios v2 (`kata_v2_*` en `experiments_kata.yaml`)

* `speed_action_range: [0.5, 8.0]` en el wrapper: el espacio de acción del agente pasa a ser
  [0.5, 8] m/s (los topes de etapa siguen recortando por encima). El suelo de 0.5 m/s elimina
  "quedarse parado" como política.
* PPO: `free_log_std: true` (desviación independiente del estado), presupuesto de etapa
  400k, 5 M pasos en total.
* SAC: `MultiAgentReplayBuffer` uniforme de 500k, `training_intensity: 32`, 3 M pasos.

Lanzar la segunda noche:

```bash
cd examples/multiagent && KATA_EXPS="kata_v2_PPO_shared_ProgressTimePenalty kata_v2_SAC_shared_ProgressTimePenalty" ./run_kata_overnight.sh start
```

### Nota sobre evaluación externa de checkpoints

En Ray 2.46 `Policy.compute_single_action` devuelve la acción **normalizada** en [−1, 1]; la
desnormalización a unidades del entorno la hace el `RolloutWorker`. Cualquier evaluador que
cargue solo la `Policy` debe aplicar `unsquash_action(a, policy.action_space_struct)`
(`kata_eval_tracks.py` lo hace); sin ello el coche "se arrastra" a ≤ 1 m/s y los resultados
son falsos.

## 7. Tercera noche (2026-09-14 → 15): recompensa de Kohonda bajo el currículum v2

Lanzados a las 22:00: `kata_v2_PPO_shared_Kohonda` y `kata_v2_SAC_shared_Kohonda` (mismo env v2,
currículum y set de evaluación que `kata_v2_*`; solo cambia la recompensa), con
`kata_v2_PPO_shared_ProgressTimePenalty` encolado detrás de PPO Kohonda. La máquina se
reinició a las ~04:25: PPO Kohonda ya había terminado (5 M pasos, 5.8 h); los otros dos se
reanudaron a las 10:22 desde su último checkpoint y su `curriculum_state.json`.

**PPO + Kohonda**: promoción por métrica en las etapas 1–6 (todas antes de 0.8 M pasos),
por presupuesto en 7–9. Evaluación reservada: 1.00 de éxito y ~11 vueltas por episodio a
7.9 m/s en las iteraciones 350 y 450 (1.4–1.8 M pasos), luego oscila entre 0.55 y 0.90 al
entrenar en Bassai Dai con la entropía cayendo hasta −2.3 nats. Política final (iter 1250),
por pista reservada (`docs/kata_eval_20260915_ppo_kohonda_final.csv`):

| Kata | Éxito | Vueltas | Vel. (m/s) | 1ª vuelta (s) |
|---|---|---|---|---|
| 01 Taikyoku Shodan | 1.00 | 11.6 | 7.9 | 7.8 |
| 02 Taikyoku Nidan | 1.00 | 11.3 | 7.9 | 12.0 |
| 03 Taikyoku Sandan | 1.00 | 11.3 | 7.9 | 12.0 |
| 04 Heian Shodan | 1.00 | 11.0 | 8.0 | 13.7 |
| 05 Heian Nidan | 1.00 | 11.0 | 7.9 | 11.1 |
| 06 Heian Sandan | 0.50 | 5.6 | 6.9 | 13.4 |
| 07 Heian Yondan | 0.50 | 5.5 | 7.0 | 17.3 |
| 08 Heian Godan | 0.25 | 3.1 | 6.8 | 15.7 |
| 09 Tekki Shodan | 0.75 | 8.4 | 7.7 | 14.5 |
| 10 Bassai Dai | 0.00 | 0.3 | 6.8 | – |

Global 0.70 de éxito, 7.9 vueltas por agente-episodio, frente a 0.57 / 2.7 vueltas / ~3 m/s
de PPO + ProgressTimePenalty v1 (§6). El checkpoint de la iteración 470
(`..._iter470.csv`) toma la horquilla de Heian Yondan en el 100 % de los casos y Heian Godan
en el 75 %, pero falla en Taikyoku Shodan a esa velocidad; el mismo 0.70 global con otro
reparto. Lección: guardar checkpoints por métrica de evaluación, no por retorno de
entrenamiento (`checkpoint_score_attribute`), porque el mejor punto (iter 350–450) no se
conservó.

Confundidor pendiente: respecto a la noche 1 cambian a la vez la recompensa y el espacio de
acción v2. `kata_v2_PPO_shared_ProgressTimePenalty` (en curso) separa los dos efectos; a
0.63 M pasos ya estaba en la etapa 5 por métrica con 0.35 de éxito en evaluación.

**SAC + Kohonda**: a 0.4 M pasos (6.4 h) conduce a 3.8 m/s y choca a los ~4 s; 0.22 vueltas,
sin éxito; `alpha` vuelve a colapsar (0.21 → 0.01 en 120k pasos). Tercera variante
(congelarse, arrastrarse, correr y chocar) con el mismo síntoma de fondo: SAC deja de
explorar antes de encontrar la primera vuelta. Siguiente palanca: `target_entropy` fijo
(≈ −1) e `initial_alpha` alto con `alpha_lr` bajo. Reanudado para completar su presupuesto.

Vídeos: `examples/multiagent/eval_videos/kata_20260915_ppo_kohonda/`.

## 8. Matriz PPO cerrada y SAC v3 / v3b (2026-09-15)

`kata_v2_PPO_shared_ProgressTimePenalty` terminó (5 M pasos, 4.7 h). Evaluación reservada final
0.60 de éxito en RLlib y 0.68 / 6.3 vueltas / ~7 m/s en la evaluación externa por pista
(`docs/kata_eval_20260915_ppo_v2_ptp_final.csv`): vueltas en 8 de 10 katas, falla Heian Yondan
(horquilla) y Bassai Dai. Etapas 1–6 por métrica, 7–9 por presupuesto.

| PPO, política final | Éxito | Vueltas/episodio | Velocidad | Katas con vuelta |
|---|---|---|---|---|
| ProgressTimePenalty, acción v1 (noche 1) | 0.57 | 2.7 | ~3 m/s | 6/10 |
| ProgressTimePenalty, acción v2 | 0.68 | 6.3 | ~7 m/s | 8/10 |
| Kohonda, acción v2 | 0.70 (pico 1.00 en iter 350–450) | 7.9 | ~7.9 m/s | 9/10 |

El espacio de acción v2 explica el salto de velocidad; Kohonda añade éxito y, sobre todo,
aprendizaje más rápido (0.90 frente a 0.25 en evaluación a 1.6 M pasos).

**SAC v3** (`initial_alpha 1.0`, `target_entropy −1`, `alpha_lr 3e-5`): `alpha` ya no colapsa
(0.85 → 0.43 en 200k pasos), pero con la recompensa de Kohonda (~0.15 por paso) un `alpha` de
0.5 hace que el término de entropía domine: 1.5 m/s, choque a los 5 s, 0 vueltas, Q media
17–25 frente a retornos reales ~11. Parado a los 196k pasos.
**SAC v3b** (`initial_alpha 0.1`, resto igual) lanzado a las 16:58; `alpha` estable en 0.1 y Q
media coherente en las primeras iteraciones.
