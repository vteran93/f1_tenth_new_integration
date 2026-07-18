# Consolidación de ramas → `publicacion-paper`

**Fecha:** 2026-07-18
**Rama resultante:** `publicacion-paper` (derivada de `origin/main`)
**Tag de seguridad:** `backup/pre-consolidation-multiagent` → `e83c728` (estado de `multiagent` antes de empezar)

Objetivo: consolidar en una sola rama derivada de `main` todo el trabajo
rescatable disperso en ramas locales y de `origin`, para trabajar el paper
sobre `publicacion-paper`.

## Estructura de la consolidación

`publicacion-paper` se creó desde `origin/main` y se construyó con merges
`--no-ff` (un merge por rama rescatada, para trazabilidad):

```
publicacion-paper
├── Consolidate multiagent mainline        (= main + multiagent + working-tree salvage)
├── Merge origin/benchmark_integrations
├── Merge origin/mmarllib-sergio
├── Merge origin/tfm_implement_multi_agent
├── Merge kohonda_rewards
├── Merge origin/kohonda_simplify
├── Merge sac_experimental
├── Merge origin/training_tracks
└── Merge origin/tensorboard_validation
```

Rama intermedia `salvage/working-tree-multiagent` (local): contiene el rescate
del working tree sin commitear de `multiagent`.

## Working tree rescatado (sin basura)

Se commitearon los cambios trackeados pendientes (`experiments.yaml`, `run.py`)
y archivos útiles no trackeados: helpers `run_*.py`, `open_video.py`,
`render_best.py`, utilidades `nan_detection.py`/`safe_nan_detection.py`, docs
(incluido `BUG_REPORT.md`), notebooks e `instructions.md`.

**Excluidos como basura / secretos** (y añadidos a `.gitignore`):
`*:Zone.Identifier` (ADS de Windows), `Victor.conf` / `examples/Victor.conf`
(configs VPN — posibles credenciales), `vpnv/`, `*.mp4` (`best_episode.mp4`),
`*.zip`, `.claude/` (tooling local).

## Ramas incluidas

| Rama | Qué aportó | Resolución de conflictos |
|---|---|---|
| `origin/benchmark_integrations` | Integración `f1tenth_benchmarks` + deps | Limpio |
| `origin/mmarllib-sergio` | Setup multiagente, devcontainer, scripts de análisis | `.gitignore` por unión; `requirements.txt` curado conservado, freeze de Windows preservado como `requirements-windows.txt` |
| `origin/tfm_implement_multi_agent` | `terraform/` (infra cloud) + `train_cloud/` (SAC en nube) + `f1tenth_gym/kohonda/` | Limpio |
| `kohonda_rewards` | Configs SAC/Kohonda | `rewards.py` y `experiments.yaml` consolidados conservados; sus experimentos preservados como `configs/experiments_kohonda_sac.yaml` (typo `falseº`→`false`) |
| `origin/kohonda_simplify` | `Kohonda_experiments.yml`, `sac_hyperparameter_tunning.ipynb`, `requirements_debug.txt` | Se conservó `rewards.py`/`REWARDS.md` consolidados (se descartó typo `centerlPine`); se excluyó un `*:Zone.Identifier` |
| `sac_experimental` | `run_debug_*.py`, `tensorboard_test_sac.ipynb` | Se respetó la eliminación del script legacy `examples/multiagent_sac.py` |
| `origin/training_tracks` | Tracks (oval_small, figure8, complex_circuit, semi_rectangular) + scripts | Se **excluyeron** outputs: `training_tracks/models/` (checkpoints, 15M) y `training_tracks/runs/` (tensorboard, 3.3M) |
| `origin/tensorboard_validation` | Scripts SAC experimentales, `reward_policy.py`, `find_closest_point()` + limpieza PEP8 en `cubic_spline.py` | `.gitignore` por unión; eliminación legacy respetada |

## Ramas excluidas (con motivo)

| Rama | Motivo |
|---|---|
| `origin/ray-new-api` | Su único cambio es una edición divergente a `examples/multiagent_ppo.py` (que la mainline consolidada ya eliminó); no aporta archivos nuevos |
| `origin/evans_integration_pepe`, `origin/evans_integration_show_simulation_pepe` | Ramas huérfanas (historia no relacionada, sin merge-base); volcados del repo completo plagados de `*:Zone.Identifier` |
| `origin/deprecated` | Código marcado explícitamente como obsoleto |

Ramas ya **contenidas al 100%** en `multiagent` (0 commits únicos), no requirieron
merge: `bugfix_error`, `multi_agent_debug_ray`, `hyperparameter-tuning`,
`ray-sergio`, `rl_example`, `revert_multi_agent`, `sac_ray_victor`, `pepe-TFM`.

## Política de resolución de conflictos

1. **Código canónico protegido**: para la librería consolidada
   (`lib/rewards.py`, `lib/utils.py`, `experiments.yaml`, docs) se conservó la
   versión de `multiagent` (la referenciada por los configs y auditada en
   `BUG_REPORT.md`). Las variantes de las ramas se preservaron como archivos
   aparte cuando aportaban valor (p. ej. `experiments_kohonda_sac.yaml`).
2. **Scripts legacy** (`examples/multiagent_sac.py`, `examples/multiagent_ppo.py`):
   se respetó su eliminación en la mainline (superados por `examples/multiagent/`).
3. **Outputs no van a git**: checkpoints (`*/models/`) y logs (`*/runs/`, tfevents)
   se excluyeron y se añadieron a `.gitignore`.
4. **Configs de dependencias**: se preservaron ambas (`requirements.txt` curado +
   `requirements-windows.txt` fijado).

## Verificación

- Sin marcadores de conflicto en el árbol trackeado.
- `py_compile` OK en `rewards.py`, `multiagent_env.py`, `utils.py`, `callbacks.py`,
  `run.py`, `cubic_spline.py`.
- 0 archivos basura trackeados.
- ⚠️ Los tests de runtime (`examples/multiagent/tests/`) **no** se ejecutaron aquí
  porque `ray`/`f1tenth_gym` no están instalados en este entorno. Ejecutar en un
  entorno con dependencias antes de basar experimentos:
  `pytest examples/multiagent/tests/`.

## Cómo revertir

El estado previo de `multiagent` está en el tag `backup/pre-consolidation-multiagent`.
Las ramas originales permanecen intactas en `origin`.
