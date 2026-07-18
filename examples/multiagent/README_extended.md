# Extended F1TENTH Multi-Agent Evaluation

Este archivo extiende la funcionalidad de `run.py` siguiendo el principio Open/Closed de SOLID para agregar capacidades de evaluación masiva.

## Características Nuevas

### Evaluación Masiva (`run_mass_evaluation`)

- **100,000 episodios** de evaluación sin render para máxima velocidad
- **Detección automática del mejor episodio** basado en criterios múltiples:
  1. Mayor progreso máximo en la pista (max_lap_progress)
  2. Mayor retorno (reward total)
  3. Menos colisiones
  4. Menor longitud de episodio
- **Métricas agregadas** guardadas en JSON
- **Grabación opcional** de la mejor trayectoria en JSON y video MP4
- **Semillas reproducibles** para evaluación consistente

## Configuración

El archivo `experiments.yaml` ha sido extendido con nuevos parámetros de evaluación masiva:

```yaml
evaluation: &evaluation
  episodes: 100                # Episodios para evaluación estándar
  # Mass evaluation configuration
  mass_episodes: 100000        # Episodios para evaluación masiva
  record_best: true           # Guardar mejor episodio
  seed_base: 0               # Semilla base para reproducibilidad
  render_best: false         # Renderizar video del mejor episodio
```

## Uso

### Evaluación Masiva

```bash
# Evaluar el mejor checkpoint de un experimento
python run_extended.py mass-eval --experiment oval_small_PPO_Individual_Policy_ProgressRewardAdvanced

# Evaluar un trial específico
python run_extended.py mass-eval --experiment oval_small_PPO_Individual_Policy_ProgressRewardAdvanced --trial trial_123
```

### Archivos de Salida

La evaluación masiva genera los siguientes archivos en `<storage_path>/<experiment_name>/<date>/eval_runs/`:

- `seeds.json`: Lista de semillas utilizadas
- `mass_eval_summary.json`: Métricas agregadas de todos los episodios
- `best_episode.json`: Trayectoria detallada del mejor episodio
- `best_episode.mp4`: Video del mejor episodio (si `render_best: true`)

### Ejemplo de `mass_eval_summary.json`

```json
{
  "episodes": 100000,
  "return_mean": 1234.56,
  "return_std": 123.45,
  "length_mean": 567.8,
  "length_std": 56.78,
  "max_lap_progress_mean": 0.856,
  "max_lap_progress_std": 0.123,
  "successes": 85000,
  "collisions": 1500,
  "best": {
    "max_lap_progress": 1.000,
    "return": 2000.0,
    "length": 400,
    "collisions": 0,
    "seed": 1337
  },
  "timestamp": 1697123456
}
```

## Criterio de Mejor Episodio

El mejor episodio se determina por prioridad jerárquica:

1. **Máximo progreso en la pista** (`max_lap_progress`): Prioridad más alta - qué tan lejos llegó el agente
2. **Máximo retorno**: En caso de empate en progreso
3. **Mínimas colisiones**: En caso de empate en retorno
4. **Mínima longitud**: En caso de empate en colisiones

Esto asegura que se prioricen episodios donde los agentes completen más distancia en la pista, incluso si tienen menor reward total.

## Dependencias Adicionales

Para la funcionalidad de video (opcional):

```bash
pip install imageio[ffmpeg]
```

## Arquitectura

El archivo `run_extended.py` sigue el principio Open/Closed:

- **Abierto para extensión**: Nuevas funcionalidades sin modificar código existente
- **Cerrado para modificación**: El archivo original `run.py` permanece intacto
- **Reutilización**: Importa y reutiliza funciones existentes de `run.py`

## Integración con CI/CD

Puede integrarse en pipelines de CI/CD para evaluación automática:

```bash
# Ejemplo de pipeline
python run.py train --experiment my_experiment
python run_extended.py mass-eval --experiment my_experiment
```

## Casos de Uso

1. **Benchmarking**: Evaluación exhaustiva de modelos entrenados
2. **Selección de modelos**: Identificar el mejor episodio para análisis
3. **Validación**: Verificar rendimiento en gran escala
4. **Análisis de robustez**: Evaluación bajo múltiples semillas
5. **Creación de demos**: Grabar el mejor episodio para presentaciones