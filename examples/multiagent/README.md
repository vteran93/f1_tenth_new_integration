# ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡ NECESITA ACTUALIZARSE!!!!!!!!!!!!!!!!!!!!

# F1TENTH Gym - Módulo Multiagente

Este documento proporciona una guía detallada para desarrolladores sobre cómo utilizar, evaluar y comprender el código dentro de la carpeta `multiagent`.

## Arquitectura e Implementación

El módulo `multiagent` está diseñado para entrenar y evaluar agentes de aprendizaje por refuerzo (RL) en un entorno de carreras de F1TENTH con múltiples vehículos. La arquitectura se basa en la biblioteca `ray[rllib]` para el entrenamiento distribuido y `click` para la interfaz de línea de comandos.

### Componentes Principales

-   **`run.py`**: Es el script principal para ejecutar el entrenamiento y la evaluación de los modelos. Utiliza una configuración basada en archivos YAML para definir los parámetros del entorno, el algoritmo y el entrenamiento.
-   **`run_hyperparameter_tuning.py`**: Este script se utiliza para realizar una búsqueda y optimización de hiperparámetros utilizando `ray.tune` y `Optuna`.
-   **`configs/`**: Este directorio contiene los archivos de configuración en formato YAML:
    -   `run.yaml`: Configuración principal que define qué otros archivos de configuración usar.
    -   `env.yaml`: Parámetros específicos del entorno de simulación (mapa, número de agentes, etc.).
    -   `ppo.yaml` / `sac.yaml`: Hiperparámetros para los algoritmos PPO y SAC.
    -   `hyperparameter_tuning.yaml`: Configuración para el script de ajuste de hiperparámetros.
-   **`lib/`**: Contiene módulos de utilidad, como `utils.py` para cargar configuraciones y `rewards.py` para definir funciones de recompensa personalizadas.

## Dependencias

Asegúrate de tener Python 3.8+ instalado. Las dependencias de Python se pueden instalar usando `pip`:

```bash
pip install "ray[rllib]" torch numpy pyyaml optuna
```

## Uso

Todos los comandos deben ejecutarse desde la raíz del repositorio.

### Entrenamiento

Para iniciar un nuevo entrenamiento, utiliza el script `run.py` con el flag `--train`.

```bash
python examples/multiagent/run.py --train
```

El script cargará la configuración definida en `examples/multiagent/configs/run.yaml` por defecto. Puedes especificar una ruta de configuración diferente con el argumento `--config_path`.

### Reanudar Entrenamiento

Si un entrenamiento se interrumpió, puedes reanudarlo desde el último punto de control (checkpoint) usando el flag `--resume`.

```bash
python examples/multiagent/run.py --train --resume
```

### Evaluación

Para evaluar un modelo ya entrenado, utiliza el flag `--eval`. El script cargará el mejor checkpoint del experimento especificado en el archivo de configuración y ejecutará la simulación en modo de renderizado.

```bash
python examples/multiagent/run.py --eval
```

### Ajuste de Hiperparámetros

Para buscar los mejores hiperparámetros para el algoritmo PPO, ejecuta el siguiente comando:

```bash
python examples/multiagent/run_hyperparameter_tuning.py --train
```

Este script utilizará el espacio de búsqueda y la configuración definidos en `examples/multiagent/configs/hyperparameter_tuning.yaml`.

## Parámetros de Configuración

Puedes modificar los archivos YAML en la carpeta `configs` para ajustar el comportamiento de los scripts.

### `run.yaml`

-   **`experiment_name`**: Nombre del experimento en `ray`.
-   **`storage_path`**: Directorio donde se guardarán los resultados y checkpoints.
-   **`env_config`**: Nombre del archivo de configuración del entorno (ej. `env.yaml`).
-   **`training.algorithm`**: Algoritmo a utilizar (`PPO` o `SAC`).
-   **`training.timesteps_total`**: Número total de pasos de tiempo para el entrenamiento.
-   **`training.reward_function`**: Nombre de la clase de la función de recompensa a utilizar (definida en `lib/rewards.py`).

### `env.yaml`

-   **`map`**: Nombre del mapa a utilizar (ej. `Spielberg`).
-   **`num_agents`**: Número de agentes en la simulación.
-   **`render_mode`**: Modo de renderizado (`human` para visual, `None` para entrenamiento).
-   ... y otros parámetros específicos del simulador F1TENTH.

### `ppo.yaml` / `sac.yaml`

Estos archivos contienen los hiperparámetros específicos de cada algoritmo de `rllib`, como:

-   **`lr`**: Tasa de aprendizaje (learning rate).
-   **`gamma`**: Factor de descuento.
-   **`train_batch_size`**: Tamaño del lote de entrenamiento.
-   **`model.fcnet_hiddens`**: Arquitectura de la red neuronal (capas y neuronas).


## 📊 Reward Functions Incluidas

### De `rewards.py`:
- **ProgressRewardEnv** - Recompensa basada en progreso en la pista
- **SpeedRewardEnv** - Recompensa basada en velocidad
- **SACBasicReward** - Recompensa básica para SAC
- **SACGeminiReward** - Recompensa mejorada para SAC
- **SpeedReward** - Recompensa enfocada en velocidad
- **SafetyReward** - Recompensa enfocada en seguridad

### De `rewards_pepe.py`:
- **GeminiReward** - Recompensa de progreso y supervivencia
- **SpeedReward** - Recompensa basada en velocidad
- **WaypointReward** - Recompensa por pasar waypoints
- **CompetitiveOvertakingReward** - Recompensa competitiva con adelantamientos

## 🗂️ Estructura de Salida

Los modelos se guardan en la siguiente estructura:

```
../models_batch/
├── batch_training_PPO_ProgressRewardEnv/
│   ├── experiment_config.json
│   ├── PPO_ProgressRewardEnv_*/
│   │   ├── checkpoint_*/
│   │   ├── events.out.tfevents.*
│   │   └── result.json
│   └── ...
├── batch_training_PPO_SpeedRewardEnv/
├── batch_training_SAC_ProgressRewardEnv/
├── batch_training_SAC_SpeedRewardEnv/
└── ...
```

### Checkpoints:
- Se guardan cada 10 iteraciones
- Se mantienen los 3 mejores checkpoints por modelo
- Checkpoint final siempre se guarda

## 🔍 Monitoreo de Progreso
 Poner como lanzar el tensorboard para ver el progreso del entrenamiento.
### Durante el entrenamiento:
Con el mismo tensorboard
### Analisis de resultados:
Para analizar los resultados de un entrenamiento, puedes usar el notebook `examples/multiagent/analysis.ipynb`. Este notebook te permite cargar los resultados de un experimento y visualizar métricas como la recompensa promedio, la velocidad y otros parámetros relevantes.