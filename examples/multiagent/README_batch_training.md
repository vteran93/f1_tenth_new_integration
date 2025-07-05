# Batch Training Scripts para F1TENTH Reward Functions

Este conjunto de scripts permite entrenar automáticamente modelos con todas las funciones de reward disponibles en `rewards.py` y `rewards_pepe.py` usando algoritmos PPO y SAC.

## 📁 Archivos Incluidos

1. **`batch_train_all_rewards.py`** - Script principal para entrenar modelos
2. **`model_summary.py`** - Script para generar resúmenes de modelos entrenados
3. **`multi_training_rewards.py`** - Script de validación/testing (opcional)

## 🚀 Uso Principal

### Entrenar todos los modelos (5000 timesteps por defecto)

```bash
python batch_train_all_rewards.py
```

### Entrenar con configuraciones específicas

```bash
# Solo PPO con 5000 timesteps
python batch_train_all_rewards.py --algorithms PPO

# Solo SAC con 10000 timesteps
python batch_train_all_rewards.py --algorithms SAC --timesteps 10000

# Ambos algoritmos con timesteps personalizados
python batch_train_all_rewards.py --timesteps 7500

# Saltar clases específicas
python batch_train_all_rewards.py --skip_pepe --skip_base

# Continuar entrenamiento aunque falle algún modelo
python batch_train_all_rewards.py --continue_on_error

# Especificar directorio de salida
python batch_train_all_rewards.py --storage_path ../my_models
```

### Ver resumen de modelos entrenados

```bash
python model_summary.py

# Con directorio personalizado
python model_summary.py --storage_path ../my_models
```

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

## 📈 Configuraciones por Defecto

### Configuración PPO:
- **Timesteps**: 5000 (configurable)
- **Eval Interval**: Cada 5% del entrenamiento (mín. 250 pasos)
- **Train Batch Size**: Adaptado automáticamente
- **Learning Rate**: 0.00005
- **Shared Policy**: Sí (por defecto)

### Configuración SAC:
- **Timesteps**: 5000 (configurable)
- **Eval Interval**: Cada 5% del entrenamiento (mín. 250 pasos)
- **Target Entropy**: auto
- **Alpha LR**: 0.0003
- **Shared Policy**: Sí (por defecto)

### Configuración del Entorno:
- **Mapa**: oval_small
- **Número de Agentes**: 2
- **Timestep**: 0.01
- **Num Beams**: 36
- **Control Input**: [speed, steering_angle]

## 💾 Almacenamiento

### Espacio Estimado:
- **Por modelo**: ~50-200 MB (dependiendo del algoritmo y configuración)
- **Total estimado**: ~2-8 GB para todos los modelos

### Checkpoints:
- Se guardan cada 10 iteraciones
- Se mantienen los 3 mejores checkpoints por modelo
- Checkpoint final siempre se guarda

## 🔍 Monitoreo de Progreso

### Durante el entrenamiento:
```bash
# Monitorear logs en tiempo real
tail -f ../models_batch/batch_training_PPO_ProgressRewardEnv/*/result.json

# Ver resumen mientras entrena
python model_summary.py
```

### Después del entrenamiento:
```bash
# Resumen completo
python model_summary.py

# Verificar modelos específicos
ls -la ../models_batch/batch_training_PPO_*/*/checkpoint_*
```

## 🛠️ Opciones Avanzadas

### Variables de Entorno:
```bash
# Configurar número de CPUs para Ray
export RAY_NUM_CPUS=8

# Configurar memoria para Ray
export RAY_OBJECT_STORE_MEMORY=2000000000
```

### Configuración personalizada:
- Los scripts usan las configuraciones base de `configs/ppo_config.yaml` y `configs/sac_config.yaml`
- Para cambios permanentes, modifica estos archivos base
- Para cambios temporales, usa los parámetros de línea de comandos

## ⚠️ Consideraciones

### Recursos del Sistema:
- **RAM**: Mínimo 8GB recomendado (16GB+ para entrenamiento paralelo)
- **CPU**: Múltiples cores recomendados
- **GPU**: Opcional pero acelera significativamente
- **Almacenamiento**: 10GB+ libres recomendados

### Tiempo de Entrenamiento:
- **Por modelo**: 5-30 minutos (dependiendo del hardware)
- **Total estimado**: 2-10 horas para todos los modelos

### Manejo de Errores:
- Usa `--continue_on_error` para no parar el entrenamiento por fallos individuales
- Los logs se guardan por separado para cada modelo
- Los errores se reportan al final del proceso

## 📋 Ejemplos de Uso Completos

### Entrenamiento rápido para testing:
```bash
# Entrenar solo algunas clases con pocos pasos
python batch_train_all_rewards.py --algorithms PPO --skip_pepe --timesteps 1000
```

### Entrenamiento completo para producción:
```bash
# Entrenar todos los modelos con alta calidad
python batch_train_all_rewards.py --timesteps 10000 --continue_on_error
```

### Análisis post-entrenamiento:
```bash
# Ver resumen
python model_summary.py

# Evaluar modelo específico usando run.py
python run.py eval --config_path configs/ppo_config.yaml --run_name ProgressRewardEnv
```

## 🔧 Troubleshooting

### Problema: "No module named 'lib.rewards'"
**Solución**: Ejecutar desde el directorio `multiagent/`

### Problema: "CUDA out of memory"
**Solución**: Reducir `num_env_runners` en las configuraciones

### Problema: Ray no se inicializa
**Solución**: 
```bash
ray stop  # Parar Ray si está corriendo
python batch_train_all_rewards.py  # Reintentar
```

### Problema: Modelos no se guardan
**Solución**: Verificar permisos de escritura en `storage_path`

## 📞 Soporte

Para problemas específicos:
1. Revisar logs en el directorio de storage
2. Usar `model_summary.py` para diagnóstico
3. Verificar configuraciones base en `configs/`
