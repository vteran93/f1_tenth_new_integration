# Multi-Training Rewards Test Script

Este script (`multi_training_rewards.py`) permite probar todas las clases de reward de los archivos `rewards.py` y `rewards_pepe.py` con los algoritmos PPO y SAC.

## Características

- **Prueba automática**: Ejecuta entrenamientos con todas las combinaciones de algoritmos y reward functions
- **Configuración flexible**: Permite especificar timesteps, algoritmos específicos, y filtrar clases
- **Validación completa**: Verifica que cada reward function funcione correctamente con ambos algoritmos
- **Logging detallado**: Proporciona información sobre cada prueba ejecutada
- **Manejo de errores**: Captura y reporta errores de manera clara

## Reward Functions Incluidas

### De `rewards.py`:
- **MultiAgentF110 classes**: `ProgressRewardEnv`, `SpeedRewardEnv`
- **BaseReward classes**: `SACBasicReward`, `SACGeminiReward`, `SpeedReward`, `SafetyReward`

### De `rewards_pepe.py`:
- **RewardFunction classes**: `GeminiReward`, `SpeedReward`, `WaypointReward`, `CompetitiveOvertakingReward`

## Uso

### Ejecutar todas las pruebas (por defecto):
```bash
python multi_training_rewards.py
```

### Especificar número de timesteps:
```bash
python multi_training_rewards.py --timesteps 5000
```

### Probar solo un algoritmo específico:
```bash
python multi_training_rewards.py --algorithms PPO
python multi_training_rewards.py --algorithms SAC
python multi_training_rewards.py --algorithms PPO SAC
```

### Saltar clases específicas:
```bash
# Saltar clases de rewards_pepe.py
python multi_training_rewards.py --skip_pepe

# Saltar clases BaseReward de rewards.py
python multi_training_rewards.py --skip_base
```

### Pruebas rápidas (100 timesteps):
```bash
python multi_training_rewards.py --quick
```

### Combinaciones de opciones:
```bash
# Solo PPO, sin rewards_pepe.py, 2000 timesteps
python multi_training_rewards.py --algorithms PPO --skip_pepe --timesteps 2000

# Prueba rápida solo con SAC
python multi_training_rewards.py --algorithms SAC --quick
```

## Parámetros

- `--timesteps N`: Número de timesteps para cada entrenamiento (default: 1000)
- `--algorithms ALG [ALG ...]`: Algoritmos a probar (`PPO`, `SAC`, o ambos)
- `--skip_pepe`: Saltar pruebas de `rewards_pepe.py`
- `--skip_base`: Saltar pruebas de clases BaseReward de `rewards.py`
- `--quick`: Ejecutar pruebas rápidas con 100 timesteps

## Salida

El script proporciona:

1. **Logs en tiempo real**: Información sobre cada prueba en ejecución
2. **Resumen final**: Número total de pruebas, exitosas y fallidas
3. **Lista de errores**: Detalles sobre pruebas que fallaron
4. **Código de salida**: 0 si todas las pruebas pasaron, 1 si alguna falló

### Ejemplo de salida:
```
=== Testing PPO with ProgressRewardEnv (1000 timesteps) ===
✅ SUCCESS: PPO + ProgressRewardEnv (1000 timesteps)
=== Testing PPO with SpeedRewardEnv (1000 timesteps) ===
✅ SUCCESS: PPO + SpeedRewardEnv (1000 timesteps)
...

============================================================
TEST SUMMARY
============================================================
Total tests: 16
Passed: 15
Failed: 1
Failed tests:
  - SAC + CompetitiveOvertakingReward
============================================================
```

## Archivos Generados

- **Modelos**: Se guardan en `../models_test/ALGORITHM_REWARDFUNCTION_TIMESTEPS/`
- **Checkpoints**: Se mantienen solo los mejores para ahorrar espacio
- **Configuraciones temporales**: Se crean y eliminan automáticamente

## Requisitos

- Ray RLlib instalado y configurado
- Entorno F1TENTH funcional
- Archivos `rewards.py` y `rewards_pepe.py` en el directorio `lib/`
- Configuraciones base en `configs/ppo_config.yaml` y `configs/sac_config.yaml`

## Notas

- Las pruebas se ejecutan de forma secuencial para evitar conflictos de recursos
- Cada prueba es independiente y un fallo no afecta las demás
- Los timesteps se ajustan automáticamente para las configuraciones de entrenamiento
- El script es compatible con las arquitecturas de reward functions existentes
