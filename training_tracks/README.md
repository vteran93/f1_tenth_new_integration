# F1TENTH Multi-Agent Training System

Este sistema implementa un entorno de entrenamiento multi-agente para F1TENTH con arquitectura polimórfica de recompensas y soporte para múltiples algoritmos de RL.

## 🏗️ Arquitectura del Sistema

### Principio de Sustitución de Liskov
El sistema sigue el principio de Liskov con una jerarquía de clases bien estructurada:

```
MultiAgentF110 (Clase Base)
├── MultiAgentF110PPO (Implementación PPO)
└── MultiAgentF110SAC (Implementación SAC)
```

### Sistema de Recompensas Polimórfico
Las funciones de recompensa están separadas en clases independientes controladas por diccionario:

```python
# Ejemplo de uso
reward_function = get_reward_function("ppo", "progress")
env = MultiAgentF110PPO(config, reward_function=reward_function)
```

## 📁 Estructura de Archivos

```
training_tracks/
├── multiagent_env.py           # Clases de entorno (Base, PPO, SAC)
├── rewards.py                  # Sistema polimórfico de recompensas
├── training_with_oval_track.py # Script de entrenamiento unificado
├── create_tracks.py            # Generador de pistas personalizadas
├── check_tracks.py            # Verificador de pistas
├── visualize_tracks.py        # Visualizador de pistas
├── basic_test.py              # Pruebas básicas del sistema
└── tracks/                    # Pistas generadas
    ├── oval_small/
    ├── semi_rectangular/
    ├── figure8_track/
    └── complex_circuit/
```

## 🚀 Uso del Sistema

### 1. Generar Pistas
```bash
# Generar todas las pistas personalizadas
python create_tracks.py

# Verificar pistas generadas
python check_tracks.py
```

### 2. Entrenar Modelos
```bash
# Entrenamiento PPO con pista oval y recompensa de progreso
python training_with_oval_track.py --train --algo ppo --track oval_small --reward progress --timesteps 100000

# Entrenamiento SAC con recompensa Gemini
python training_with_oval_track.py --train --algo sac --track semi_rectangular --reward gemini --timesteps 50000
```

### 3. Evaluar Modelos
```bash
# Evaluación del último modelo entrenado
python training_with_oval_track.py --algo ppo --track oval_small --reward progress --episodes 5
```

### 4. Ver Opciones Disponibles
```bash
# Listar todas las funciones de recompensa
python training_with_oval_track.py --list-rewards

# Ver ayuda completa
python training_with_oval_track.py --help
```

## 🏆 Funciones de Recompensa Disponibles

### PPO
- **default/progress**: `PPOProgressReward` - Recompensa basada en progreso directo
- **speed**: `SpeedReward` - Combina progreso con velocidad
- **safety**: `SafetyReward` - Enfoque en conducción segura

### SAC
- **default/basic**: `SACBasicReward` - Recompensa básica de progreso
- **gemini**: `SACGeminiReward` - Recompensa mejorada con supervivencia
- **speed**: `SpeedReward` - Combina progreso con velocidad
- **safety**: `SafetyReward` - Enfoque en conducción segura

### Compartidas
- **default/progress**: `DefaultProgressReward` - Implementación base
- **speed**: `SpeedReward` - Velocidad + progreso
- **safety**: `SafetyReward` - Seguridad + progreso

## 🛣️ Pistas Disponibles

1. **oval_small**: Pista ovalada básica (800x600px, 40x30m)
2. **semi_rectangular**: Pista semi-rectangular con curvas de 115° (1000x600px, 40x24m)
3. **figure8_track**: Pista en forma de 8 (800x600px, 40x30m)
4. **complex_circuit**: Circuito complejo con múltiples curvas (1000x800px, 40x32m)

## 🔧 Configuración de Entornos

### Diferencias entre PPO y SAC

| Aspecto | PPO | SAC |
|---------|-----|-----|
| Observaciones | Normalizadas [-1,1] | Valores originales |
| Espacio de Acción | Extraído dinámicamente | Fijo [-1,1] x [0,10] |
| Reset Config | `cl_grid_static` | `rl_random_static` |
| Espacios | Compartidos entre agentes | Individual por agente |

## 🧪 Testing

### Pruebas Básicas
```bash
python basic_test.py
```

### Pruebas Completas (requiere f1tenth_gym)
```bash
python test_system.py
```

## 📊 Logging y Monitoreo

Los entrenamientos generan logs de TensorBoard en:
```
runs/multiagent_{algorithm}_{timestamp}/
```

Para visualizar:
```bash
tensorboard --logdir=runs/
```

## 🏁 Ejemplo de Entrenamiento Completo

```bash
# 1. Generar pistas
python create_tracks.py

# 2. Verificar sistema
python basic_test.py

# 3. Entrenar PPO en pista oval
python training_with_oval_track.py --train --algo ppo --track oval_small --reward progress --timesteps 50000

# 4. Evaluar modelo entrenado
python training_with_oval_track.py --algo ppo --track oval_small --reward progress

# 5. Entrenar SAC con recompensa avanzada
python training_with_oval_track.py --train --algo sac --track semi_rectangular --reward gemini --timesteps 30000
```

## 🎯 Características del Sistema

### ✅ Implementado
- [x] Clase base `MultiAgentF110` con funcionalidad común
- [x] Clases específicas `MultiAgentF110PPO` y `MultiAgentF110SAC`
- [x] Sistema polimórfico de recompensas con diccionario de control
- [x] Script de entrenamiento unificado con parámetros CLI
- [x] Generador de pistas personalizadas
- [x] Principio de Sustitución de Liskov
- [x] Soporte para múltiples tipos de recompensa
- [x] Configuraciones específicas por algoritmo

### 🎨 Extensible
- Fácil agregar nuevas funciones de recompensa
- Soporte para nuevos algoritmos
- Sistema de pistas personalizable
- Arquitectura modular y limpia

## 🔄 Polimorfismo en Acción

```python
# El entorno acepta cualquier función de recompensa que implemente _get_rewards()
reward_function = get_reward_function("sac", "gemini")
env = MultiAgentF110SAC(config, reward_function=reward_function)

# Cambiar recompensa dinámicamente
new_reward = get_reward_function("sac", "speed")
env.reward_function = new_reward
```

Este sistema permite experimentar fácilmente con diferentes combinaciones de algoritmos, pistas y funciones de recompensa para optimizar el entrenamiento de agentes F1TENTH.
