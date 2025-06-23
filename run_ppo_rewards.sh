#!/bin/bash

# Script para ejecutar pruebas sistemáticas de PPO con 4 recompensas en F1TENTH
# Ejecuta cada recompensa (progress, speed_track, crosstrack, talearning) con 800,000 pasos
# Continúa con la siguiente recompensa si una falla
# Genera logs y resultados en runs/ para TensorBoard

# Configuración inicial
PROJECT_DIR="/home/pepe/Desarrollo/f1_tenth_new_integration-ray-sergio/f1_tenth_new_integration-ray-sergio"
CONFIG_FILE="${PROJECT_DIR}/examples/config.yaml"
VENV_ACTIVATE="${PROJECT_DIR}/../venv/bin/activate"
PYTHON_SCRIPT="${PROJECT_DIR}/examples/multiagent_simplified_tensorboard.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Lista de recompensas a probar
REWARDS=("progress" "speed_track" "crosstrack" "talearning")

# Verificar que el entorno virtual existe
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "Error: Entorno virtual no encontrado en $VENV_ACTIVATE"
    exit 1
fi

# Verificar que config.yaml existe
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Archivo config.yaml no encontrado en $CONFIG_FILE"
    exit 1
fi

# Verificar que el script de Python existe
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Script Python no encontrado en $PYTHON_SCRIPT"
    exit 1
fi

# Activar entorno virtual
source "$VENV_ACTIVATE" || { echo "Error: No se pudo activar el entorno virtual"; exit 1; }

# Respaldar config.yaml
cp "$CONFIG_FILE" "${CONFIG_FILE}.bak_${TIMESTAMP}" || { echo "Error: No se pudo respaldar config.yaml"; exit 1; }
echo "Config.yaml respaldado como ${CONFIG_FILE}.bak_${TIMESTAMP}"

# Iterar sobre cada recompensa
for REWARD in "${REWARDS[@]}"; do
    echo "--------------------------------------------------"
    echo "Ejecutando PPO con recompensa: ${REWARD}"
    echo "Inicio: $(date)"
    echo "--------------------------------------------------"

    # Crear archivo de log para esta ejecución
    LOG_FILE="${PROJECT_DIR}/logs/ppo_${REWARD}_${TIMESTAMP}.log"
    mkdir -p "${PROJECT_DIR}/logs" || { echo "Error: No se pudo crear directorio de logs"; continue; }

    # Modificar config.yaml para la recompensa actual
    sed -i "s/mode: .*/mode: \"train\"/" "$CONFIG_FILE" || { echo "Error: No se pudo modificar mode en config.yaml"; continue; }
    sed -i "s/strategy: .*/strategy: \"${REWARD}\"/" "$CONFIG_FILE" || { echo "Error: No se pudo modificar strategy en config.yaml"; continue; }
    sed -i "s/total_timesteps: .*/total_timesteps: 800000/" "$CONFIG_FILE" || { echo "Error: No se pudo modificar total_timesteps en config.yaml"; continue; }

    # Verificar que las modificaciones se aplicaron
    echo "Contenido actual de config.yaml para ${REWARD}:"
    grep -E "mode|strategy|total_timesteps" "$CONFIG_FILE"

    # Ejecutar el entrenamiento y redirigir salida al log
    echo "Ejecutando: python ${PYTHON_SCRIPT} --config ${CONFIG_FILE}"
    python "$PYTHON_SCRIPT" --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 || {
        echo "Error: Falló la ejecución para ${REWARD}. Revisa ${LOG_FILE}"
        echo "Continuando con la siguiente recompensa..."
    }

    echo "Fin de ejecución para ${REWARD}: $(date)"
    echo "Resultados guardados en ${LOG_FILE}"
    echo "Logs de TensorBoard en ${PROJECT_DIR}/runs/multiagent_ppo_*"
    echo "--------------------------------------------------"
done

# Restaurar config.yaml original
mv "${CONFIG_FILE}.bak_${TIMESTAMP}" "$CONFIG_FILE" || echo "Error: No se pudo restaurar config.yaml"
echo "Config.yaml restaurado"

# Desactivar entorno virtual
deactivate || echo "No se pudo desactivar el entorno virtual"

echo "--------------------------------------------------"
echo "Pruebas completadas: $(date)"
echo "Revisa los logs en ${PROJECT_DIR}/logs/"
echo "Inicia TensorBoard con: tensorboard --logdir ${PROJECT_DIR}/runs/"
echo "--------------------------------------------------"