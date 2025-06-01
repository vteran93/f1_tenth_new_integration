# SAC Training with Incremental Checkpoints

## Modificaciones Realizadas

He modificado el script `sac_example.py` para implementar un sistema de guardado incremental que permite retomar el entrenamiento desde donde se dejó si el computador se apaga.

### Nuevas Funcionalidades

1. **Guardado Automático cada 100,000 pasos**: El modelo se guarda automáticamente cada 100,000 pasos de entrenamiento.

2. **Detección Automática de Checkpoints**: Al iniciar el script, busca automáticamente el último checkpoint disponible.

3. **Reanudación Inteligente**: Si encuentra un checkpoint, retoma el entrenamiento desde ese punto exacto.

4. **Continuidad de WandB**: Si existe un run_id previo, mantiene la continuidad en los logs de WandB.

### Archivos y Directorios Creados

```
examples/
├── models/                          # Modelos guardados incrementalmente
│   ├── sac_checkpoint_100000.zip    # Checkpoint a los 100k pasos
│   ├── sac_checkpoint_200000.zip    # Checkpoint a los 200k pasos
│   ├── sac_checkpoint_300000.zip    # Checkpoint a los 300k pasos
│   └── sac_final_<run_id>.zip       # Modelo final
├── checkpoints/                     # Información de checkpoints
│   └── latest_checkpoint.json       # Último checkpoint guardado
└── runs/                           # Logs de TensorBoard
    └── <run_id>/
```

### Cómo Funciona

1. **Primera Ejecución**: 
   - Inicia un nuevo entrenamiento desde cero
   - Crea directorios necesarios
   - Guarda checkpoints cada 100,000 pasos

2. **Ejecución Después de Interrupción**:
   - Detecta automáticamente el último checkpoint
   - Carga el modelo desde ese punto
   - Calcula pasos restantes para completar el entrenamiento
   - Continúa desde donde se dejó

### Uso

```bash
# Iniciar o continuar entrenamiento
python sac_example.py

# Ver información de checkpoints
python checkpoint_demo.py
```

### Componentes Principales Añadidos

1. **`IncrementalSaveCallback`**: Callback personalizado que guarda el modelo cada 100,000 pasos.

2. **`find_latest_checkpoint()`**: Busca el archivo de checkpoint más reciente.

3. **`find_latest_model()`**: Como respaldo, busca el modelo más reciente si no hay archivo de checkpoint.

4. **Lógica de Reanudación**: Detecta checkpoints existentes y retoma el entrenamiento.

### Ventajas

- ✅ **Resistente a fallos**: El entrenamiento puede continuar después de interrupciones
- ✅ **Sin pérdida de progreso**: No se pierde tiempo de entrenamiento
- ✅ **Automático**: No requiere intervención manual
- ✅ **Compatible con WandB**: Mantiene continuidad en los logs
- ✅ **Flexible**: Funciona tanto para entrenamientos nuevos como reanudados

### Configuración

Puedes modificar la frecuencia de guardado cambiando la variable `save_freq`:

```python
save_freq = 100_000  # Cambiar a 50_000 para guardar cada 50k pasos
```

El script mantiene toda la funcionalidad original pero añade estas capacidades de recuperación automática.
