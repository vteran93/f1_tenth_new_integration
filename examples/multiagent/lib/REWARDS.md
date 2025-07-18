# Comparativa Conceptual de Funciones de Recompensa

| Recompensa                   | ¿Qué incentiva principalmente?         | ¿Cómo penaliza?                  | Complejidad | Personalización | Ideal para...                         | Observaciones clave                                      |
|------------------------------|----------------------------------------|----------------------------------|-------------|-----------------|----------------------------------------|---------------------------------------------------------|
| **ProgressRewardEnv**        | Avance a lo largo del circuito         | Penaliza colisión levemente      | Muy baja    | Baja            | Baseline, agentes simples              | Solo mide progreso, no considera velocidad ni seguridad  |
| **ProgressRewardAdvancedEnv**| Progreso rápido y supervivencia        | Penaliza colisión fuertemente    | Baja        | Media           | Agentes que deben evitar choques        | Escala el progreso, bonus por sobrevivir                |
| **SpeedRewardEnv**           | Velocidad y movimiento constante       | Penaliza colisión fuertemente    | Media       | Media           | Agentes que deben ir rápido             | Usa la mayor de dos velocidades, bonus por alta velocidad|
| **WaypointRewardEnv**        | Progreso estructurado y precisión      | Penaliza colisión y desviación   | Media       | Alta            | Aprendizaje de trayectorias             | Penaliza alejarse del centro, bonus por vueltas         |
| **SafetyRewardEnv**          | Seguridad y evitar obstáculos          | Penaliza colisión fuertemente    | Media       | Media           | Entornos peligrosos o estrechos         | Usa LiDAR para premiar distancia a obstáculos           |
| **KohondaMultiAgentF110Env** | Progreso basado en raceline            | Penaliza colisión fijo o escalado| Media       | Media           | Seguimiento de trayectoria óptima       | Usa raceline en lugar de centerlPine, progreso por distancia entre waypoints |

---

## Resumen conceptual

- **ProgressRewardEnv**: Muy básica, ideal para pruebas iniciales o como baseline.
- **ProgressRewardAdvancedEnv**: Similar a la anterior pero más estricta con los choques y recompensa la supervivencia.
- **SpeedRewardEnv**: Incentiva ir rápido, pero puede ser riesgoso si no se combina con seguridad.
- **WaypointRewardEnv**: Incentiva seguir el camino correcto y penaliza desviaciones, útil para agentes que deben aprender trayectorias.
- **SafetyRewardEnv**: Ideal para situaciones donde la seguridad es prioritaria, usa sensores para evitar riesgos.
- **KohondaMultiAgentF110Env**: Enfocada en seguir la trayectoria de carrera óptima (raceline), con penalización adaptativa por colisión y progreso basado en waypoints de la línea de carrera.