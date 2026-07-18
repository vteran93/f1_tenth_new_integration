import gymnasium as gym
import numpy as np
import f1tenth_gym

# Configuración para usar la pista Spielberg
config = {
    'map': 'Spielberg',
    'num_agents': 1,
    'observation_config': {'type': 'original'}
}

# Cambia el id si lo modificaste en el registro
env = gym.make('victor-multi-agent-v0', render_mode='human', config=config)

obs, info = env.reset()
rewards = []
collisions = []
speed_history = []

print("=== PRUEBA DE POLÍTICA VICTOR ===")
print("Configuración:")
print("- Velocidad objetivo: 8.0 m/s (en rango óptimo 7-10 m/s)")
print("- Steering: 0.0 (línea recta)")
print("- Pasos: 1000")
print("- Formato de acción CORREGIDO: [steering, speed]")
print()

for step in range(1000):
    # CORREGIDO: F1TENTH usa formato [steering_angle, speed]
    # steering_angle: -0.4189 to 0.4189 rad, speed: -5.0 to 20.0 m/s
    if env.num_agents == 1:
        action = np.array([[0.0, 8.0]], dtype=np.float32)  # [steering, velocidad] - FORMATO CORRECTO
    else:
        action = np.full((env.num_agents, 2), [0.0, 8.0], dtype=np.float32)

    obs, reward, done, truncated, info = env.step(action)
    rewards.append(reward)

    # Obtener velocidad real del vehículo para debugging
    if hasattr(env.unwrapped, 'sim') and hasattr(env.unwrapped.sim, 'agents'):
        agent = env.unwrapped.sim.agents[0]
        actual_speed = np.sqrt(agent.standard_state['v_x']**2 + agent.standard_state['v_y']**2)
        speed_history.append(actual_speed)

    # Si quieres registrar colisiones:
    if hasattr(env, 'collisions'):
        collisions.append(np.copy(env.collisions))

    # Mostrar información cada 100 pasos
    if step % 100 == 0 and step > 0:
        print(f"Paso {step}: reward={reward:.3f}, velocidad_real={speed_history[-1]:.3f} m/s")

    if done or truncated:
        print(f"Episode finished at step {step}")
        break

print("\n=== RESULTADOS ===")
print(f"Recompensa total: {np.sum(rewards):.2f}")
print(f"Recompensa media por paso: {np.mean(rewards):.3f}")
if collisions:
    print(f"Colisiones totales: {np.sum(collisions)}")
if speed_history:
    print(f"Velocidad real promedio: {np.mean(speed_history):.3f} m/s")
    print(f"Velocidad real min/max: {np.min(speed_history):.3f}/{np.max(speed_history):.3f} m/s")
