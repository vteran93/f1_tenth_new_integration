"""
Ejemplo de uso del entorno F1Tenth gym con el mapa Vegas (PNG/YAML)
Basado en waypoint_follow.py pero configurado específicamente para cargar vegas.png y vegas.yaml
"""

import time
from typing import Tuple
import os

import gymnasium as gym
import numpy as np
from numba import njit

from f1tenth_gym.envs.f110_env import F110Env

"""
Planner Helpers - Copiados de waypoint_follow.py
"""


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(
    point, radius, trajectory, t=0.0, wrap=False
):
    """
    Encuentra el primer punto en la trayectoria que intersecta con un círculo.
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)

    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start).astype(np.float32)

        a = np.dot(V, V)
        b = np.float32(2.0) * np.dot(V, start - point)
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - np.float32(2.0) * np.dot(start, point)
            - radius * radius
        )
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)

        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Calcula la actuación (velocidad y ángulo de dirección) para seguir un punto objetivo.
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)], dtype=np.float32),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


class SimplePlanner:
    """
    Planner simple para demostrar el uso del mapa Vegas
    """

    def __init__(self, track, wb):
        self.wheelbase = wb
        self.track = track
        # Usar la línea de carrera del track si está disponible
        if hasattr(track, 'raceline') and track.raceline is not None:
            self.waypoints = np.stack(
                [track.raceline.xs, track.raceline.ys, track.raceline.vxs]
            ).T
        else:
            # Crear waypoints básicos si no hay raceline
            self.waypoints = self._create_basic_waypoints()

        self.max_reacquire = 20.0
        self.lookahead_point = None
        self.current_index = None
        self.lookahead_point_render = None

    def _create_basic_waypoints(self):
        """
        Crea waypoints básicos para el mapa Vegas si no hay raceline disponible
        """
        # Waypoints básicos para demostración (pueden ser ajustados según el mapa)
        waypoints = np.array([
            [0.0, 0.0, 3.0],    # x, y, velocidad
            [5.0, 0.0, 3.0],
            [10.0, 2.0, 2.5],
            [10.0, 5.0, 3.0],
            [5.0, 7.0, 2.5],
            [0.0, 5.0, 3.0],
            [-2.0, 2.0, 2.5],
        ], dtype=np.float32)
        return waypoints

    def render_lookahead_point(self, e):
        """
        Renderiza el punto de look-ahead
        """
        if self.lookahead_point is not None:
            points = self.lookahead_point[:2][None]
            if self.lookahead_point_render is None:
                self.lookahead_point_render = e.render_points(
                    points, color=(255, 0, 0), size=3
                )
            else:
                self.lookahead_point_render.setData(points)

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        Obtiene el waypoint actual a seguir
        """
        wpts = waypoints[:, :2]
        lookahead_distance = np.float32(lookahead_distance)
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)

        if nearest_dist < lookahead_distance:
            t1 = np.float32(i + t)
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, t1, wrap=True
            )
            if i2 is None:
                return None, None
            current_waypoint = np.empty((3,), dtype=np.float32)
            current_waypoint[0:2] = wpts[i2, :]
            current_waypoint[2] = waypoints[i, -1]
            return current_waypoint, i
        elif nearest_dist < self.max_reacquire:
            current_waypoint = np.empty((3,), dtype=np.float32)
            current_waypoint[0:2] = wpts[i, :]
            current_waypoint[2] = waypoints[i, -1]
            return current_waypoint, i
        else:
            return None, None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        Planifica la actuación dado el estado actual
        """
        position = np.array([pose_x, pose_y])
        lookahead_point, i = self._get_current_waypoint(
            self.waypoints, lookahead_distance, position, pose_theta
        )

        if lookahead_point is None:
            return 2.0, 0.0  # Velocidad baja y sin giro si no hay waypoint

        # Guardar para rendering
        self.lookahead_point = lookahead_point
        self.current_index = i

        # Calcular actuación
        speed, steering_angle = get_actuation(
            pose_theta,
            self.lookahead_point,
            position,
            lookahead_distance,
            self.wheelbase,
        )
        speed = vgain * speed

        return speed, steering_angle


def main():
    """
    Función principal que demuestra cómo cargar el mapa Vegas
    """
    print("🏎️  Iniciando ejemplo con mapa Vegas (PNG/YAML)")

    # Configuración del planner
    work = {
        "mass": 3.463388126201571,
        "lf": 0.15597534362552312,
        "tlad": 0.82461887897713965 * 5,  # Lookahead distance reducida para el ejemplo
        "vgain": 0.8,  # Ganancia de velocidad reducida para mayor control
    }

    # Configuración del entorno
    num_agents = 1

    # Ruta absoluta al mapa Vegas (sin extensión)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    maps_dir = os.path.join(os.path.dirname(current_dir), "maps")
    vegas_map_path = os.path.join(maps_dir, "vegas")

    print(f"📍 Cargando mapa Vegas desde: {vegas_map_path}")
    print(f"   - Archivo YAML: {vegas_map_path}.yaml")
    print(f"   - Archivo PNG:  {vegas_map_path}.png")

    # Verificar que los archivos existen
    if not os.path.exists(f"{vegas_map_path}.yaml"):
        print("❌ Error: No se encontró vegas.yaml")
        return
    if not os.path.exists(f"{vegas_map_path}.png"):
        print("❌ Error: No se encontró vegas.png")
        return

    print("✅ Archivos del mapa encontrados")

    # Crear el entorno con el mapa Vegas
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": vegas_map_path,  # Ruta sin extensión
            "map_ext": ".png",      # Extensión de la imagen
            "num_agents": num_agents,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "model": "mb",
            "observation_config": {"type": "kinematic_state"},
            "params": F110Env.fullscale_vehicle_params(),
            "reset_config": {"type": "rl_random_static"},
            "scale": 1.0,
        },
        render_mode="human",
    )

    print("✅ Entorno creado exitosamente")

    # Obtener el track y crear el planner
    track = env.unwrapped.track
    print(f"📏 Información del mapa Vegas:")
    print(f"   - Resolución: {track.spec.resolution} m/pixel")
    print(f"   - Origen: {track.spec.origin}")
    print(f"   - Tamaño de imagen: {track.occupancy_grid.shape}")

    planner = SimplePlanner(
        track=track,
        wb=(
            F110Env.fullscale_vehicle_params()["lf"]
            + F110Env.fullscale_vehicle_params()["lr"]
        ),
    )

    # Agregar callbacks de renderizado
    if hasattr(track, 'raceline') and track.raceline is not None:
        env.unwrapped.add_render_callback(track.raceline.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_lookahead_point)

    # Inicializar el entorno
    obs, info = env.reset()
    done = False
    env.render()

    print("🚦 Iniciando simulación...")
    print("   - Usa ESC para cerrar la ventana")
    print("   - El coche seguirá los waypoints automáticamente")

    laptime = 0.0
    start = time.time()
    step_count = 0

    try:
        while not done:
            # Planificar acción para cada agente
            action = env.action_space.sample()  # Inicializar con acción aleatoria

            for i, agent_id in enumerate(obs.keys()):
                speed, steer = planner.plan(
                    obs[agent_id]["pose_x"],
                    obs[agent_id]["pose_y"],
                    obs[agent_id]["pose_theta"],
                    work["tlad"],
                    work["vgain"],
                )
                action[i] = np.array([steer, speed])

            # Ejecutar paso de simulación
            obs, step_reward, done, truncated, info = env.step(action)
            laptime += step_reward
            frame = env.render()

            step_count += 1

            # Mostrar información cada 100 pasos
            if step_count % 100 == 0:
                agent_id = list(obs.keys())[0]
                print(f"Paso {step_count}: x={obs[agent_id]['pose_x']:.2f}, "
                      f"y={obs[agent_id]['pose_y']:.2f}, "
                      f"θ={obs[agent_id]['pose_theta']:.2f}")

    except KeyboardInterrupt:
        print("\n🛑 Simulación interrumpida por el usuario")

    finally:
        env.close()
        print(f"\n📊 Estadísticas de la simulación:")
        print(f"   - Tiempo simulado: {laptime:.2f}s")
        print(f"   - Tiempo real: {time.time() - start:.2f}s")
        print(f"   - Pasos totales: {step_count}")
        print("✅ Ejemplo completado")


if __name__ == "__main__":
    main()
