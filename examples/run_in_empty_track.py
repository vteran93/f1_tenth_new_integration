import numpy as np
from waypoint_follow import PurePursuitPlanner
from f1tenth_gym.envs.track import Track
import gymnasium as gym


def main():
    """
    Demonstrate the creation of an empty map with a custom reference line.
    This is useful for testing and debugging control algorithms on standard maneuvers.
    """

    # Evitamos escalas absurdas
    track_length = 20  # Máximo 20 metros (antes: 100)
    resolution = 0.1   # Esto es solo referencia para calcular velxs, no lo pasamos al Track porque no lo acepta

    # Línea sinusoidal suave
    xs = np.linspace(0, track_length, int(track_length / resolution))
    ys = np.sin(xs / 2.0) * 2.0  # Altura moderada
    velxs = 3.0 * (1 + np.abs(np.cos(xs / 2.0)))  # Velocidad basada en curvatura

    # Crea pista desde refline (Track se encarga del resto)
    track = Track.from_refline(x=xs, y=ys, velx=velxs)

    # Inicializa entorno
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": track,
            "num_agents": 1,
            "observation_config": {"type": "kinematic_state"},
        },
        render_mode="human",
    )

    planner = PurePursuitPlanner(track=track, wb=0.17145 + 0.15875)

    env.unwrapped.add_render_callback(track.raceline.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_lookahead_point)

    obs, info = env.reset()
    done = False
    env.render()

    while not done:
        speed, steer = planner.plan(
            obs["agent_0"]["pose_x"],
            obs["agent_0"]["pose_y"],
            obs["agent_0"]["pose_theta"],
            lookahead_distance=0.8,
            vgain=1.0,
        )
        action = np.array([[steer, speed]])
        obs, timestep, terminated, truncated, infos = env.step(action)
        done = terminated or truncated
        env.render()

    env.close()


if __name__ == "__main__":
    main()
