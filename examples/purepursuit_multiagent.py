
import numpy as np
from f1tenth_gym.envs import F110Env
from planners import PurePursuitPlanner

def main():
    env = F110Env(config={'map': 'Spielberg', 'num_agents': 2, 'render_mode': 'human'})
    planner = PurePursuitPlanner(
        map_path="../maps/Spielberg/Spielberg_raceline.csv",
        lookahead_distance=1.5,
        wheelbase=0.33,
        max_steer=0.5
    )
    obs, info = env.reset()
    done = False
    while not done:
        actions = {}
        for agent_id in range(2):
            x, y, theta = obs['poses_x'][agent_id], obs['poses_y'][agent_id], obs['poses_theta'][agent_id]
            speed = 5.0
            speed, steer = planner.get_control(x, y, theta, speed)
            actions[f"agent_{agent_id}"] = np.array([speed, steer])
        obs, _, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        env.render()
    env.close()

if __name__ == "__main__":
    main()