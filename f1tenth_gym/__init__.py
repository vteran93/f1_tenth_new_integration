import gymnasium as gym

gym.register(
    id="f1tenth-v0",
    entry_point="f1tenth_gym.envs:F110Env",
)


gym.register(
    id="kohonda-multi-agent-v0",
    entry_point="f1tenth_gym.kohonda:KohondaMultiAgentF110Env",
)


gym.register(
    id="victor-multi-agent-v0",
    entry_point="f1tenth_gym.kohonda:VictorMultiAgentEnv",
)
