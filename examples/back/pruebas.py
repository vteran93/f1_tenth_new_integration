import gymnasium as gym
env = gym.make('f110_gym:f110-v0')
env.reset()
env.render()
input("¿Ves la ventana? Pulsa Enter para salir...")