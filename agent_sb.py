import gym, gym_mupen64plus

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

# Parallel environments
env = gym.make('Mario-Kart-Discrete-Luigi-Raceway-v0')
check_env(env)

model = A2C("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("mk__a2c_cnn")
#model = A2C.load("a2c_cartpole")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    #env.render()