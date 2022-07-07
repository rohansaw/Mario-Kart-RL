import gym, gym_mupen64plus

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
import wandb
from wandb.integration.sb3 import WandbCallback
# Parallel environments
env = gym.make('Mario-Kart-Discrete-Luigi-Raceway-v0')
check_env(env)
wandb.init(monitor_gym=True)
model = A2C("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10_000_000, callback=WandbCallback())
model.save("models/mk__a2c_cnn_1kk_reset_impl")
#model = A2C.load("models/mk__a2c_cnn_2_no_cp")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    env.render()