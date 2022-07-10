import gym, gym_mupen64plus

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np
from wandb.integration.sb3 import WandbCallback


steps = 10_000_000
run = wandb.init(monitor_gym=True, config={"project": "Stable-Baselines", "steps": steps}, sync_tensorboard=True)

class RewardLogger(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # print(self.locals.keys())
        # print(self.locals["env"].envs[0].last_episode_reward)
        if self.n_calls % 500 == 0:
            wandb.log({"reward": np.sum(self.locals["rewards"])}, step=self.locals["env"].envs[0].reset_count)
            # wandb.log({"reward": self.locals["env"].envs[0].last_episode_reward}, step=self.locals["env"].envs[0].reset_count)
        return True

def make_env():
    env = gym.make('Mario-Kart-Discrete-Luigi-Raceway-v0')
    env = Monitor(env)
    return env
# Parallel environments
env = DummyVecEnv([make_env])
env.reset()
# print(env.render(mode="rgb_array"))

# check_env(env)
env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 10000 == 0, video_length=1250)

model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(total_timesteps=steps, callback=[RewardLogger(), WandbCallback(verbose=2, gradient_save_freq=5000, log="all")])
# model.learn(total_timesteps=steps, callback=[RewardLogger, WandbCallback(verbose=2, gradient_save_freq=5000, log="all")])
model.save("models/mk__a2c_cnn_1kk_reset_impl")
#model = A2C.load("models/mk__a2c_cnn_2_no_cp")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    wandb.log({"evaluation/rewards": rewards})
    env.render()
