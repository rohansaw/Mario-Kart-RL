from multiprocessing.dummy import DummyProcess
import gym, gym_mupen64plus
import time

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv
import wandb
from gym.envs import registry
from wandb.integration.sb3 import WandbCallback

# WANDB = False
WANDB = True
VIDEO_RECORD_FREQUENCY = 10

if __name__ == "__main__":
    steps = 10_000_000
    learning_rate = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95

    config = {"project": "Stable-Baselines", "steps": steps, "learning_rate": learning_rate, "n_epochs": n_epochs,
            "n_steps": n_steps, "gamma": gamma, "gae_lambda": gae_lambda, "batch_size": batch_size}

    if WANDB:
        run_id = wandb.init(monitor_gym=True, config=config, sync_tensorboard=True).id
    else:
        run_id = "run"
    mario_kart_envs = [name for name in registry.env_specs.keys() if "Mario-Kart-Discrete" in name]
    def make_env(i, seed=0):
        def f():
            env = gym.make(mario_kart_envs[i])
            env.seed(seed + 2 ** i)
            env = Monitor(env)
            env = gym.wrappers.RecordVideo(env, "./recordings", episode_trigger=lambda x: x % VIDEO_RECORD_FREQUENCY == 0)
            return env
        set_random_seed(seed)
        return f
    env = DummyVecEnv([make_env(0)])
    env.reset()

    def trigger(x):
        # print("got ", x)
        result = x % 5000 == 0
        if result:
            print("TRIGGER")
        return result

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=f"runs/{run_id}", learning_rate=learning_rate, n_steps=n_steps,
                gamma=gamma, gae_lambda=gae_lambda, batch_size=batch_size, n_epochs=n_epochs)
    model.learn(total_timesteps=steps, callback=WandbCallback(verbose=2, model_save_path="models/", model_save_freq=100000) if WANDB else None)
    model.save("models/mk__a2c_cnn_1kk_reset_impl")
    wandb.save(f"models/mk__a2c_cnn_1kk_reset_impl")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(rewards)
        wandb.log({"evaluation/rewards": rewards})
        env.render()
