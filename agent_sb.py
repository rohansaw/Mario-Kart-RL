from pathlib import Path
from argparse import ArgumentParser
from multiprocessing.dummy import DummyProcess
import gym, gym_mupen64plus
import time
import os
from src.utils import next_free_port

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv
import wandb
from gym.envs import registry
from wandb.integration.sb3 import WandbCallback

# WANDB = False
os.environ['DISPLAY'] = ':0'

SEED = 123
def make_env(i, mario_kart_envs, video_record_frequency, video_store_path, input_port=8030, run=None, seed=SEED, **kwargs):
    Path(video_store_path).mkdir(parents=True, exist_ok=True)
    def f():
        # time.sleep(10 * i)
        # time.sleep(12 * i + 1.3 ** i)
        if run is not None:
            wandb.gym.monitor()
        env = gym.make(mario_kart_envs[i], input_port=input_port, use_wandb=(i==0), run=run, **kwargs)
        env.seed(seed + 2 ** i)
        check_env(env)
        env = Monitor(env)
        if i == 0:
            env = gym.wrappers.RecordVideo(env, video_store_path, name_prefix=f"env-{i}-rl-video", episode_trigger=lambda x: x % video_record_frequency == 0)
        return env
    set_random_seed(seed)
    return f

def main(args):
    # steps = 10_000_000
    # learning_rate = 3e-4
    # n_steps: int = 256
    # batch_size: int = 64
    # n_epochs: int = 10
    # gamma: float = 0.99
    # gae_lambda: float = 0.95

    wandb.require("service")

    config = {"project": "Stable-Baselines", "steps": args.steps, "learning_rate": args.lr, "n_epochs": args.n_epochs,
            "n_steps": args.n_steps, "gamma": args.gamma, "gae_lambda": args.gae_lambda, "batch_size": args.batch_size}


    if args.wandb:
        wandb_run = wandb.init(monitor_gym=True, config=config, sync_tensorboard=True)
        run_id = wandb_run.id
    else:
        run_id = "run"
        wandb_run = None
    
    args.n_steps = int(args.n_steps / args.num_envs)
    
    mario_kart_envs = [name for name in registry.env_specs.keys() if "Mario-Kart-Discrete" in name]
    print("available envs:", mario_kart_envs)
    start_port = next_free_port(8030)
    env = SubprocVecEnv([
        make_env(
            i,
            mario_kart_envs,
            args.video_record_frequency,
            args.video_record_path,
            input_port=start_port + i,
            run=wandb_run,
            random_tracks=args.random_tracks,
            auto_abort=args.auto_abort,
            num_tracks=args.num_tracks,
            containerized=args.containerized,
        )
    for i in range(args.num_envs)])
    env.reset()
    # print(env.render(mode="rgb_array"))

    if args.from_pretrained is None:
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=f"runs/{run_id}", learning_rate=args.lr, n_steps=args.n_steps,
                    gamma=args.gamma, gae_lambda=args.gae_lambda, batch_size=args.batch_size, n_epochs=args.n_epochs, seed=SEED)
    else:
        model = PPO.load(args.from_pretrained, env=env)

    model_store_path = Path(args.model_store_path) / run_id
    model_store_path.mkdir(parents=True, exist_ok=True)
    
    model.learn(total_timesteps=args.steps, callback=WandbCallback(verbose=2, model_save_path=model_store_path, model_save_freq=10000) if args.wandb else None)
    model.save(model_store_path / "best_model")
    wandb.save(str(model_store_path / "best_model_wandb"))
    if args.evaluate_after_training:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            print(rewards)
            wandb.log({"evaluation/rewards": rewards})
            env.render()
    else:
        env.close()
    print("FINISHED!")

if __name__ == "__main__":
    parser = ArgumentParser("stable baselines for mario kart")
    parser.add_argument("--wandb", action="store_true", default=False, help="toggles weather to log to wandb or not")
    parser.add_argument("--random-tracks", action="store_true", default=False, help="toggles weather to train model on random tracks")
    parser.add_argument("--containerized", action="store_true", default=False, help="toggles weather to abort episode if stuck")
    parser.add_argument("--auto-abort", action="store_true", default=False, help="toggles weather to abort episode if stuck")
    parser.add_argument("--num-tracks", type=int, default=2)
    parser.add_argument("--evaluate-after-training", action="store_true", default=False, help="toggles weather to evaluate model after training finished")
    parser.add_argument("--steps", type=int, default=10_000_000, help="number of steps to train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--video-record-frequency", type=int, default=10)
    parser.add_argument("--video-record-path", type=str, default="./recordings/", help="path to where recorded videos shall be stored")
    parser.add_argument("--model-store-path", type=str, default="./models/", help="path to where trained models shall be stored")
    parser.add_argument("--from-pretrained", type=str, default=None, help="path to pretrained model. If none, training new model")
    args = parser.parse_args()
    
    main(args)
