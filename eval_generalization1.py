from pathlib import Path
from argparse import ArgumentParser
from multiprocessing.dummy import DummyProcess
import random
import gym, gym_mupen64plus
import time
import os

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gym_mupen64plus.envs.MarioKart64.mario_kart_env import MarioKartEnv
import wandb
from gym.envs import registry
from wandb.integration.sb3 import WandbCallback

# WANDB = False
os.environ['DISPLAY'] = ':0'

SEED = 123
def make_env(i, mario_kart_envs, video_record_frequency, video_store_path, seed=SEED, **kwargs):
    Path(video_store_path).mkdir(parents=True, exist_ok=True)
    def f():
        time.sleep(10 * i)
        # time.sleep(12 * i + 1.3 ** i)
        env = gym.make(mario_kart_envs[i], input_port=8030 + i, **kwargs)
        env.seed(seed + 2 ** i)
        check_env(env)
        env = Monitor(env)
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

    config = {"project": "Stable-Baselines", "steps": args.steps, "steps_env_switch": args.steps_env_switch, "learning_rate": args.lr, "n_epochs": args.n_epochs,
            "n_steps": args.n_steps, "gamma": args.gamma, "gae_lambda": args.gae_lambda, "batch_size": args.batch_size}


    if args.wandb:
        run_id = wandb.init(monitor_gym=True, config=config, sync_tensorboard=True).id
    else:
        run_id = "run"
    
    mario_kart_envs = [name for name in registry.env_specs.keys() if "Mario-Kart-Discrete" in name]
    print("available envs:", mario_kart_envs)
    
    all_tracks = list(MarioKartEnv.COURSES.keys())
    random.seed(123)
    random.shuffle(all_tracks)
    num_t_tracks = max(args.num_tracks, len(MarioKartEnv.COURSES) - 2)
    training_tracks = all_tracks[:num_t_tracks]
    validation_track = all_tracks[-1]
    env = SubprocVecEnv([
        make_env(
            i,
            mario_kart_envs,
            args.video_record_frequency,
            args.video_record_path,
            training_tracks=training_tracks,
            random_tracks=True,
            auto_abort=args.auto_abort,
            num_tracks=0,
            containerized=args.containerized,
        )
    for i in range(args.num_envs)])
    env.reset()
    # print(env.render(mode="rgb_array"))

    eval_env = SubprocVecEnv([
        make_env(
            i,
            mario_kart_envs,
            args.video_record_frequency,
            args.video_record_path,
            training_tracks=[validation_track],
            random_tracks=True,
            auto_abort=args.auto_abort,
            num_tracks=0,
            containerized=args.containerized
        )
    for i in range(0,1)])

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=f"runs/{run_id}", learning_rate=args.lr, n_steps=args.n_steps,
                gamma=args.gamma, gae_lambda=args.gae_lambda, batch_size=args.batch_size, n_epochs=args.n_epochs, seed=SEED)

    model_store_path = Path(args.model_store_path) / run_id
    model_store_path.mkdir(parents=True, exist_ok=True)

    for i in range(0, int(args.steps / 10000)):
        env.reset()
        model.learn(total_timesteps=10000, callback=WandbCallback(verbose=2, model_save_path=model_store_path, model_save_freq=10000) if args.wandb else None)
        eval_env.reset()
        res = evaluate_policy(model, eval_env)
        wandb.log({"eval/mean_reward" : res, "i": i})

    model.save(model_store_path / "eval_gen_model")
    wandb.save(str(model_store_path / "eval_gen_model_wandb"))
    env.close()
    eval_env.close()

if __name__ == "__main__":
    parser = ArgumentParser("stable baselines for mario kart")
    parser.add_argument("--wandb", action="store_true", default=False, help="toggles weather to log to wandb or not")
    parser.add_argument("--containerized", action="store_true", default=False, help="toggles weather to abort episode if stuck")
    parser.add_argument("--auto-abort", action="store_true", default=True, help="toggles weather to abort episode if stuck")
    parser.add_argument("--num-tracks", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10_000_000, help="number of steps to train")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-05)
    parser.add_argument("--video-record-frequency", type=int, default=10)
    parser.add_argument("--video-record-path", type=str, default="./recordings/", help="path to where recorded videos shall be stored")
    parser.add_argument("--model-store-path", type=str, default="./models/", help="path to where trained models shall be stored")
    parser.add_argument("--steps-env-switch", type=str, default=200, help="Steps how long the model should be trained on a certain environment until it switches")
    args = parser.parse_args()
    
    main(args)
