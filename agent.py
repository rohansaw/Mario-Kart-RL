from pathlib import Path
from copyreg import pickle
from curses import termname
import random
from urllib import request
from tqdm import tqdm
from ast import arg
from select import epoll
import time

import torch
from actor import LSTMActor, SmullActor, BigActor
from critic import SmullCritic, BigCritic
import gym, gym_mupen64plus
from threading import Thread
from multiprocessing import Process
import argparse
import logging
from src.utils import set_logging, TimeMeasurement
from gym_mupen64plus.envs.MarioKart64.discrete_envs import DiscreteActions
from torchvision import transforms

import wandb


class Memory():
    def __init__(self, num_values, buffer_size=16, device="cpu"):
        self.buffer = [torch.zeros((buffer_size, 1)) for _ in range(num_values)]
        self.device = device
        self.buffer_size = buffer_size
        self.buffer_idx = 0
    
    def add(self, *data):
        if self.buffer_idx == 0:
            self.buffer = [torch.zeros((self.buffer_size, 1), device=self.device) for _ in range(len(self.buffer))]
        if self.buffer_idx >= len(self.buffer[0]):
            raise Exception("added too many images")
        for idx, value in enumerate(data):
            self.buffer[idx][self.buffer_idx] = value
        self.buffer_idx += 1
    
    def flush(self):
        if self.buffer_idx < len(self.buffer[0]):
            filled_idx = self.buffer_idx
            self.reset()
            return [i[:filled_idx] for i in self.buffer]
        self.buffer = [i.to(self.device) for i in self.buffer]
        self.reset()
        return list(self.buffer)

    def reset(self):
        self.buffer_idx = 0
        
class Context():
    def __init__(self, input_shape, context_size=16, device="cpu"):
        self.buffer = torch.zeros((context_size, *input_shape), device=device)
        self.device = device
        self.buffer_size = context_size
        self.buffer_idx = 0
    
    def add(self, data):
        self.buffer[self.buffer_idx] = data
        self.buffer_idx += 1
        self.buffer_idx %= self.buffer_size
    
    def get_context(self):
        output = torch.cat((self.buffer[self.buffer_idx:], self.buffer[:self.buffer_idx]))
        output = output.to(self.device)
        return output

    def reset(self):
        self.buffer = torch.zeros(self.buffer.shape, device=self.device)
        self.buffer_idx = 0
    
    def pop(self):
        self.buffer_idx = (self.buffer_idx + self.buffer_size - 1) % self.buffer_size
        
torch.autograd.set_detect_anomaly(True)

class MarioKartAgent():
    def __init__(self, graphic_output=True, num_episodes=10, max_steps=1150, use_wandb=True, visualize_last=True, visualize_every=10, load_model=False):
        self.env = gym.make('Mario-Kart-Discrete-Luigi-Raceway-v0')
        # input_size = (30, 40, 3)
        input_size = (60, 80, 3)
        self.actor = SmullActor(input_size=input_size,
                                 output_size=self.env.action_space.n)
        self.critic = SmullCritic(input_size=input_size)
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.alpha = 0.001 # actor lr
        self.beta = 0.001 # critic lr
        self.gamma = 0.95 # discount factor
        self.step_size = 16
        self.context_size = 16
        self.warmup_episodes = 0
        
        self.output_path = Path("models/")
        
        self.visualize_last_run = visualize_last
        self.visualize_every = visualize_every
        
        # self.max_steps -= 100

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.beta)
        if load_model:
            self.load_model()
        
        self.critic_loss = torch.nn.MSELoss()
        # self.critic_loss = MSE()

        self.graphic_output = graphic_output
        
        self.grafic_transform = transforms.Compose(
            [
                transforms.Resize(input_size[:2]),
            ]
        )
        self.device = "cpu"
        wandb.init(config={"actor_lr": self.alpha, "critic_lr": self.beta, "discount_factor": self.gamma, "episodes": self.num_episodes}, mode="online" if use_wandb else "disabled")
        
    
    def step(self, action):
        obs, rew, end, info = self.env.step(action)
        self.conditional_render()
        return obs, rew, end, info
    
    def _transform_state(self, state):
        state = torch.from_numpy(state.copy()).to(self.device).to(torch.float32)
        state = torch.movedim(state, 2, 0)
        state = (state - 128.0) / 128.0 # state now is in range [-1, 1]
        return self.grafic_transform(state)
    
    def select_action(self, context, step):
        '''Returns one-hot encoded action to play next and log_prob of this action in the distribution'''
        input_values = context.get_context()
        probs = self.actor(input_values)
        
        # Use a categorical policy to sample the action that should be played next
        prob_dist = torch.distributions.Categorical(probs)
        action_index = prob_dist.sample() # Returns index of action to plays
        action_prob = prob_dist.log_prob(action_index)
        return action_index[0], action_prob

    def conditional_render(self):
        if self.graphic_output:
            self.env.render()

    def _compute_advantage(self, observed_reward, state, next_state):
        # Calculate one-step td_error
        next_state = self._transform_state(next_state)
        state = self._transform_state(state)
        target = observed_reward + self.gamma * self.critic(next_state)
        error = target - self.critic(state)
        return  error

    # inspired by https://github.com/hermesdt/reinforcement-learning/blob/master/a2c/cartpole_a2c_episodic.ipynb
    def discount_values(self, rewards, q_value, terminates):
        size = len(rewards)
        q_vals = torch.zeros((size, 1), device=self.device)
        for i in range(size):
            reward = rewards[size - 1 - i]
            terminated = terminates[size - 1 - i]
            q_value = reward + self.gamma * q_value * (1.0 - terminated)
            q_vals[size - 1 - i] = q_value # store values from the end to the beginning
        return q_vals

    def train(self, action_probs, critic_values, rewards, terminated, last_q_value, step, episode):
        action_probs = action_probs.to(self.device)
        critic_values = critic_values.to(self.device)
        terminated = terminated.to(self.device)
        rewards = rewards.to(self.device)
        
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        q_values = self.discount_values(rewards, last_q_value, terminated)
        advantage = q_values - critic_values
        
        critic_loss = advantage.pow(2).mean()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = (-action_probs * advantage.detach()).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        wandb.log({"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(), "advantage": advantage.mean(), "episode": episode}, step=(episode * self.max_steps) + step)

    def reset(self):
        obs = self.env.reset()
        self.conditional_render()
        return obs
    
    def store_model(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        # with (self.output_path / "actor.pckl").open() as actor_store_file:
            # pickle.dump()
        torch.save(self.actor.state_dict(), self.output_path / "actor.pckl")
        torch.save(self.critic.state_dict(), self.output_path / "critic.pckl")
    
    def load_model(self):
        self.actor.load_state_dict(torch.load(self.output_path / "actor.pckl"))
        self.critic.load_state_dict(torch.load(self.output_path / "critic.pckl"))


    def run(self, device="cpu"):
        all_rewards = [] # Rewards of all episodes
        self.device = device
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        buffer = Memory(4, self.step_size)
        previous_graphic_output = self.graphic_output
        
        best_reward = -1e6
        # wandb.watch(self.actor, log_freq=100)
        # wandb.watch(self.critic, log="all", log_freq=10)
        for episode_num in range(1, self.num_episodes):
            if episode_num == self.num_episodes - 1 and self.visualize_last_run:
            # if (episode_num % self.visualize_every == 0) or (episode_num == self.num_episodes - 1 and self.visualize_last_run):
                self.graphic_output = True
            # else:
            #     self.grafic_output = False
            state = self.reset()
            buffer.reset()
            episode_reward = 0
            context = Context(self._transform_state(state).shape, self.context_size, self.device)
            logging.info(f"------ episode {episode_num} ------")
            logging.info("phase 1") # NOOP until green light
            for _ in range(100):
                (obs, rew, end, info) = self.step(0)
                context.add(self._transform_state(obs))
                self.conditional_render()

            logging.info("phase 2") # Train actor and critic networks
            self.actor.reset_model()
            terminated_during_run = False
            for t in tqdm(range(1,self.max_steps)):
                context.add(self._transform_state(state))
                
                action, action_prob = self.select_action(context, t + episode_num * self.max_steps)
                wandb.log({"action": action.detach()}, step= t + episode_num * self.max_steps)
                if episode_num < self.warmup_episodes:
                    action = random.randint(0, self.env.action_space.n - 1)
                next_state, observed_reward, terminated, _ = self.step(action if isinstance(action, int) else action.detach())
                buffer.add(action_prob, self.critic(context.get_context()), observed_reward, terminated)
                
                if terminated or (t != 0 and t % self.step_size == 0):
                    action_probs, critic_values, rewards, done = buffer.flush()
                    context.add(self._transform_state(next_state))
                    last_q_value = self.critic(context.get_context()).detach()
                    self.train(action_probs, critic_values, rewards, done, last_q_value, t, episode_num)
                    context.pop()

                episode_reward += observed_reward
                state = next_state
                # time.sleep(1)
                if terminated:
                    terminated_during_run = True
                    logging.info(f'Episode {episode_num} finished with reward: {episode_reward}')
                    wandb.log({"reward": episode_reward}, step=t + episode_num * self.max_steps)
                    all_rewards.append(episode_reward)
                    break
            if not terminated_during_run:
                logging.info(f'Episode {episode_num} finished with reward: {episode_reward}')
                wandb.log({"reward": episode_reward}, step=t + episode_num * self.max_steps)
                all_rewards.append(episode_reward)
            
            if all_rewards[-1] > best_reward:
                print("storing best model which achieved reward:", all_rewards[-1])
                self.store_model()
                best_reward = all_rewards[-1]
            
        # input("press <enter> to exit....")
        # print("visualizing result:")

        # for _ in range(self.max_steps):
        #     ...
        self.running = False
        self.env.close()

def main(args):
    set_logging(args.log_file, args.log_level, not args.stop_log_stdout)
    agent = MarioKartAgent(args.graphic_output, use_wandb=args.wandb)
    agent.run(args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("RL Agent for Mario Kart for N64 Emulator")
    parser.add_argument("--log-level", type=str, default="info",
                     choices=["debug", "info", "warning", "error", "critical"],
                     help="log level for logging message output")
    parser.add_argument("--log-file", type=str, default="log.log",
                     help="output file path for logging. default to stdout")
    parser.add_argument("--device", type=str, default="cpu",
                     help="device to train on. choose from 'cpu' and 'cuda:0'")
    parser.add_argument("--stop-log-stdout", action="store_false", default=True,
                     help="toggles force logging to stdout. if a log file is specified, logging will be "
                     "printed to both the log file and stdout")
    parser.add_argument("--wandb", action="store_true", default=False,
                     help="toggles logging to wandb")
    parser.add_argument("--graphic-output", action="store_true", default=False,
                        help="toggles weather the graphical output of Mario Kart should be rendered")

    args = parser.parse_args()
    main(args)
