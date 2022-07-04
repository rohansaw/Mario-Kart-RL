from ast import arg
from select import epoll
import time

import torch
from actor import LSTMActor, SimpleActor
from critic import SimpleCritic
import gym, gym_mupen64plus
from threading import Thread
from multiprocessing import Process
import argparse
import logging
from src.utils import set_logging
from gym_mupen64plus.envs.MarioKart64.discrete_envs import DiscreteActions
from torchvision import transforms
import wandb

class MarioKartAgent():
    def __init__(self, graphic_output=True, num_episodes=400, max_steps=10000, use_wandb=True):
        self.env = gym.make('Mario-Kart-Discrete-Luigi-Raceway-v0')
        self.actor = SimpleActor(input_size=self.env.observation_space.shape,
                                 output_size=self.env.action_space.n)
        self.critic = SimpleCritic(input_size=self.env.observation_space.shape)
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.alpha = 0.01 # actor lr
        self.beta = 0.01 # critic lr
        self.gamma = 0.99 # discount factor

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.beta)
        
        self.graphic_output = graphic_output
        
        self.grafic_transform = transforms.Compose(
            [
                transforms.Resize((120, 160)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                # transforms.Grayscale(),
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
        state = torch.movedim(state, 2, 0).unsqueeze(0)
        return self.grafic_transform(state)
    
    def select_action(self, state):
        '''Returns one-hot encoded action to play next and log_prob of this action in the distribution'''
        # print("forwarding: ", state.shape, state)
        state = self._transform_state(state)
        probs = self.actor(state)
        
        # Use a categorical policy to sample the action that should be played next
        prob_dist = torch.distributions.Categorical(probs)
        action_index = prob_dist.sample() # Returns index of action to plays
        action_prob = prob_dist.log_prob(action_index)
        # print(action_index, type(action_index))
        # action = DiscreteActions.get_action(action_index[0])
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

    def train(self, state, next_state, action_prob, observed_reward, step, episode):
        
        advantage = self._compute_advantage(observed_reward=observed_reward, state=state, next_state=next_state)
        # Critic loss is basically MSE, since advantage is the error we square it
        critic_loss = advantage.pow(2)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -action_prob*advantage.detach()
        actor_loss.backward()
        self.actor_optimizer.step()
        print(critic_loss.item(), actor_loss.item())
        wandb.log({"critic": critic_loss.item(), "actor": actor_loss.item(), "step": step, "episode": episode}, step=step)

    def reset(self):
        obs = self.env.reset()
        self.conditional_render()
        return obs

    def run(self, device="cpu"):
        rewards = [] # Rewards of all episodes
        self.device = device
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        for episode_num in range(1, self.num_episodes):
            state = self.reset()
            episode_reward = 0

            logging.info("phase 1") # NOOP until green light
            for _ in range(100):
                (obs, rew, end, info) = self.step(0) 
                self.conditional_render()

            logging.info("phase 2") # Train actor and critic networks
            for t in range(1,self.max_steps):
                action, action_prob = self.select_action(state)
                # ToDo maybe? Do we need some kind of mapping to correct controller actions form onehot encoded action
                start = time.time()
                next_state, observed_reward, terminated, info = self.step(action)
                print("step time:", time.time() - start)
                self.train(state=state, next_state=next_state,
                           action_prob=action_prob, observed_reward=observed_reward, step=t, episode=episode_num)

                episode_reward += observed_reward
                state = next_state
                if terminated:
                    logging.info(f'Episode {episode_num} finished with reward: {episode_reward}')
                    rewards.append(episode_reward)
                    break

        input("press <enter> to exit....")
        self.running = False
        self.env.close()

def main(args):
    set_logging(args.log_file, args.log_level, not args.stop_log_stdout)
    agent = MarioKartAgent(args.graphic_output, use_wandb=not args.no_wandb)
    agent.run()

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
    parser.add_argument("--graphic-output", action="store_true", default=False,
                        help="toggles weather the graphical output of Mario Kart should be rendered")
    parser.add_argument("--no-wandb", action="store_true", help="Do not publish results to wandb") #we can also change this to activate it with the command

    args = parser.parse_args()
    main(args)