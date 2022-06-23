from ast import arg
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

class MarioKartAgent():
    def __init__(self, graphic_output=True, num_episodes=400, max_steps=10000):
        self.env = gym.make('Mario-Kart-Discrete-Luigi-Raceway-v0')
        hidden_size = (self.env.action_space.n + self.env.observation_space.shape) / 2
        self.actor = SimpleActor(input_size=self.env.observation_space.shape,
                                 hidden_size=hidden_size,
                                 output_size=self.env.action_space.n)
        self.critic = SimpleCritic(input_size=self.env.observation_space.shape,
                             hidden_size=hidden_size,
                             output_size=1)
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.alpha = 0.01 # actor lr
        self.beta = 0.01 # critic lr
        self.gamma = 0.99 # discount factor

        self.actor_optimizer = torch.optim.Adam(self.actor.paramters(), lr=self.alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.beta)
        
        self.graphic_output = graphic_output
    
    def step(self, action):
        obs, rew, end, info = self.env.step(action)
        self.conditional_render()
        return obs, rew, end, info

    def select_action(self, state):
        '''Returns one-hot encoded action to play next and log_prob of this action in the distribution'''
        probs = self.actor(state)
        # Use a categorical policy to sample the action that should be played next
        prob_dist = torch.distributions.Categorical(probs)
        action_index = prob_dist.sample() # Returns index of action to plays
        action_prob = prob_dist.log_prob(action_index)
        return torch.nn.functional.one_hot(action_index, self.env.action_space.n), action_prob

    def conditional_render(self):
        if self.graphic_output:
            self.env.render()

    def _compute_advantage(self, observed_reward, state, next_state):
        # Calculate one-step td_error
        target = observed_reward + self.gamma * self.critic(next_state)
        error = target - self.critic(state)
        return  error

    def train(self, state, next_state, action_prob, observed_reward):
        advantage = self._compute_advantage(observed_reward=observed_reward, state=state, next_state=next_state)
        # Critic loss is basically MSE, since advantage is the error we square it
        critic_loss = advantage.pow(2)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -action_prob*advantage.detach()
        actor_loss.backward()
        self.actor_optimizer.step()

    def reset(self):
        self.state = self.env.reset()
        self.conditional_render()

    def run(self):
        rewards = [] # Rewards of all episodes
        for episode_num in range(1, self.num_episodes):
            self.reset()
            episode_reward = 0

            logging.info("phase 1") # NOOP until green light
            for _ in range(100):
                (obs, rew, end, info) = self.step([0, 0, 0, 0, 0]) 
                self.conditional_render()

            logging.info("phase 2") # Train actor and critic networks
            for t in range(1,self.max_steps):
                action, action_prob = self.select_action(state)
                # ToDo maybe? Do we need some kind of mapping to correct controller actions form onehot encoded action
                next_state, observed_reward, terminated, info = self.step(action)
                self.train(state=state, next_state=next_state,
                           action_prob=action_prob, observed_reward=observed_reward)

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
    agent = MarioKartAgent(args.graphic_output)
    agent.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("RL Agent for Mario Kart for N64 Emulator")
    parser.add_argument("--log-level", type=str, default="info",
                     choices=["debug", "info", "warning", "error", "critical"],
                     help="log level for logging message output")
    parser.add_argument("--log-file", type=str, default="log.log",
                     help="output file path for logging. default to stdout")
    parser.add_argument("--stop-log-stdout", action="store_false", default=True,
                     help="toggles force logging to stdout. if a log file is specified, logging will be "
                     "printed to both the log file and stdout")
    parser.add_argument("--graphic-output", action="store_true", default=False,
                        help="toggles weather the graphical output of Mario Kart should be rendered")

    args = parser.parse_args()
    main(args)