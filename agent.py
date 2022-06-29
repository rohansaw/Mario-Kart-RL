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
from gym_mupen64plus.envs.MarioKart64.discrete_envs import DiscreteActions
from torchvision import transforms

class MarioKartAgent():
    def __init__(self, graphic_output=True):
        self.env = gym.make('Mario-Kart-Discrete-Luigi-Raceway-v0')
        self.actor = SimpleActor(input_size=self.env.observation_space.shape,
                                 output_size=self.env.action_space.n)
        self.critic = SimpleCritic(input_size=self.env.observation_space.shape)
        self.alpha = 0.01 # actor lr
        self.beta = 0.01 # critic lr
        self.gamma = 0.99 # discount factor

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.beta)
        
        self.graphic_output = graphic_output
        
        self.grafic_transform = transforms.Compose(
            [
                transforms.Resize((240, 320)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                # transforms.Grayscale(),
            ]
        )
        
    def step(self, action):
        obs, rew, end, info = self.env.step(action)
        self.conditional_render()
        return obs, rew, end, info
    
    def _transform_state(self, state):
        state = torch.movedim(torch.from_numpy(state.copy()).to(torch.float32), 2, 0).unsqueeze(0)
        return self.grafic_transform(state).to(torch.float32)
    
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

    def train(self, state, next_state, action_prob, observed_reward):
        advantage = self._compute_advantage(observed_reward=observed_reward, state=state, next_state=next_state)
        # Critic loss is basically MSE, since advantage is the error we square it
        critic_loss = advantage.pow(2)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -action_prob*advantage.detach()
        actor_loss.backward()
        self.actor_optimizer.step()
        print(critic_loss.item(), actor_loss.item())

    def reset(self):
        obs = self.env.reset()
        self.conditional_render()
        return obs