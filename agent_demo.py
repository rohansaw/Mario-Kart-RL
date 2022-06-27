#!/bin/python
import time
import gym, gym_mupen64plus
from gym_mupen64plus.envs.MarioKart64.discrete_envs import DiscreteActions

render = False

env = gym.make('Mario-Kart-Discrete-Luigi-Raceway-v0')
env.reset()
if render:
    env.render()

print("NOOP waiting for green light")
for i in range(100):
    (obs, rew, end, info) = env.step(0) # NOOP until green light
    if render:
        env.render()

print("GO! ...drive straight as fast as possible...")
start = time.time()
for i in range(1000):
    # if i != 0 and i % 10 == 0:
    #     start = end
    (obs, rew, end, info) = env.step(1) # Drive straight
    if render:
        env.render()
end = time.time()
print(1000 / (end - start), "frames per second, total:", (end-start), "seconds")
print((end - start), "seconds for 1000 steps")

# print("Doughnuts!!")
# for i in range(10000):
#     if i % 100 == 0:
#         print("Step " + str(i))
#     (obs, rew, end, info) = env.step([-80, 0, 1, 0, 0]) # Hard-left doughnuts!
#     (obs, rew, end, info) = env.step([-80, 0, 0, 0, 0]) # Hard-left doughnuts!#
#     env.render()

input("Press <enter> to exit... ")

env.close()

