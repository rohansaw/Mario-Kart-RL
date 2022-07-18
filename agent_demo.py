#!/bin/python
from PIL import Image
import time
import gym, gym_mupen64plus
from gym_mupen64plus.envs.mupen64plus_env import COUNTPEROP

render = False
# render = True
env = gym.make('Mario-Kart-Luigi-Raceway-v0', resolution="supersmall", base_episode_length=3000)
env.reset()
if render:
    env.render()

print("NOOP waiting for green light")
for i in range(100):
    (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) # NOOP until green light
    if render:
        env.render()

print("GO! ...drive straight as fast as possible...")
start = time.time()
rews = []
obs = None
steps = 0
for i in range(300):
    # (obs, rew, end, info) = env.step([0, -80, 0, 1, 0]) # Drive straight
    (obs, rew, end, info) = env.step([0, 0, 1, 0, 0]) # Drive straight
    if end:
        print("isch over", i)
        break
    # print("rew:", rew)
    if render:
        env.render()
    # time.sleep(0.02)
    rews.append(rew)
print("steps:", i)
steps += i
print("now left")
for i in range(270):
    # (obs, rew, end, info) = env.step([0, -80, 0, 1, 0]) # Drive straight
    (obs, rew, end, info) = env.step([-40,   0, 1, 0, 0]) # Drive straight
    if end:
        print("isch over", i)
        break
    # print("rew:", rew)
    if render:
        env.render()
    # time.sleep(0.02)
    rews.append(rew)
print("steps:", i)

steps += i
print("now straight")
for i in range(500):
    # (obs, rew, end, info) = env.step([0, -80, 0, 1, 0]) # Drive straight
    (obs, rew, end, info) = env.step([0, 0, 1, 0, 0]) # Drive straight
    if end:
        print("isch over", i)
        break
    # print("rew:", rew, "sum:", sum(rews))
    if render:
        env.render()
    # time.sleep(0.02)
    
    rews.append(rew)

print("steps:", i)
steps += i
# time.sleep(0.2)
print("storing...")
im = Image.fromarray(obs)
im.save("output_emulator.png")
# time.sleep(30)
end = time.time()
print("total:", steps, "steps!")
print(steps / (end - start), "frames per second, total:", (end-start), "seconds")
print(end - start, f" seconds per {steps} steps")
# print(rews)
print(sum(rews))
rc = 10
start = time.time()
for _ in range(rc):
    env.reset()
print(f"resetting env {rc} times took ", time.time() - start, "seconds!")
# print("Doughnuts!!")
# for i in range(10000):
#     if i % 100 == 0:
#         print("Step " + str(i))
#     (obs, rew, end, info) = env.step([-80, 0, 1, 0, 0]) # Hard-left doughnuts!
#     (obs, rew, end, info) = env.step([-80, 0, 0, 0, 0]) # Hard-left doughnuts!#
#     env.render()

# input("Press <enter> to exit... ")

env.close()
exit(1)

