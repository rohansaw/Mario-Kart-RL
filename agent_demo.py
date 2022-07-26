#!/bin/python
from curses.ascii import DEL
from PIL import Image
import time
import gym, gym_mupen64plus
from gym_mupen64plus.envs.mupen64plus_env import COUNTPEROP, IMAGE_HELPER

# render = False
render = True
env = gym.make('Mario-Kart-Discrete-Luigi-Raceway-v0', auto_abort=False, resolution="supersmall", base_episode_length=30000)
# env.reset()
# if render:
#     env.render()

DELAY = 0.00

steps_for_full = [
    (300, 1),
    (270, 3),
    (200, 1),
    (110, 7),
    (230, 1),
    (221, 3),
    # new lap
    (330, 1),
    (310, 3),
    (200, 1),
    (110, 7),
    (230, 1),
    (250, 3),
    # new lap
    (450, 1),
    (270, 3),
    (200, 1),
    (110, 7),
    (230, 1),
    (230, 3),
    (150, 1),
]

print("NOOP waiting for green light")

print("GO! ...drive straight as fast as possible...")
start = time.time()
rews = []
obs = None
steps = 0
obs = None
total_steps = 100

for j in range(2):
    env.reset()
    if render:
        env.render()
    # input("resetted!")
    start = time.time()
    for i in range(100):
        (obs, rew, end, info) = env.step(0) # NOOP until green light
        if render:
            env.render()
    for segment in steps_for_full:
        steps, action = segment
        for i in range(steps):
            # (obs, rew, end, info) = env.step([0, -80, 0, 1, 0]) # Drive straight
            (obs, rew, end, info) = env.step(action) # Drive straight
            # print("rew:", rew)
            if end:
                print("isch over", i)
                
                break
            if render:
                env.render()
            time.sleep(DELAY)
            rews.append(rew)
        total_steps += i
    print("total steps: ", total_steps)
    print(total_steps / (time.time() - start), "steps per second!")
    last_obs = env.render(mode="rgb_array")
    im = Image.fromarray(last_obs)
    im.save(f"images/final_screen_{j}.png")

    print(sum(rews))
    input("press enter")

print(sum(rews))
input("press enter")
env.close()
exit(1)

