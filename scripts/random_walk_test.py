import gym
import time
import matplotlib.pyplot as plt
import numpy as np

#import gym_ste.envs

DEBUG = True
env = gym.make('gym_ste:BasicSteEnv-v0')

for e in range(100):
    obs = env.reset()
    cumulated_reward = 0
    i = 0
    done = False
    env.render_background(mode='human')
    while not done and i <= 100:
    #while 1:
        i += 1
        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)     # take a random action
    #    env.render_background(mode='human')

        env.render(mode='human')
        cumulated_reward += rew
        if DEBUG:
            print(info)
    if DEBUG and done:
        time.sleep(3)

    print("episode ended with cumulated rew", cumulated_reward, "and done:", done)
    env.close()
