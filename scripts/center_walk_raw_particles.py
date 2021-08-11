import gym
import time
import matplotlib.pyplot as plt
import numpy as np

import math

#import gym_ste.envs

DEBUG = True
env = gym.make('gym_ste:StePFilterConvHardEnv2-v0')

for e in range(100):
    obs = env.reset()
    cumulated_reward = 0
    i = 0
    done = False
    env.render_background(mode='human')
    pf_num = 300
    etc_num = 9
    while not done and i <= 300:
    #while 1:
        i += 1
#        print(obs[7:17]*55)
        c_x = np.sum(obs[etc_num+pf_num*2:etc_num+pf_num*3]*(obs[etc_num:etc_num+pf_num] - obs[3])*60)
        c_y = np.sum(obs[etc_num+pf_num*2:etc_num+pf_num*3]*(obs[etc_num+pf_num:etc_num+pf_num*2] - obs[4])*60)
        act = math.atan2(c_y,c_x)/math.pi + np.random.rand(1)/10
#        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)     # take a random action
    #    env.render_background(mode='human')

        env.render(mode='human')
        cumulated_reward += rew
        time.sleep(0.1)
        if DEBUG:
            print(info)
#        env.close()
    if DEBUG and done:
        time.sleep(3)

    print("episode ended with cumulated rew", cumulated_reward, "and done:", done)
    env.close()
