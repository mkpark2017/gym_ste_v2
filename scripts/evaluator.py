import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from observation_processor import queue
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import *
import time


writer = SummaryWriter()

class Evaluator(object):

    def __init__(self, args):
        self.num_episodes = args.validate_episodes
        self.interval = args.validate_steps
        self.max_episode_length = args.max_episode_length
        self.window_length = args.window_length
        self.save_path = args.output
        self.results = np.array([]).reshape(self.num_episodes,0)
        self.result = []
        self.pause_t = args.pause_time

    def __call__(self, env, policy, debug=False, visualize=False, save=True):

        self.is_training = False
        episode_memory = queue()
        observation = None
        self.result = []

        for episode in range(self.num_episodes):

            # reset at the start of episode
            env.close()
            observation = env.reset()
            env.render_background(mode='human')
#            print("---------------------------------------------")
#            print("goal (x: " + str(env.goal_x) + ", y: " + str(env.goal_y) + ")")
            episode_memory.append(observation)
            observation = episode_memory.getObservation(self.window_length, observation)
            episode_steps = 0
            episode_reward = 0.
                
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation)
#                print("current action:" + str(action))
                observation, reward, done, info = env.step(action)
                episode_memory.append(observation)
                observation = episode_memory.getObservation(self.window_length, observation)
                # Change the episode when episode_steps reach max_episode_length
                if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                    env.render(mode='human')
                    done = True

                if visualize:
                    env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1
                time.sleep(self.pause_t)

                
#            print("goal (x: " + str(env.goal_x) + ", y: " + str(env.goal_y) + ")")
#            print("---------------------------------------------")
            if debug:
                prRed('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
#                env.close()
            self.result.append(episode_reward)

        self.result = np.array(self.result).reshape(-1,1)
        self.results = np.hstack([self.results, self.result])

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(self.result)

    def save_results(self, fn):

        y = np.mean(self.results, axis=0)
        y_single = np.mean(self.result, axis=0)
        error=np.std(self.results, axis=0)
        x = range(0,self.results.shape[1]*self.interval,self.interval)
#        print(self.results.shape[1]*self.interval)
#        print(y_single)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.results})
        writer.add_scalar("Mean Reward/train", y_single, self.results.shape[1]*self.interval)
        writer.flush()

