import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from observation_processor import queue

from utils import *

class Evaluator(object):

    def __init__(self, args):
        self.num_episodes = args.validate_episodes
        self.interval = args.validate_steps
        self.max_episode_length = args.max_episode_length
        self.window_length = args.window_length
        self.save_path = args.output
        self.results = np.array([]).reshape(self.num_episodes,0)

    def __call__(self, env, policy, debug=False, visualize=False, save=True):

        self.is_training = False
        episode_memory = queue()
        observation = None
        result = []

        for episode in range(self.num_episodes):

            # reset at the start of episode
            env.close()
            observation = env.reset()
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
                observation, reward, done, info = env.step(action)
                episode_memory.append(observation)
                observation = episode_memory.getObservation(self.window_length, observation)
                # Change the episode when episode_steps reach max_episode_length
                if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                    done = True
                
                if visualize:
                    env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1

            if debug:
                prRed('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
#                env.close()
            result.append(episode_reward)

        result = np.array(result).reshape(-1,1)
        self.results = np.hstack([self.results, result])

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result)

    def save_results(self, fn):

        y = np.mean(self.results, axis=0)
        error=np.std(self.results, axis=0)
                    
        x = range(0,self.results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.results})
