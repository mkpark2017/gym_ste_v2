import argparse
from copy import deepcopy
import torch
import gym
import numpy as np
import os
from observation_processor import queue

from evaluator import Evaluator
from ddpg_multi_action import DDPG
from utils import *

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

#    for i in range(num_episodes):
    validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
    if debug:
        prRed('[Evaluate] mean_reward:{}'.format(validate_reward))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DDPG')
    # Set Environment
    parser.add_argument('--mode', default='test', type=str, help='support option: train/test')
    parser.add_argument('--env', default='gym_ste:SteEstHardEnv-v0', type=str, help='open-ai gym environment')
    # Set network parameter
    parser.add_argument('--hidden1', default=1000, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=400, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--hidden3', default=300, type=int, help='hidden num of third fully connect layer')

    parser.add_argument('--rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.00001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--discount', default=0.95, type=float, help='Discount factor for next Q values')
    parser.add_argument('--init_w', default=0.003, type=float, help='Initial network weight')
    parser.add_argument('--tau', default=0.0001, type=float, help='moving average for target network')
    # Set learning parameter
    parser.add_argument('--warmup', default=1000000, type=int, help='time without training but only filling the replay memory')
    #warmup 5e5
    parser.add_argument('--rmsize', default=1000000, type=int, help='Memory size, after exceeding this limits, older entries will be replaced by newer ones')
    #repeat memory 1e7
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--window_length', default=1, type=int, help='Number of observations to be concatenated as "state", (e.g., Atrai game one used this)')
    parser.add_argument('--train_iter', default=10000000, type=int, help='Total number of steps for training')
    #tain iter 1e6
    parser.add_argument('--max_episode_length', default=300, type=int, help='Number of steps for each episode (num_episode = train_iter/max_episode_length')
    parser.add_argument('--validate_episodes', default=20, type=int, help='Number of episodes to perform validation')
    parser.add_argument('--validate_steps', default=100000, type=int, help='Validation step interval (only validate each validation step)')
    # validate 2e4
    parser.add_argument('--epsilon', default=8000000, type=int, help='linear decay of exploration policy')
    # epsilon 5e4
    # Random process for action (Gaussian-Markov process)
    parser.add_argument('--ou_theta', default=0.015, type=float, help='Noise theta')
    parser.add_argument('--ou_sigma', default=0.02, type=float, help='Noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='Noise mu')
    # User convenience parameter
    parser.add_argument('--output', default='output_ddpg_3layers_0811_est', type=str, help='Output root')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Debug')
    parser.add_argument('--seed', default=-1, type=int, help='Random seed')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--pause_time', default=0, type=float, help='Pause time for evaluation')
    args = parser.parse_args()
    
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output_ddpg_3layers_0811_est/{}-run9'.format(args.env)

    env = gym.make(args.env)

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, args.validate_steps, args.output, args.max_episode_length, debug=True)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume, visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
