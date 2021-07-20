import argparse
from copy import deepcopy
import torch
import gym
import numpy as np

from DDPG import DDPG
from utils import *

def train(num_iterations, agent, env, evaluate, validate_steps, output, max_episode_length=None, debug=False):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    while step < num_iterations:
        #reset if it is the start of episode
        if observation is None:
        ######################## Bla Bla Bla
def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    ###################### Bla Bla Bla

if __name__ == "__main__"
    parser = argparse.ArgumentParser(description='PyTorch DDPG')
    # Set Environment
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='gym_ste:BasicSteEnv-v0', type=str, help='open-ai gym environment')
    # Set network parameter
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    # Set learning parameter
    parser.add_argument('--rmsize', default=10000, type=int, help='Memory size, after exceeding this limits, older entries will be replaced by newer ones')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--window_length', default=10, type=int, help='Number of observations to be concatenated as "state", (e.g., Atrai game one used this)')
    parser.add_argument('--train_iter', default=300000, type=int, help='Total number of steps for training')
    parser.add_argument('--max_episode_length', default=300, type=int, help='Number of steps for each episode (num_episode = train_iter/max_episode_length')
    parser.add_argument('--validate_episodes', default=20, type=int, help='Number of episodes to perform validation')
    parser.add_argument('--validate_steps', default=2000, type=int, help='Validation step interval (only validate each validation step)')
    # Random process for action (Gaussian process)
    parser.add_argument('--ou_theta', default=0.15, type=float, help='Noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='Noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='Noise mu')
    # User convenience parameter
    parser.add_argument('--output', default='output', type=str, help='output root')
    parser.add_argument('--debug', dest='debug', action='store_true')
    
    args = parser.parse_args()
    
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default'
        args.resume = 'output/{}-run0'.format(args.env)

    env = gym.make('gym_ste:BasicSteEnv-v0')

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]


    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes, args.validate_steps, args.output, args.max_episode_length)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, args.validate_steps, args.output, args.max_episode_length, args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume, visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
