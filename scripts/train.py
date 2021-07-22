import argparse
from copy import deepcopy
import torch
import gym
import numpy as np
import os
from observation_processor import queue

from evaluator import Evaluator
from ddpg import DDPG
from utils import *

#gym.undo_logger_setup()
#gym.undo_logger_setup()
def train(num_iterations, agent, env, evaluate, validate_steps, output, max_episode_length=None, debug=False):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    episode_memory = queue()
    obs = None # Observation
    # Set device
#    device = torch.device("cuda" if torch.cuda.is_avaulable() else "cpu")
 #   txt_logger.info(f"Device: {device}\n")
  #  if device == "cuda":
   #     agent.cuda();
#    print(num_iterations)
    while step < num_iterations:
#    for step in range(num_iterations):
        #reset if it is the start of episode
        if obs is None:
            episode_memory.clear()
            obs = deepcopy(env.reset())
            episode_memory.append(obs)
            obs = episode_memory.getObservation(args.window_length, obs)
            agent.reset(obs)
        # Agent pick action
#        print(step)
        if step <= args.warmup:
#            print(step)
            action = agent.random_action()
#            print(action)
#            print('\n')
        else:
            action = agent.select_action(obs)
        # Env response with next_obs, reward, done, terminate_info
        obs2, reward, done, info = env.step(action)
        episode_memory.append(obs2)
        obs2 = episode_memory.getObservation(args.window_length, obs2)
#        obs2 = deepcopy(obs2)

        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True
        # Agent observe and update policy
        agent.observe(reward, obs2, done)
        if step > args.warmup:
            agent.update_policy()
        # Evaluation
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 1 and step > args.warmup:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=True)
            if debug:
                prRed('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
        # Save intermediate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)
        # Update
        step += 1
        episode_steps += 1
        episode_reward += reward
#        obs = deepcopy(obs2)
 #       env.render(mode='human')


        if done: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))
            agent.memory.append(
                obs,
                agent.select_action(obs),
                0., False
            )

            # reset
  #          env.close()
            obs = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug:
            prRed('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DDPG')
    # Set Environment
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='gym_ste:BasicSteEnv-v0', type=str, help='open-ai gym environment')
    # Set network parameter
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--discount', default=0.99, type=float, help='Discount factor for next Q values')
    parser.add_argument('--init_w', default=0.003, type=float, help='Initial network weight')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    # Set learning parameter
    parser.add_argument('--warmup', default=100000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--rmsize', default=1000000, type=int, help='Memory size, after exceeding this limits, older entries will be replaced by newer ones')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--window_length', default=1, type=int, help='Number of observations to be concatenated as "state", (e.g., Atrai game one used this)')
    parser.add_argument('--train_iter', default=1000000, type=int, help='Total number of steps for training')
    parser.add_argument('--max_episode_length', default=100, type=int, help='Number of steps for each episode (num_episode = train_iter/max_episode_length')
    parser.add_argument('--validate_episodes', default=2, type=int, help='Number of episodes to perform validation')
    parser.add_argument('--validate_steps', default=10000, type=int, help='Validation step interval (only validate each validation step)')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    # Random process for action (Gaussian-Markov process)
    parser.add_argument('--ou_theta', default=0.015, type=float, help='Noise theta')
    parser.add_argument('--ou_sigma', default=0.02, type=float, help='Noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='Noise mu')
    # User convenience parameter
    parser.add_argument('--output', default='output', type=str, help='Output root')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Debug')
    parser.add_argument('--seed', default=-1, type=int, help='Random seed')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    args = parser.parse_args()
    
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run2'.format(args.env)

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
