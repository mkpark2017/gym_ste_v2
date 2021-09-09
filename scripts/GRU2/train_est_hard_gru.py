import argparse
from copy import deepcopy
import torch
import gym
import numpy as np
import os
from observation_processor import queue

from evaluator import Evaluator
from ddpg_multi_action_gru import DDPG_GRU
from utils import *
from memory_gru import ReplayBufferLSTM2

def train(replay_buffer, num_iterations, agent, env, evaluate, validate_steps, output, max_episode_length=None, debug=False):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward_sum = 0.
    episode_memory = queue()
    obs = None # Observation
    highest_reward = -10.
    validate_reward = -10.

    hidden_out = agent.actor.init_hidden(1)
    hidden_out = hidden_out.data

    while step < num_iterations:
        #reset if it is the start of episode
        if obs is None:
            episode_memory.clear()
            obs = deepcopy(env.reset())
            episode_memory.append(obs)
            obs = episode_memory.getObservation(args.window_length, obs)
            agent.reset(obs)
            last_action = np.array([0, 0])
            episode_state = []
            episode_action = []
            episode_last_action = []
            episode_reward = []
            episode_next_state = []
            episode_done = []
            episode_hidden_in = []
            episode_hidden_out = []
        
        hidden_in = hidden_out
        # Agent pick action
#        if step <= args.warmup:
#            action = agent.random_action()
#        else:
        action, hidden_out = agent.select_action(obs, hidden_in)
#            print(action)
        # Env response with next_obs, reward, done, terminate_info
        obs2, reward, done, info = env.step(action)
        episode_memory.append(obs2)
        obs2 = episode_memory.getObservation(args.window_length, obs2)

        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True
        # Agent observe and update policy
        # agent.observe(hidden_in, hidden_out, reward, obs2, done)
        episode_hidden_in.append(hidden_in.cpu().detach().numpy())
        episode_hidden_out.append(hidden_out.cpu().detach().numpy())
#        episode_hidden_in.append(torch.cat(hidden_in, dim=-2).detach())
#        episode_hidden_out.append(torch.cat(hidden_out, dim=-2).detach())

        episode_state.append(obs)
        episode_action.append(action)
        episode_last_action.append(last_action)
        episode_reward.append(reward)
        episode_next_state.append(obs2) #next state
        episode_done.append(done)

        print("buffer", len(replay_buffer))
        if len(replay_buffer) > args.bsize:
            agent.update_policy()
            # Evaluation
            if evaluate is not None and validate_steps > 0 and step % validate_steps == 1:
                policy = lambda x: agent.select_action(x, decay_epsilon=False)
                validate_reward = evaluate(env, policy, debug=False, visualize=True)
                if debug:
                    prRed('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
                # Save intermediate model
                if highest_reward < validate_reward:
                    prRed('Highest reward: {}, Validate_reward: {}'.format(highest_reward, validate_reward))
                    highest_reward = validate_reward
                    agent.save_model(output)

        if episode_steps == 0:
            ini_hidden_in = hidden_in
            ini_hidden_out = hidden_out
#            ini_hidden_in = hidden_in.cpu().detach().numpy()
#            ini_hidden_out = hidden_out.cpu().detach().numpy()

        step += 1
        episode_steps += 1

        episode_reward_sum += reward

        last_action = action

#        replay_buffer.push(
#            hidden_in.cpu().detach().numpy(),
#            hidden_out.cpu().detach().numpy(),
#            obs,
#            action,
#            last_action,
#            reward,
#            obs2,
#            done
#            )

        if done: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward_sum,step))
            new_action, _ = agent.select_action(obs, hidden_in)
            replay_buffer.push(
                ini_hidden_in,
                ini_hidden_out,
                episode_state,
                episode_action,
                episode_last_action,
                episode_reward,
                episode_next_state,
                episode_done)

            obs = None
            episode_steps = 0
            episode_reward_sum = 0.
            episode += 1

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
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='gym_ste:SteEstHardEnv-v0', type=str, help='open-ai gym environment')
    # Set network parameter
    parser.add_argument('--hidden', default=100, type=int, help='hidden num of layer')
#    parser.add_argument('--hidden2', default=400, type=int, help='hidden num of second fully connect layer')
#    parser.add_argument('--hidden3', default=300, type=int, help='hidden num of third fully connect layer')

    parser.add_argument('--rate', default=0.00005, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.000005, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--discount', default=0.95, type=float, help='Discount factor for next Q values')
    parser.add_argument('--init_w', default=0.003, type=float, help='Initial network weight')
    parser.add_argument('--tau', default=0.00005, type=float, help='moving average for target network')

    parser.add_argument('--n_layers', default=2, type=float, help='number of hidden layer')

    # Set learning parameter
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    #warmup 5e5
    parser.add_argument('--rbsize', default=1000000, type=int, help='Replay buffer size, after exceeding this limits, older entries will be replaced by newer ones')
    #repeat memory 1e7
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--window_length', default=1, type=int, help='Number of observations to be concatenated as "state", (e.g., Atrai game one used this)')
    parser.add_argument('--train_iter', default=100000000, type=int, help='Total number of steps for training')
    #tain iter 1e6
    parser.add_argument('--max_episode_length', default=300, type=int, help='Number of steps for each episode (num_episode = train_iter/max_episode_length')
    parser.add_argument('--validate_episodes', default=20, type=int, help='Number of episodes to perform validation')
    parser.add_argument('--validate_steps', default=1000000, type=int, help='Validation step interval (only validate each validation step)')
    # validate 2e4
    parser.add_argument('--epsilon', default=80000000, type=int, help='linear decay of exploration policy')
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
        args.resume = 'output_ddpg_gru_0813/{}-run1'.format(args.env)

    env = gym.make(args.env)

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    replay_buffer = ReplayBufferLSTM2(args.rbsize)
    agent = DDPG_GRU(replay_buffer, nb_states, nb_actions, args)
    evaluate = Evaluator(args)

    if args.mode == 'train':
        train(replay_buffer, args.train_iter, agent, env, evaluate, args.validate_steps, args.output, args.max_episode_length, debug=True)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume, visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
