import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model_gru import (Actor, Critic) # Define activation function
from memory_gru import ReplayBufferLSTM2
from random_process import OrnsteinUhlenbeckProcess
from utils import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

class DDPG_GRU(object):
    def __init__(self, replay_buffer, nb_states, nb_actions, args):
        
        self.replay_buffer = replay_buffer
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states * args.window_length
        self.nb_actions= nb_actions
        self.hidden = args.hidden
        self.n_layers = args.n_layers
        self.bsize = args.bsize
        
        self.actor = Actor(self.nb_states, self.hidden, self.nb_actions, self.n_layers) # call network
#        self.actor.init_hidden(args.bsize)
#        print(self.nb_states)
        self.actor_target = Actor(self.nb_states, self.hidden, self.nb_actions, self.n_layers)
#        self.actor_target.init_hidden(args.bsize)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.hidden, self.nb_actions, self.n_layers)
#        self.critic.init_hidden(args.bsize)
        self.critic_target = Critic(self.nb_states, self.hidden, self.nb_actions, self.n_layers)
#        self.critic_target.init_hidden(args.bsize)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
#        self.memory = SequentialMemory(limit=args.rbsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        # 
        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        hidden_in, hidden_out, state_batch, action_batch, last_action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.replay_buffer.sample(self.bsize)
#        print('sample:', state_batch, action_batch,  reward_batch, terminal_batch)
#        print(len(hidden_in))
#        print(len(hidden_in[0]), "\n====================")
#        print(len(hidden_in[1]))

#        hidden_in      = torch.FloatTensor(hidden_in).to(device)
#        hidden_out      = torch.FloatTensor(hidden_out).to(device)
        state      = torch.FloatTensor(state_batch).to(device)
        next_state = torch.FloatTensor(next_state_batch).to(device)
#        print(action_batch,"\n============================================")
        action     = torch.FloatTensor(action_batch).to(device)
#        print(last_action_batch)
        last_action     = torch.FloatTensor(last_action_batch).to(device)
        reward     = torch.FloatTensor(reward_batch).unsqueeze(-1).to(device)  
        done       = torch.FloatTensor(np.float32(terminal_batch)).unsqueeze(-1).to(device)
#        print(done)
        # use hidden states stored in the memory for initialization, hidden_in for current, hidden_out for target
#        print(hidden_in[0], "\n====================")
#        print(hidden_in[1])
#        print(len(hidden_in[0][0]))
#        print("---------------\n", hidden_in.size(), "\n------------------")
        predict_q, _ = self.critic(state, action, hidden_in) # for q 
        new_action, _ = self.actor.evaluate(state, last_action, hidden_in) # for policy
        new_next_action, _ = self.actor_target.evaluate(next_state, action, hidden_out)  # for q
        predict_target_q, _ = self.critic_target(next_state, new_next_action, hidden_out)  # for q

        predict_new_q, _ = self.critic(state, new_action, hidden_in) # for policy. as optimizers are separated, no detach for q_h_in is also fine
        target_q = reward+(1-done)*gamma*predict_target_q # for q
        # reward = reward_scale * (reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std

        q_loss = criterion(predict_q, target_q.detach())
        policy_loss = -torch.mean(predict_new_q)

        # train qnet
        self.critic_optim.zero_grad()
        q_loss.backward(retain_graph=True)  # no need for retain_graph here actually
        self.critic_optim.step()

        # train policy_net     
        self.actor_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()

            
        # update the target_qnet
#        if self.update_cnt%target_update_delay==0:
        self.target_qnet=self.target_soft_update(self.qnet, self.target_qnet, soft_tau)
        self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, h_in, h_out, r_t, s_t1, last_action, done):
        if self.is_training:
         #   print("===========================================")
         #   print(h_in.size(), "\n", h_out.size(), "\n", self.s_t, "\n", self.a_t, "\n", r_t, "\n", done)
         #   print(self.memory)
            self.memory.append(h_in, h_out, self.s_t, self.a_t, last_action, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
#        action = env.action_space.sample()
        self.a_t = action
        return action

    def select_action(self, s_t, h_in, decay_epsilon=True):
#        print(to_tensor(np.array([s_t])).shape)
        s_t = torch.FloatTensor(s_t).unsqueeze(0).unsqueeze(0).cuda()
#        print(s_t.size())
#        print(h_in.size())
        action, h_out= self.actor(s_t, h_in)
        action = to_numpy(action).squeeze(0)
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
#        print("original_action: " + str(action))
        while action[0] > 1: # Angle wrapping
            action[0] = action[0] - 2
        while action[0] < -1:
            action[0] = action[0] + 2
        #action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action, h_out

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
