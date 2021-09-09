import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as debug

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Policy network, Let's assume input state = [obs]
class Actor(nn.Module):
    def __init__(self, nb_states, hidden, nb_actions, n_layers, drop_prob=0.2, init_w=3e-3):
        super(Actor, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        
        self.linear1 = nn.Linear(nb_states, hidden)
        self.linear2 = nn.Linear(nb_states+nb_actions, hidden)
        self.lstm1 = nn.LSTM(hidden, hidden, n_layers, batch_first=True, dropout=drop_prob)
        self.linear3 = nn.Linear(2*hidden, hidden)
        self.linear4 = nn.Linear(hidden, nb_actions) # output dim = dim of action

        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
    

    def forward(self, state, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        activation=F.relu
        # branch 1
        fc_branch = activation(self.linear1(state)) 
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = activation(self.linear2(lstm_branch))   # lstm_branch: sequential data
        # hidden only for initialization, later on hidden states are passed automatically for sequential data
        lstm_branch,  lstm_hidden = self.lstm1(lstm_branch, hidden_in)    # no activation after lstm
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1)   
        x = activation(self.linear3(merged_branch))
        x = F.tanh(self.linear4(x))
        x = x.permute(1,0,2)  # permute back

        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden).zero_().to(device)
        return hidden

# Q network, Let's assume input state = [obs, last_action, new_action]
class Critic(nn.Module):
    def __init__(self, nb_states, hidden, nb_actions, n_layers, drop_prob=0.2):
        super(Critic, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers

        self.fc0 = nn.Linear(nb_states+nb_actions, hidden)
        self.lstm = nn.LSTM(hidden, hidden, n_layers, batch_first=True, dropout=drop_prob)
        self.fc1 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, a, h):
        """ 
        state shape: (batch_size, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for GRU needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        print(x.size())
        x = x.permute(1,0,2)
        print(a.size())
        a = a.permute(1,0,2)
        x = torch.cat([x, a], -1)
        x = self.fc0(self.relu(x))
        x, h = self.lstm(x, h)
        out = self.fc1(self.relu(x))
        out = out.permute(1,0,2) # back to same axes as input 
        return out, h
        
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden).zero_().to(device)
        return hidden

    def evaluate(self, state, last_action, hidden_in, noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, last_action, hidden_in)
        noise = noise_scale * normal.sample(action.shape).cuda()
        action = self.action_range*action+noise
        return action, hidden_out
