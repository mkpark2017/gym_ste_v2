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
    def __init__(self, nb_states, hidden, nb_actions, n_layers, drop_prob=0.2):
        super(Actor, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        
        self.fc0 = nn.Linear(nb_states, hidden)
        self.gru = nn.GRU(hidden, hidden, n_layers, batch_first=True, dropout=drop_prob)
        self.fc1 = nn.Linear(hidden, nb_actions)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
#        x = torch.cat([x, l_a], -1)
        x = self.fc0(self.relu(x)) # x = activation(self.fc0(x))   # lstm_branch: sequential data
#        print(x.size())
#        print(h)
        out, h = self.gru(x, h)
        out = self.fc1(self.relu(out[:,-1]))
        return out, h
        
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
        self.gru = nn.GRU(hidden, hidden, n_layers, batch_first=True, dropout=drop_prob)
        self.fc1 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, a, h):
        """ 
        state shape: (batch_size, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for GRU needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        print(x.size())
#        x = x.permute(1,0,2)
        print(a.size())
#        a = a.permute(1,0,2)
        x = torch.cat([x, a], -1)
        x = self.fc0(self.relu(x))
        x, h = self.gru(x, h)
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
