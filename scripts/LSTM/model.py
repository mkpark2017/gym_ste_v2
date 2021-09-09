import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden=300, n_layers=2, init_w=3e-3, drop_prob=0.2):
        super(Actor, self).__init__()
#        print(nb_states)
        
        self.fc1 = nn.Linear(nb_states, hidden)
        self.lstm = nn.LSTM(hidden, hidden, n_layers, batch_first=True, dropout=drop_prob)
        self.fc2 = nn.Linear(hidden, nb_actions)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
#        print(self.fc1.weight.data.size())
        self.fc2.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x, h):
#        print(x.shape)
        out = self.fc1(x)
        out = self.relu(out)

        out, h = self.gru(out, h)

        out = self.fc2(out)
        out = self.tanh(out)
        return out, h

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden=300, n_layers=2, init_w=3e-3, drop_prob=0.2):
        super(Critic, self).__init__()



        self.fc1 = nn.Linear(nb_states, hidden)
        # can I set different size of hidden for fc1 and lstm?
        self.lstm = nn.LSTM(hidden_nb_actions, hidden, n_layers, batch_first=True, dropout=drop_prob)
        # this is not sure, do I have to set same size of input and output?
        self.fc2 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x, a, h):
        out = self.fc1(x)
        out = self.relu(out)
        # debug()

        out = self.fc2(torch.cat([out,a],1))
        out, h = self.lstm(out, h)

        out = self.fc2(out)
        return out, h
