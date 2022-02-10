import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste_v2.envs.pf_conv_cent.ste_pf_converge_center_base import *
from gym.envs.registration import register

from datetime import datetime
import scipy.io


class StePfConvCentExtMatEnv(BaseEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        BaseEnv.__init__(self)


        mat_file = scipy.io.loadmat('/root/gym_ste_v2/gym_ste_v2/envs/common/mat_files/random_states_rate.mat')
        self.agent_x_list = mat_file['agent_x_rate']*self.court_lx
        self.agent_y_list = mat_file['agent_y_rate']*self.court_ly
        self.goal_x_list = mat_file['goal_x_rate']*self.court_lx #goal_rate is already set between 0.2 and 0.8 by matlab
        self.goal_y_list = mat_file['goal_y_rate']*self.court_ly
        self.gas_d_list = 5 + mat_file['gas_d_rate']*10
        self.gas_t_list = 500 + mat_file['gas_t_rate']*1000
        self.gas_q_list = 1000 + mat_file['gas_q_rate']*2000
        self.wind_mean_phi_list = mat_file['wind_mean_phi_rate']*360
        self.wind_mean_speed_list = mat_file['wind_mean_speed_rate']*4

        self.env_list = 0

        #self.reset()

    def _set_init_state(self):
        # set initial state randomly
        print(self.env_list)
        self.agent_x = self.agent_x_list[self.env_list][0]
        self.agent_y = self.agent_y_list[self.env_list][0]
        
        self.goal_x = self.goal_x_list[self.env_list][0]
        self.goal_y = self.goal_y_list[self.env_list][0]
        
        self.gas_d = self.gas_d_list[self.env_list][0]                # diffusivity [10m^2/s]
        self.gas_t = self.gas_t_list[self.env_list][0]            # gas life time [1000se$
        self.gas_q = self.gas_q_list[self.env_list][0]           # gas strength
        
        self.wind_mean_phi = self.wind_mean_phi_list[self.env_list][0]        # mean wind direction
        self.wind_mean_speed = self.wind_mean_speed_list[self.env_list][0]
        
        self.env_list += 1

register(
    id='StePfConvCentExtMatEnv-v0',
    entry_point='gym_ste_v2.envs.pf_conv_cent:StePfConvCentExtMatEnv',
)

