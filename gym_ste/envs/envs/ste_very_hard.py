import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste.envs.ste_base import *

from gym.envs.registration import register

from datetime import datetime


class SteVeryHardEnv(SteBaseEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        SteBaseEnv.__init__(self)

        self.reset()

    def _set_init_state(self):
        # set initial state randomly
        self.agent_x = self.np_random.uniform(low=0, high=self.court_lx)
        self.agent_y = self.np_random.uniform(low=0, high=self.court_ly)
        # self.agent_x = 5
        # self.agent_y = 10
        # self.goal_x = self.np_random.uniform(low=0, high=self.court_lx)
        # self.goal_y = self.np_random.uniform(low=0, high=self.court_lx)
        self.goal_x = self.np_random.uniform(low=self.court_lx*0.2, high=self.court_lx*0.8)
        self.goal_y = self.np_random.uniform(low=self.court_ly*0.2, high=self.court_ly*0.8)
        # self.goal_x = 44
        # self.goal_y = 44

        # self.gas_d = self.np_random.uniform(low=0, high=20)                # diffusivity [10m^2/s]
        # self.gas_t = self.np_random.uniform(low=500, high=1500)            # gas life time [1000se$
        # self.gas_q = self.np_random.uniform(low=1500, high=2500)           # gas strength
        # wind_angle = math.atan2(self.goal_y - self.court_ly/2, self.goal_x - self.court_lx/2)/math.pi * 180 + 270

        # self.wind_mean_phi = self.np_random.uniform(low=wind_angle-5, high=wind_angle+5)        # mean wind direction
        self.wind_mean_phi = self.np_random.uniform(low=0, high=360)
        self.gas_d = 10                 # diffusivity [10m^2/s]
        self.gas_t = 1000               # gas life time [1000sec]
        self.gas_q = 2000               # gas strength
        # self.wind_mean_phi = 310        # mean wind direction [degree]

register(
    id='SteVeryHardEnv-v0',
    entry_point='gym_ste.envs:SteVeryHardEnv',
)

