import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste.envs.ste_pf_easy import *

from gym.envs.registration import register


class StePFilterHardEnv(StePFilterEasyEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        StePFilterEasyEnv.__init__(self)

    def reset(self):
        self.count_actions = 0
        self.positions = []
        # set initial state randomly
        self.agent_x = self.np_random.uniform(low=0, high=self.court_lx)
        self.agent_y = self.np_random.uniform(low=0, high=self.court_ly)
        # self.agent_x = 5
        # self.agent_y = 10
        self.goal_x = self.np_random.uniform(low=0, high=self.court_lx)
        self.goal_y = self.np_random.uniform(low=0, high=self.court_lx)
        #self.goal_x = 44
        #self.goal_y = 44

        # self.gas_d = self.np_random.uniform(low=0, high=20)                # diffusivity [10m^2/s]
        # self.gas_t = self.np_random.uniform(low=500, high=1500)            # gas life time [1000se$
        # self.gas_q = self.np_random.uniform(low=1500, high=2500)           # gas strength
        # self.wind_mean_phi = self.np_random.uniform(low=0, high=360)        # mean wind direction
        self.gas_d = 10                 # diffusivity [10m^2/s]
        self.gas_t = 1000               # gas life time [1000sec]
        self.gas_q = 2000               # gas strength
        self.wind_mean_phi = 310        # mean wind direction [degree]

        self.dur_t = 0
        self.last_highest_conc = self.conc_eps

        self.pf_x = self.np_random.uniform(low=self.pf_low_state_x, high=self.pf_high_state_x)
        self.pf_y = self.np_random.uniform(low=self.pf_low_state_y, high=self.pf_high_state_y)
        self.pf_q = self.np_random.uniform(low=np.zeros(self.Wpnorms.size), high=np.ones(self.Wpnorms.size)*10000)
        self.Wps = np.ones(self.pf_num)/self.pf_num
        self.Wpnorms = self.Wps

        if self.goal_y == self.agent_y or self.goal_x == self.agent_x:
            self.reset()
        self.positions.append([self.agent_x, self.agent_y])
        if self.debug:
            print("x/y  - x/y", self.agent_x, self.agent_y, self.goal_x, self.goal_y)
            print("scale x/y  - x/y", self.agent_x*self.scale, self.agent_y*self.scale, self.goal_x*self.scale, self.goal_y*self.scale)

        obs = self._observation()
        if self.normalization:
            return self._normalize_observation(obs)
        else:
            return obs


register(
    id='StePFilterHardEnv-v0',
    entry_point='gym_ste.envs:StePFilterHardEnv',
)

