import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste.envs.ste_pf_base import *

from gym.envs.registration import register

from datetime import datetime


class StePFilterRelativeBaseEnv(StePFilterBaseEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        StePFilterBaseEnv.__init__(self)

#        self.pf_low_state_x = np.zeros(self.pf_num) # particle filter (x1,x2,x3, ...)
#        self.pf_low_state_y = np.zeros(self.pf_num) # particle filter (y1,y2,y3, ...)
        self.pf_low_state_x = -np.ones(self.pf_num)*self.court_lx
        self.pf_low_state_y = -np.ones(self.pf_num)*self.court_ly

        pf_low_state_wp = np.zeros(self.pf_num) # particle filter (q1,q2,q3, ...)
        
        etc_low_state = np.array([-1, 0, 0, 0, 0, -1, 0, 0, 0]) # [wind_direction, wind speed (m/s), duration time, current pos (x,y), last action (direction), last concentration, concentration, highest conc


        self.obs_low_state = np.concatenate((etc_low_state, self.pf_low_state_x, self.pf_low_state_y, pf_low_state_wp), axis=None)

        self.conc_max = 100
        self.pf_high_state_x = np.ones(self.pf_num)*self.court_lx
        self.pf_high_state_y = np.ones(self.pf_num)*self.court_ly
        pf_high_state_wp = np.ones(self.pf_num)
        
        etc_high_state = np.array([1, 20,  self.max_step, self.court_lx, self.court_ly, 1, self.conc_max, self.conc_max, self.conc_max])
        self.obs_high_state = np.concatenate((etc_high_state, self.pf_high_state_x, self.pf_high_state_y, pf_high_state_wp), axis=None)
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

        self.Wps = np.ones(self.pf_num)/self.pf_num
        self.Wpnorms = self.Wps

        self.seed()
#        self.reset()

    def _observation(self):
        self.wind_d, self.wind_s = self._wind_sensor() # wind direction & speed
        moved_dist = math.sqrt(pow(self.last_x - self.agent_x,2) + pow(self.last_y - self.agent_y,2))
#        print("------------------------------------------------------")
        self.gas_measure = self._gas_measure(self.agent_x, self.agent_y)
        self._particle_filter()
#        x_warning, y_warning = self._boundary_warning_sensor()

        etc_state = np.array([float(self.wind_d/math.pi), float(self.wind_s), float(self.dur_t), float(self.agent_x), float(self.agent_y), float(self.last_action), float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc)])


        return np.concatenate((etc_state, self.pf_x-self.agent_x, self.pf_y-self.agent_y, self.Wpnorms), axis=None)

