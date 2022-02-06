import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste_v2.envs.common.ste_pf_base import *

from gym.envs.registration import register

from datetime import datetime


class BaseEnv(StePFilterBaseEnv):

    def __init__(self):
        StePFilterBaseEnv.__init__(self)

        #self.agent_v = 6                # 2m/s
        #self.agent_dist = self.agent_v * self.delta_t

        self.pf_low_state_x = np.zeros(self.pf_num) # particle filter (x1,x2,x3, ...)
        self.pf_low_state_y = np.zeros(self.pf_num) # particle filter (y1,y2,y3, ...)
        self.pf_low_state_q = np.zeros(self.pf_num) # particle filter (q1,q2,q3, ...)
        pf_low_state_wp = np.zeros(self.pf_num) # particle filter (w1,w2,w3, ...)
        etc_low_state = np.array([self.agent_dist, self.gas_d, self.gas_t, self.court_lx, self.court_ly, self.pf_num, self.sensor_sig_m, self.env_sig,
                                  -1, 0, 0, 0, 0, -1, 0, 0, 0])
        # [dist, gas_d, gas_t, court_lx, court_ly, pf_num, sensor_sig, env_sig, 
        #   wind_dir, wind speed (m/s), duration, current pos (x,y), last action (direction), last conc, conc, highest conc, ]

        self.obs_low_state = np.concatenate((etc_low_state, self.pf_low_state_x, self.pf_low_state_y, self.pf_low_state_q, pf_low_state_wp), axis=None)


        self.pf_high_state_x = np.ones(self.pf_num)*self.court_lx
        self.pf_high_state_y = np.ones(self.pf_num)*self.court_ly
        self.pf_high_state_q = np.ones(self.pf_num)*self.max_q
        pf_high_state_wp = np.ones(self.pf_num)
        etc_high_state = np.array([self.agent_dist, self.gas_d, self.gas_t, self.court_lx, self.court_ly, self.pf_num, self.sensor_sig_m, self.env_sig, 
                                   1, 20,  self.max_step, self.court_lx, self.court_ly, 1, self.conc_max, self.conc_max, self.conc_max])

        self.obs_high_state = np.concatenate((etc_high_state, self.pf_high_state_x, self.pf_high_state_y, self.pf_high_state_q, pf_high_state_wp), axis=None)

        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

        self.normalization = False


    def _observation(self):
        self.wind_d, self.wind_s = self._wind_sensor() # wind direction & speed
        #print("env_wind_d: ", self.wind_d)
        moved_dist = math.sqrt(pow(self.last_x - self.agent_x,2) + pow(self.last_y - self.agent_y,2))
#        print("------------------------------------------------------")
        self.gas_measure = self._gas_measure()
#        self._particle_filter()
        self.pf_x, self.pf_y, self.pf_q, self.Wpnorms = self.particle_filter._weight_update(self.gas_measure, self.agent_x, self.agent_y,
                                                                                            self.pf_x, self.pf_y, self.pf_q, self.Wpnorms,
                                                                                            self.wind_d, self.wind_s)

        self.CovXxp = np.var(self.pf_x)
        self.CovXyp = np.var(self.pf_y)
        self.CovXqp = np.var(self.pf_q)

        # x_warning, y_warning = self._boundary_warning_sensor()

        etc_state = np.array([float(self.agent_v*self.delta_t), float(self.gas_d), float(self.gas_t), float(self.court_lx), float(self.court_ly), float(self.pf_num), float(self.sensor_sig_m), float(self.env_sig),
                              float(self.wind_d/math.pi), float(self.wind_s), float(self.dur_t), float(self.agent_x), float(self.agent_y),
                              float(self.last_action), float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc)])
        # wind_d [radian]
        return np.concatenate((etc_state, self.pf_x, self.pf_y, self.pf_q, self.Wpnorms), axis=None)


#    def _reward_goal_reached(self):
#        return 1000

    def step(self, action):
        self.warning = False
        self.outborder = False
        self.count_actions += 1
        self._calculate_position(action)
        obs = self._observation()

        self.last_action = action

        #print("agnet_x: ", self.agent_x, "       |       agent_y: ", self.agent_y)



        #x_warning, y_warning = self._boundary_warning_sensor()

        # done for step rewarding
        self.cov_val = np.sqrt(self.CovXxp + self.CovXyp)
        converge_done = bool(self.cov_val < 1)
        #done = bool(self._distance(self.agent_x, self.agent_y) <= self.eps)

        rew = 0
#        if self.outborder: # Agent get out to search area
#           rew += self._border_reward()

        pf_center = np.array([np.mean(self.pf_x), np.mean(self.pf_y)])
        nearby = math.sqrt( pow(pf_center[0]-self.goal_x,2) + pow(pf_center[1]-self.goal_y,2) )
        if not converge_done:
            rew += 0
#            rew += self._step_reward()
        else: # particle filter is converged
            nearby_bool = bool(nearby<1)
            if nearby_bool:
                rew = self._reward_goal_reached()

        terminate_done = bool(self.count_actions >= self.max_step)

        # break if more than max_step actions taken
        done = bool(self.count_actions >= self.max_step or converge_done or self.outborder)

        # track, where agent was
        self.positions.append([self.agent_x, self.agent_y])
        self.last_x = self.agent_x
        self.last_y = self.agent_y

        if self.gas_measure > self.last_highest_conc:
           self.last_highest_conc = self.gas_measure

        self.last_measure = self.gas_measure

        norm_obs = []
        if self.normalization:
           norm_obs = []
           norm_obs = self._normalize_observation(obs)
           info = self._info_function(norm_obs, action, done, rew)
           return norm_obs, rew, done, info
        else:
           info = self._info_function(obs, action, done, rew)
           return obs, rew, done, info
