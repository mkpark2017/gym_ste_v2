import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste.envs.ste_pf_base import *

from gym.envs.registration import register

from datetime import datetime


class StePFilterConvBaseEnv(StePFilterBaseEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        StePFilterBaseEnv.__init__(self)
        self.agent_dist = self.agent_v * self.delta_t

#        self.pf_num = 30
#        self.pf_low_state_x = np.zeros(self.pf_num) # particle filter (x1,x2,x3, ...)
#        self.pf_low_state_y = np.zeros(self.pf_num) # particle filter (y1,y2,y3, ...)
#        pf_low_state_wp = np.zeros(self.pf_num) # particle filter (q1,q2,q3, ...)
#        etc_low_state = np.array([-1, -20, 0, -1, 0, 0, 0]) # [wind_direction, wind speed (m/s), last action (only direction), last concentration, concentration (max 100 mg/m^3), last highest conc]
#        etc_low_state = np.array([-1, 0, -self.agent_dist, -self.agent_dist, -1, 0, 0, 0]) # [wind_direction, wind speed (m/s), boundary warning (x,y), last action (only direction), last concentration, concentration (max 100 mg/m^3)]
#        self.obs_low_state = np.concatenate((etc_low_state, self.pf_low_state_x, self.pf_low_state_y, pf_low_state_wp), axis=None)

#        self.conc_max = 100
#        self.pf_high_state_x = np.ones(self.pf_num)*self.court_lx
#        self.pf_high_state_y = np.ones(self.pf_num)*self.court_ly
#        pf_high_state_wp = np.ones(self.pf_num)
#        etc_high_state = np.array([1, 20,  self.max_step,  1, self.conc_max, self.conc_max, self.conc_max])
#        etc_high_state = np.array([1, 20,  self.agent_dist, self.agent_dist, 1, self.conc_max, self.conc_max, self.conc_max])
#        self.obs_high_state = np.concatenate((etc_high_state, self.pf_high_state_x, self.pf_high_state_y, pf_high_state_wp), axis=None)
#        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

#        self.Wps = np.ones(self.pf_num)/self.pf_num
#        self.Wpnorms = self.Wps

    def _border_reward(self):
        reward = -1 #-100
        return reward

    def _reward_goal_reached(self):
        return 10 #100

    def _reward_failed(self):
        return 0 #-100

    def _set_init_state(self):
        # set initial state randomly
        self.agent_x = self.np_random.uniform(low=0, high=self.court_lx)
        self.agent_y = self.np_random.uniform(low=0, high=self.court_ly)
        self.goal_x = 44
        self.goal_y = 44

        self.gas_d = 10                 # diffusivity [10m^2/s]
        self.gas_t = 1000               # gas life time [1000sec]
        self.gas_q = 2000               # gas strength
        self.wind_mean_phi = 310        # mean wind direction [degree]

    def _boundary_warning_sensor(self):
        dist_x = self.court_lx - self.agent_x
        if dist_x == 0: dist_x = 1e-2
        dist_y = self.court_ly - self.agent_y
        if dist_y == 0: dist_y = 1e-2
        x_warning = 0
        y_warning = 0
        if dist_x < self.agent_dist*0.99:
            x_warning = dist_x
            self.warning = True
        if dist_y < self.agent_dist*0.99:
            y_warning = dist_y
            self.warning = True
        dist_x = 0 - self.agent_x
        if dist_x == 0: dist_x = -1e-2
        dist_y = 0 - self.agent_y
        if dist_y == 0: dist_y = -1e-2
        if abs(dist_x) < self.agent_dist*0.99:
            x_warning = dist_x
            self.warning = True
        if abs(dist_y) < self.agent_dist*0.99:
            y_warning = dist_y
            self.warning = True

        return x_warning, y_warning

    def _observation(self):
        self.wind_d, self.wind_s = self._wind_sensor() # wind direction & speed
        self.last_x = self.agent_x
        self.last_y = self.agent_y

        self.gas_measure = self._gas_measure(self.agent_x, self.agent_y)
        self._particle_filter()

        x_warning, y_warning = self._boundary_warning_sensor()

        etc_state = np.array([float(self.wind_d/math.pi), float(self.wind_s), x_warning, y_warning, float(self.last_action), float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc)])

        return np.concatenate((etc_state, self.pf_x-self.agent_x, self.pf_y-self.agent_y, self.Wpnorms), axis=None)

    def step(self, action):
        self.warning = False
        self.outborder = False
        self.count_actions += 1
        self._calculate_position(action)
        obs = self._observation()

        self.last_action = action

        x_warning, y_warning = self._boundary_warning_sensor()

        # done for step rewarding
        agent_dist = self.agent_v*self.delta_t
        self.cov_val = np.sqrt(self.CovXxp + self.CovXyp)
        converge_done = bool(self.cov_val < agent_dist)
        done = bool(self._distance(self.agent_x, self.agent_y) <= self.eps)

        rew = 0
        if self.outborder: # Agent get out to search area
           rew += self._border_reward()
#        if not converge_done:
#            rew = self._step_reward()
        pf_center = np.array([np.mean(self.pf_x), np.mean(self.pf_y)])
        nearby = math.sqrt( pow(pf_center[0]-self.goal_x,2) + pow(pf_center[1]-self.goal_y,2) )
        if not converge_done:
            rew += self._step_reward()
        else: # particle filter is converged
            nearby_bool = bool(nearby<agent_dist*2)
            if nearby_bool:
                rew = self._reward_goal_reached()
#            else:
#                rew = self._reward_goal_reached()/2
#            else:
#                rew = self._reward_failed()

#        source_done = bool(self._distance(self.agent_x, self.agent_y) <= self.eps)
#        if source_done:
#            rew = self._reward_goal_reached()

        terminate_done = bool(self.count_actions >= self.max_step)
#        if terminate_done: # Reach the max_step without finding source
#            rew = self._reward_failed()

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
           norm_obs = self._normalize_observation(obs)
           warning_x_text = str(norm_obs[2]*(self.agent_dist*2)-self.agent_dist)
           warning_y_text = str(norm_obs[3]*(self.agent_dist*2)-self.agent_dist)

           info = "time step:" + str(self.count_actions) + ", act:" + str(
                   round(float(action)*180,2)) + ", local flow: (" + str(round(norm_obs[0]*180,2)) + "degree, " + str(
                   round(norm_obs[1]*20,2)) + "m/s), Range warning: (" + str(
                   warning_x_text) + ", " + str(warning_y_text) + "), last acton: " + str(
                   round(float(norm_obs[4])*180,2)) + ", last conc:" + str(round(norm_obs[5]*self.conc_max,2)) + ", conc:" + str(
                   round(norm_obs[6]*self.conc_max,2)) + ", highest conc:" + str(round(norm_obs[7]*self.conc_max,2)) +", particles_x : " + str(
                   np.round(norm_obs[8:8+self.pf_num]*self.court_lx,2)) + ", particles_y : " + str(
                   np.round(norm_obs[8+self.pf_num:8+self.pf_num*2]*self.court_lx,2)) + ", particles_wp : " + str(
                   np.round(norm_obs[8+self.pf_num*2:8+self.pf_num*3],2)) + ", rew:" + str(rew) + ", current pos: (" + str(round(self.agent_x,2)) + "," + str(
                   round(self.agent_y,2)) + ")", "goal pos: (" + str(round(self.goal_x,2)) + "," + str(round(self.goal_y,2)) + "), done: " + str(done) + ", Dist:" + str(nearby)
           return norm_obs, rew, done, info


# register(
#     id='StePFilterConvEasyEnv-v0',
#     entry_point='gym_ste.envs:StePFilterConvEasyEnv',
# )

