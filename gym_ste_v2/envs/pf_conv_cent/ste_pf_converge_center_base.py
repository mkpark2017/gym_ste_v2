import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste_v2.envs.common.ste_pf_base import *

from gym.envs.registration import register

from datetime import datetime


class BaseEnv(StePFilterBaseEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        StePFilterBaseEnv.__init__(self)

        #self.agent_v = 6                # 2m/s
        #self.agent_dist = self.agent_v * self.delta_t

        self.obs_low_state = np.array([-1, 0,        # wind_direction, wind speed (m/s)
                                        0,           # duration time
                                        0, 0,        # current position
                                       -1,           # last action
                                        0, 0, 0,     # last conc, current conc highest conc
                                        0, 0, 0, 0]) # mean_x, mean_y, cov_x, cov_y
        self.obs_high_state = np.array([1, 20,
                                        self.max_step,
                                        self.court_lx, self.court_ly,
                                        1,
                                        self.conc_max, self.conc_max, self.conc_max,
                                        self.court_lx, self.court_ly, pow(self.court_lx,2), pow(self.court_ly,2)])
        
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

#        self.conv_eps = 1
        # set a seed and reset the environment
        seed = self.seed(8201085478471)
        print("Seed: ", seed)
#        self.reset()

    def _observation(self):
        self.wind_d, self.wind_s = self._wind_sensor() # wind direction & speed
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

#        x_warning, y_warning = self._boundary_warning_sensor()
        mean_x = sum(self.pf_x * self.Wpnorms)
        mean_y = sum(self.pf_y * self.Wpnorms)

        obs = np.array([float(self.wind_d/math.pi), float(self.wind_s),
                        float(self.dur_t),
                        float(self.agent_x), float(self.agent_y),
                        float(self.last_action),
                        float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc),
                        float(mean_x), float(mean_y), float(self.CovXxp), float(self.CovXyp)])

        return obs

    def step(self, action):
        #print(self.env_sig)
        self.warning = False
        self.outborder = False
        self.count_actions += 1
        self._calculate_position(action)
        obs = self._observation()

        self.last_action = action

        #x_warning, y_warning = self._boundary_warning_sensor()

        # done for step rewarding
        agent_dist = self.agent_v*self.delta_t
        self.cov_val = np.sqrt(self.CovXxp + self.CovXyp)
        converge_done = bool(self.cov_val < self.conv_eps)
        #done = bool(self._distance(self.agent_x, self.agent_y) <= self.eps)

        rew = 0
        if self.outborder: # Agent get out to search area
           rew += self._border_reward()

        pf_center = np.array([np.mean(self.pf_x), np.mean(self.pf_y)])
        nearby = math.sqrt( pow(pf_center[0]-self.goal_x,2) + pow(pf_center[1]-self.goal_y,2) )
        if not converge_done:
            rew += self._step_reward()
        else: # particle filter is converged
            nearby_bool = bool(nearby<self.eps)
            if nearby_bool:
                rew = self._reward_goal_reached()

        terminate_done = bool(self.count_actions >= self.max_step)

        # break if more than max_step actions taken
        done = bool(self.count_actions >= self.max_step or converge_done or self.outborder)
#        done = bool(self.count_actions >= self.max_step or converge_done)

        # track, where agent was
        self.positions.append([self.agent_x, self.agent_y])
        self.measures.append(self.gas_measure)
        self.agent_xs.append(self.agent_x)
        self.agent_ys.append(self.agent_y)


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
