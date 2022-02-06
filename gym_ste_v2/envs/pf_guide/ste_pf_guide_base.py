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

        #### Observation space
        self.obs_low_state = np.array([-1, 0,        # wind_direction, wind speed (m/s)
                                        0, 0,        # current position
                                        0, 0,        # concentration, highest conc
                                        0, 0])       # est_location (x,y)
        self.obs_high_state = np.array([1, 20,
                                        self.court_lx, self.court_ly,
                                        self.conc_max, self.conc_max,
                                        1, 1])
        
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)


        #### Action space
        self.action_low = np.array([-1,   # Angle
                                    -1, -1, # Estimated source pos (x, y)
                                    -1])   # Done (Found source)
        self.action_high = np.array([1,
                                     1, 1,
                                     1])

        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)

#        self.conv_eps = 1
        # set a seed and reset the environment
        self.seed()
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
                        float(self.agent_x), float(self.agent_y),
                        float(self.gas_measure), float(self.last_highest_conc),
                        float(self.est_location[0]/self.court_lx), float(self.est_location[1]/self.court_ly)])
        return obs


    def step(self, action):
        #print(self.env_sig)
        self.warning = False
        self.outborder = False
        self.count_actions += 1
        self._calculate_position(action[0])


        self.est_location = np.array([ (action[1]+1)*self.court_lx/2, (action[2]+1)*self.court_ly/2])

        obs = self._observation()

        self.last_action = action

        # done for step rewarding
        agent_dist = self.agent_v*self.delta_t
        self.cov_val = np.sqrt(self.CovXxp + self.CovXyp)

        uncertainty_of_source = self.cov_val/self.conv_eps
        found_done = bool(action[3] > 0.1)
        #done = bool(self._distance(self.agent_x, self.agent_y) <= self.eps)

        rew = 0
        '''
        if self.outborder: # Agent get out to search area
           rew += self._border_reward()
        '''

        #print("action (x, y)", action[1]*self.court_lx, action[2]*self.court_ly)
        self.pf_center = np.array([np.sum(self.pf_x*self.Wpnorms), np.sum(self.pf_y*self.Wpnorms)])
        #self.est_location = np.array([action[1]*self.court_lx, action[2]*self.court_ly])

        est_pf_dist = math.sqrt( pow(self.est_location[0]-self.pf_center[0],2) + pow(self.est_location[1]-self.pf_center[1],2) )
        if est_pf_dist < 1.0:
            rew += 0.01
        if self.gas_measure > self.last_highest_conc: # need to be adjusted for different source condition
            rew += 0.01 #+1
        if found_done: # particle filter is converged
            est_source_dist = math.sqrt( pow(self.est_location[0]-self.goal_x,2) + pow(self.est_location[1]-self.goal_y,2) )
            nearby_bool = bool(est_source_dist < self.eps)
            if nearby_bool:
                rew = self._reward_goal_reached()

        terminate_done = bool(self.count_actions >= self.max_step)

        # break if more than max_step actions taken
        done = bool(self.count_actions >= self.max_step or found_done or self.outborder)
#        done = bool(self.count_actions >= self.max_step or found_done)

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
