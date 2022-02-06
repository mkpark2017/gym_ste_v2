import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste_v2.envs.common.ste_pf_base import *

from gym.envs.registration import register

from datetime import datetime

import warnings
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')

class BaseEnv(StePFilterBaseEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        StePFilterBaseEnv.__init__(self)

        self.obs_low_state = np.array([-1, 0,        # wind_direction, wind speed (m/s)
                                        0,           # duration time
                                        0, 0,        # current position
                                       -1,           # last action
                                        0, 0, 0,     # last conc, current conc highest conc
                                        0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0]) # mean_x, mean_y, cov_x, cov_y, weight

        self.obs_high_state = np.array([1, 20,
                                        self.max_step,
                                        self.court_lx, self.court_ly,
                                        1,
                                        self.conc_max, self.conc_max, self.conc_max,
                                        self.court_lx, self.court_ly, pow(self.court_lx,2), pow(self.court_ly,2), 1,
                                        self.court_lx, self.court_ly, pow(self.court_lx,2), pow(self.court_ly,2), 1,
                                        self.court_lx, self.court_ly, pow(self.court_lx,2), pow(self.court_ly,2), 1])
        
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

        self.gmm_num = 3
        self.gmm = GaussianMixture(n_components=self.gmm_num, max_iter=20)
        #self.gmm_mean_x = np.ones(self.gmm_num)
        #self.gmm_mean_y = np.ones(self.gmm_num)
        #self.gmm_cov_x = np.ones(self.gmm_num)
        #self.gmm_cov_y = np.ones(self.gmm_num)

        #self.gmm_mean_x = []
        #self.gmm_mean_y = []
        #self.gmm_cov_x = []
        #self.gmm_cov_y = []


        #self.conv_eps = 1
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

#        print(self.particle_filter.update_count)
        if self.particle_filter.update_count == 0:
#            self.gmm = GaussianMixture(n_components=self.gmm_num, max_iter=5)
            pf_con = np.column_stack((self.pf_x,self.pf_y))
            #self.gmm.fit(pf_con)
            gmm_labels = self.gmm.fit_predict(pf_con)
            self.gmm_weights = self.gmm.weights_
            
            self.gmm_mean_x = np.ones(self.gmm_num)
            self.gmm_mean_y = np.ones(self.gmm_num)
            self.gmm_cov_x = np.ones(self.gmm_num)
            self.gmm_cov_y = np.ones(self.gmm_num)
            self.gmm_data = []
            for k in range(0,self.gmm_num):
                self.gmm_data.append(pf_con[gmm_labels == k])
                gmm_Wpnorms = self.Wpnorms[gmm_labels == k]
                gmm_Wpnorms = gmm_Wpnorms/sum(gmm_Wpnorms)
                data_split = np.transpose(self.gmm_data[k])

                self.gmm_mean_x[k] = sum(data_split[0] * gmm_Wpnorms)
                self.gmm_mean_y[k] = sum(data_split[1] * gmm_Wpnorms)
                #self.gmm_mean_q[k] = sum(data_split[2] * gmm_Wpnorms)
                #print(k, "th sample num", np.shape(data_split[0])[0])
                if np.shape(data_split[0])[0] == 0:
                    self.gmm_cov_x[k] = 0
                    self.gmm_cov_y[k] = 0
                #    time.sleep(1)
                else:
                    self.gmm_cov_x[k] = np.var(data_split[0])
                    self.gmm_cov_y[k] = np.var(data_split[1])
                    #self.gmm_cov_q[k] = np.var(data_split[2])



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
                        float(self.gmm_mean_x[0]), float(self.gmm_mean_y[0]), float(self.gmm_cov_x[0]), float(self.gmm_cov_y[0]), float(self.gmm_weights[0]),
                        float(self.gmm_mean_x[1]), float(self.gmm_mean_y[1]), float(self.gmm_cov_x[1]), float(self.gmm_cov_y[1]), float(self.gmm_weights[1]),
                        float(self.gmm_mean_x[2]), float(self.gmm_mean_y[2]), float(self.gmm_cov_x[2]), float(self.gmm_cov_y[2]), float(self.gmm_weights[2])])

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



    def reset(self):
        print("Reset")
        self.gmm_data = []
        #self.gmm = None
        #self.gmm = GaussianMixture(n_components=self.gmm_num, max_iter=5)

        self.count_actions = 0
        self.positions = []
        self.measures = []
        self.agent_xs = []
        self.agent_ys = []
        # set initial state randomly
        self._set_init_state()

        self.last_x = self.agent_x-self.agent_dist
        self.last_y = self.agent_y

        self.dur_t = 0
        self.last_highest_conc = 0


        self.particle_filter = ParticleFilter(self)
        self.pf_x = self.np_random.uniform(low=self.pf_low_state_x, high=self.pf_high_state_x)
        self.pf_y = self.np_random.uniform(low=self.pf_low_state_y, high=self.pf_high_state_y)
        self.pf_q = self.np_random.uniform(low=np.zeros(self.Wpnorms.size), high=np.ones(self.Wpnorms.size)*self.max_q)
        self.Wps = np.ones(self.pf_num)/self.pf_num
        self.Wpnorms = self.Wps


        self.CovXxp = np.var(self.pf_x)
        self.CovXyp = np.var(self.pf_y)
        self.cov_val = np.sqrt(pow(self.CovXxp,2) + pow(self.CovXyp,2))
        self.cov_last_highest = self.cov_val

        if math.sqrt(pow(self.goal_y - self.agent_y,2) + pow(self.goal_x - self.agent_x,2) ) < self.court_lx*0.3:
            #print("Reset")
            self.reset()
        self.positions.append([self.agent_x, self.agent_y])
        #print("x: ", self.agent_x, " y: ", self.agent_y)
        if self.debug:
            print("x/y  - x/y", self.agent_x, self.agent_y, self.goal_x, self.goal_y)
            print("scale x/y  - x/y", self.agent_x*self.scale, self.agent_y*self.scale, self.goal_x*self.scale, self.goal_y*self.scale)


        obs = self._observation()
#        print("==============================================")
        if self.normalization:
            return self._normalize_observation(obs)
        else:
            return obs

