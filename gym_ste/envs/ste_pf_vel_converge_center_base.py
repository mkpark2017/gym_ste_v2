import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste.envs.ste_pf_base import *

from gym.envs.registration import register

from datetime import datetime


class StePFilterVelConvCenterBaseEnv(StePFilterBaseEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        StePFilterBaseEnv.__init__(self)

        self.min_vel = 1
        self.max_vel = 10
        self.agent_min_dist = max(2,self.min_vel) * self.delta_t

        self.obs_low_state = np.array([-1,  0,            #wind sensor measure (dircetion, vel)
                                        0,                #durtion time
                                        0,  0,            #current pos
                                       -1,  0,            #last action (direction, vel)
                                        0,  0,  0,        #last conc, conc, highest conc
                                        0,  0,  0,  0])   #particle filter (mean_x, mean_y, cov_x, cov_y)
        self.obs_high_state = np.array([1, 20,
                                        self.max_step,
                                        self.court_lx, self.court_ly,
                                        1, 1,
                                        self.conc_max, self.conc_max, self.conc_max,
                                        self.court_lx, self.court_ly, pow(self.court_lx,2), pow(self.court_ly,2)])
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)


        self.action_low = np.array([-1, 0]) # angle, velocity
        self.action_high = np.array([1, 1])

        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)


        #self.last_action = np.array([0, 0]);           # last action

        # set a seed and reset the environment
        self.seed()
#        self.reset()

    def _calculate_position(self, action):
        angle = (action[0]) * math.pi
        step_size = (action[1] * (self.max_vel-self.min_vel) + self.min_vel) * self.delta_t
        # calculate new agent state
        self.agent_x = self.agent_x + math.cos(angle) * step_size
        self.agent_y = self.agent_y + math.sin(angle) * step_size

        # borders
        if self.agent_x < 0:
            self.agent_x = 0
            self.outborder = True
        if self.agent_x > self.court_lx:
            self.agent_x = self.court_lx
            self.outborder = True
        if self.agent_y < 0:
            self.agent_y = 0
            self.outborder = True
        if self.agent_y > self.court_ly:
            self.agent_y = self.court_ly
            self.outborder = True


    def _observation(self):
        self.wind_d, self.wind_s = self._wind_sensor() # wind direction & speed
        moved_dist = math.sqrt(pow(self.last_x - self.agent_x,2) + pow(self.last_y - self.agent_y,2))
#        print("------------------------------------------------------")
        self.gas_measure = self._gas_measure()
#        self._particle_filter()
        self.pf_x, self.pf_y, self.pf_q, self.Wpnorms = self.particle_filter._weight_update(self.gas_measure, self.agent_x, self.agent_y,
                                                                                            self.pf_x, self.pf_y, self.pf_q, self.Wpnorms,
                                                                                            self.wind_d, self.wind_s)

        #print(self.last_action)

        self.CovXxp = np.var(self.pf_x)
        self.CovXyp = np.var(self.pf_y)
        self.CovXqp = np.var(self.pf_q)

#        x_warning, y_warning = self._boundary_warning_sensor()
        mean_x = sum(self.pf_x * self.Wpnorms)
        mean_y = sum(self.pf_y * self.Wpnorms)
        mean_q = sum(self.pf_q * self.Wpnorms)

        obs = np.array([float(self.wind_d/math.pi), float(self.wind_s),
                        float(self.dur_t),
                        float(self.agent_x), float(self.agent_y),
                        float(self.last_action[0]), float(self.last_action[1]),
                        float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc),
                        float(mean_x), float(mean_y), float(self.CovXxp), float(self.CovXyp)])

        return obs

    def _info_function(self, obs, action, done, rew):
        if self.normalization:
           obs = obs*(self.obs_high_state - self.obs_low_state) + self.obs_low_state
           info = "time step:" + str(self.count_actions) + ", act: (" + str(
                   round(float(action[0])*180,2)) +  ", " +  str(
                   round(float((action[1] * (self.max_vel-self.min_vel) + self.min_vel) * self.delta_t),2)) + "), local flow: (" + str(
                   round(obs[0]*180,2)) + "degree, " + str(
                   round(obs[1],2)) + "), duration time: " + str(round(obs[2],2)) + ", current pos: (" + str(
                   round(obs[3],2)) + ", " + str(round(obs[4],2)) + "), last action: " + str(
                   round(float(obs[5])*180,2)) + ", " + str(round(float((obs[6] * (self.max_vel-self.min_vel) + self.min_vel) * self.delta_t),2)) + ", last conc:" + str(
                   round(obs[7],2)) + ", conc:" + str(
                   round(obs[8],2)) + ", highest conc:" + str(round(obs[9],2)) +", particles_x : " + str(
                   np.round(obs[10:10+self.pf_num],2)) + ", particles_y : " + str(
                   np.round(obs[10+self.pf_num:10+self.pf_num*2],2)) + ", particles_wp : " + str(
                   np.round(obs[10+self.pf_num*2:10+self.pf_num*3],2)) + ", current pos: (" + str(round(self.agent_x,2)) + "," + str(
                   round(self.agent_y,2)) + ")", "goal pos: (" + str(round(self.goal_x,2)) + "," + str(round(self.goal_y,2)) + "), done: " + str(done) + ", rew:" + str(rew)
        else:
           info = "need to edit"
           #info = "time step:" + str(self.count_actions) + ", act:" + str(
           #        round(float(action)*180,2)) + ", local flow: (" + str(round(obs[0],2)) + ", " + str(
           #        round(obs[1],2)) + "), last pos: (" + str(round(obs[2],2)) + ", " + str(
           #        round(obs[3],2)) + "), dur_t from last capture: " + str(round(obs[4],2)) + ", last action: " + str(
           #        round(float(obs[5]),2)) + ", last conc" + str(round(obs[6],2)) + ", conc" + str(round(obs[7],2)) + ", last highest conc" + str(
           #        round(obs[8],2)) + ", rew:" + str(rew) + ", current pos: (" + str(round(self.agent_x,2)) + "," + str(
           #        round(self.agent_y,2)) + ")", "goal pos: (" + str(self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)
        return info

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
        self.wind_mean_speed = 2

        self.last_action = np.array([0, 0])
        self.last_highest_conc = 0
        self.last_measure = 0

        self.last_x = self.agent_x
        self.last_y = self.agent_y


    def _reward_goal_reached(self):
        return 1000 #1000 #100

    def _step_reward(self):
        if self.gas_measure > self.last_highest_conc: # need to be adjusted for different source condition
            reward = 0.1 #0.1 #+1
            self.dur_t = 0
        else:
            reward = -0.1 #-0.1 #0
            self.dur_t += 1
        return reward

    def _border_reward(self):
        reward = -10 #100 #-10
        return reward


    def step(self, action):
        self.warning = False
        self.outborder = False
        self.count_actions += 1
        self._calculate_position(action)
        obs = self._observation()

        #print("last_x", self.last_x, "last_y", self.last_y, "current_x", self.agent_x, "current_y", self.agent_y, 'distance', math.sqrt(pow(self.last_x-self.agent_x,2)+pow(self.last_y-self.agent_y,2)))
        self.last_action = action

        #x_warning, y_warning = self._boundary_warning_sensor()

        # done for step rewarding
        self.cov_val = np.sqrt(self.CovXxp + self.CovXyp)
        converge_done = bool(self.cov_val < 1)
        done = bool(self._distance(self.agent_x, self.agent_y) <= self.eps)

        rew = 0
        if self.outborder: # Agent get out to search area
           rew += self._border_reward()

        pf_center = np.array([np.mean(self.pf_x), np.mean(self.pf_y)])
        nearby = math.sqrt( pow(pf_center[0]-self.goal_x,2) + pow(pf_center[1]-self.goal_y,2) )
        if not converge_done:
            rew += self._step_reward()
        else: # particle filter is converged
            nearby_bool = bool(nearby < 1)
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
