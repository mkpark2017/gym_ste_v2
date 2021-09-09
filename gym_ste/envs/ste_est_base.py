import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste.envs.ste_base import *

from gym.envs.registration import register

from datetime import datetime


class SteEstBaseEnv(SteBaseEnv):
    def __init__(self):
        SteBaseEnv.__init__(self)
<<<<<<< HEAD
#       [wind_direction, wind speed (m/s), duration time, current pos (x,y), last action (only direction), last prob, current action, current prob, last concentration, concentration (max 100 mg/m^3), highest conc]
        self.obs_low_state = np.array([-1, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0]) 
        self.obs_high_state = np.array([1, 20,  self.max_step, self.court_lx, self.court_ly, 1, 1, 1, 1, self.conc_max, self.conc_max, self.conc_max])
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

#       [agent angle, source probability]
        self.action_low = np.array([-1, 0])
        self.action_high = np.array([1, 1])
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)
=======
#       [wind_direction, wind speed (m/s), duration time, current pos (x,y), last action (only direction), last concentration, concentration (max 100 mg/m^3), highest conc]
        self.obs_low_state = np.array([-1, 0, 0, 0, 0, -1, 0, 0, 0]) 
        self.obs_high_state = np.array([1, 20,  self.max_step, self.court_lx, self.court_ly, 1, self.conc_max, self.conc_max, self.conc_max])
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

#       [agent angle, source probability]
        self.action_angle_low = np.array([-1, 0])
        self.action_angle_high = np.array([1, 1])
        self.action_space = spaces.Box(self.action_angle_low, self.action_angle_high, dtype=np.float32)
>>>>>>> 1211906308dfc7d77a553a01c03d1cd648aa7ea3
        
        self.eps = 0.1*self.court_lx
        self.conc_eps = 0.2 # minimum conc
        self.last_highest_conc = self.conc_eps
        self.last_action = np.array([0, 0])
        self.current_action = np.array([0, 0])
        self.normalization = True
        self.seed()
#        self.reset()

    def _observation(self):
        self.wind_d, self.wind_s = self._wind_sensor() # wind direction & speed
        self.gas_measure = self._gas_measure(self.agent_x, self.agent_y)
<<<<<<< HEAD
        etc_state = np.array([float(self.wind_d/math.pi), float(self.wind_s), float(self.dur_t), float(self.agent_x), float(self.agent_y), float(self.last_action[0]), float(self.last_action[1]), float(self.current_action[0]), float(self.current_action[1]), float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc)])
=======
        etc_state = np.array([float(self.wind_d/math.pi), float(self.wind_s), float(self.dur_t), float(self.agent_x), float(self.agent_y), float(self.last_action), float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc)])
>>>>>>> 1211906308dfc7d77a553a01c03d1cd648aa7ea3

        return etc_state

    def _info_function(self, norm_obs, action, done, rew):
        if self.normalization:
<<<<<<< HEAD
#           print(norm_obs)
#           print(self.obs_high_state)
#           print(self.obs_low_state)
           obs = norm_obs*(self.obs_high_state - self.obs_low_state) + self.obs_low_state
           info = "time step:" + str(self.count_actions) + ", act:" + str(
                   round(float(action[0])*180/math.pi,2)) + ", source_probability: " + str(
                   round(float(action[1]),2))+ ", local flow: (" + str(round(obs[0]/math.pi*180,2)) + "degree, " + str(
                   round(obs[1],2)) + "m/s), duration time: " + str(round(obs[2],2)) + ", last pos: (" + str(
                   round(obs[3],2)) + ", " + str(round(obs[4],2)) + "), last action: " + str(
                   round(float(obs[5])*180/math.pi,2)) + ", last prob: " + str(round(obs[6],2)) + ", current action: " + str(
                   round(float(obs[7])*180/math.pi,2)) + ", current prob: " + str(round(obs[8],2)) + ", last conc:" + str(round(obs[9],2)) + ", conc:" + str(
                   round(obs[10],2)) + ", highest conc:" + str(round(obs[11],2)) + ", current pos: (" + str(round(self.agent_x,2)) + "," + str(
                   round(self.agent_y,2)) + "), ", "goal pos: (" + str(round(self.goal_x,2)) + "," + str(round(self.goal_y,2)) + "), done: " + str(done) + ", rew:" + str(rew)
=======
           info = "time step:" + str(self.count_actions) + ", act:" + str(
                   round(float(action[0])*180,2)) + ", source_probability: " + str(
                   round(float(action[1]),2))+ ", local flow: (" + str(round(norm_obs[0]*180,2)) + "degree, " + str(
                   round(norm_obs[1]*20,2)) + "), duration time: " + str(round(norm_obs[2]*self.max_step,2)) + "Current pos: (" + str(
                   round(norm_obs[3]*self.court_lx,2)) + ", " + str(round(norm_obs[4]*self.court_ly,2)) + "), last action: " + str(
                   round(float(norm_obs[5])*180/math.pi,2)) + ", last conc:" + str(round(norm_obs[6]*self.conc_max,2)) + ", conc:" + str(
                   round(norm_obs[7]*self.conc_max,2)) + ", highest conc:" + str(round(norm_obs[8]*self.conc_max,2)) +", particles_x : " + str(
                   np.round(norm_obs[9:9+self.pf_num]*self.court_lx,2)) + ", particles_y : " + str(
                   np.round(norm_obs[9+self.pf_num:9+self.pf_num*2]*self.court_lx,2)) + ", particles_wp : " + str(
                   np.round(norm_obs[9+self.pf_num*2:9+self.pf_num*3],2)) + ", current pos: (" + str(round(self.agent_x,2)) + "," + str(
                   round(self.agent_y,2)) + ")", "goal pos: (" + str(round(self.goal_x,2)) + "," + str(round(self.goal_y,2)) + "), done: " + str(done) + ", rew:" + str(rew)
>>>>>>> 1211906308dfc7d77a553a01c03d1cd648aa7ea3
        else:
           info = "time step:" + str(self.count_actions) + ", act:" + str(
                   round(float(action)*180,2)) + ", local flow: (" + str(round(obs[0],2)) + ", " + str(
                   round(obs[1],2)) + "), last pos: (" + str(round(obs[2],2)*self.court_lx) + ", " + str(
                   round(obs[3],2)*self.court_ly) + "), dur_t from last capture: " + str(round(obs[4],2)) + ", last action: " + str(
                   round(float(obs[5])*180,2)) + ", last conc" + str(round(obs[6],2)) + ", conc" + str(round(obs[7],2)) + ", last highest conc" + str(
                   round(obs[8],2)) + ", rew:" + str(rew) + ", current pos: (" + str(round(self.agent_x,2)) + "," + str(
                   round(self.agent_y,2)) + ")", "goal pos: (" + str(self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)
        return info

<<<<<<< HEAD
    def _reward_goal_reached(self):
        return 1000 #100

    def _reward_failed(self):
        return -10 #-100

    def _reward_goal(self):
        self.eps = 0.05*self.court_lx
        dist = self._distance(self.agent_x, self.agent_y)
        return 1000/pow(max(dist, self.eps)/self.eps, 2)

    def _step_reward(self):
#        if self.gas_measure > self.last_highest_conc*0.5: # need to be adjusted for different source con$
#            reward = 1

        if self.gas_measure > self.last_highest_conc: # need to be adjusted for different source con$
            reward = 1 #+1
            self.dur_t = 0
        else:
            reward = 0 #0
            self.dur_t += 1
        return reward

    def _border_reward(self):
        reward = -10 #-100
        return reward

=======
>>>>>>> 1211906308dfc7d77a553a01c03d1cd648aa7ea3

    def step(self, action):
        prob_thre = 0.8

        self.outborder = False
        self.count_actions += 1
        self._calculate_position(action[0])
<<<<<<< HEAD

        self.current_action = action

        obs = self._observation()

        self.last_action = action
=======
        obs = self._observation()

        self.last_action = action[0]

>>>>>>> 1211906308dfc7d77a553a01c03d1cd648aa7ea3
        rew = 0
        if self.outborder: # Agent get out to search area
            rew += self._border_reward()
        # done for step rewarding
<<<<<<< HEAD
        done = bool(action[1] > prob_thre)
        if not done:
            rew += self._step_reward()
        else: # prob > thre
#            rew = self._reward_goal()
            self.eps = 0.45*self.court_lx
            print("dist: ", self._distance(self.agent_x, self.agent_y), ", threshold: ", self.eps)
            if bool(self._distance(self.agent_x, self.agent_y) <= self.eps): # near by source
                print("\n -----------------close enough----------------")
                rew = self._reward_goal()
#                rew = self._reward_goal_reached()
            else:
                rew = self._reward_failed()

        # break if more than max_step actions taken
        done = bool(self.count_actions >= self.max_step or action[1] > prob_thre or self.outborder)
#        done = bool(self.count_actions >= self.max_step)
=======
        done = bool(agent[1] > 0.5)
        if not done:
            rew += self._step_reward()
        else: # Reach the source
            if bool(self._distance(self.agent_x, self.agent_y) <= self.eps)
                rew = self._reward_goal_reached()

        # break if more than max_step actions taken
        done = bool(self.count_actions >= self.max_step or agent[1] > 0.5 or self.outborder)
>>>>>>> 1211906308dfc7d77a553a01c03d1cd648aa7ea3

        # track, where agent was
        self.positions.append([self.agent_x, self.agent_y])
        self.last_x = self.agent_x
        self.last_y = self.agent_y

        if self.gas_measure > self.last_highest_conc:
            self.last_highest_conc = self.gas_measure

        self.last_measure = self.gas_measure

        if self.normalization:
           norm_obs = []
           norm_obs = self._normalize_observation(obs)
           info = self._info_function(norm_obs, action, done, rew)
           return norm_obs, rew, done, info
        else:
           info = self._info_function(obs, action, done, rew)
           return obs, rew, done, info
