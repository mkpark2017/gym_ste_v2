import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste.envs.basic_ste_env import *

from gym.envs.registration import register


class SteWithoutPoseEnv(BasicSteEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        BasicSteEnv.__init__(self)
        # [local flow velocity (x,y) [m/s] (maximum 100,100), current location (x,y), t-t_last, last action (only direction), last_concentration, concentration (max 100 mg/m^3), last highest conc]
        self.obs_low_state = np.array([-100, -100, 0, -1, 0, 0, 0])
        self.conc_max = 100
        self.obs_high_state = np.array([100, 100, self.max_step,  1, self.conc_max, self.conc_max, self.conc_max])
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)
        self.conc_eps = 0.2 # minimum conc
        self.last_highest_conc = self.conc_eps
        self.normalization = True
        self.max_step = 150
        self.last_conc = 0

        # set a seed and reset the environment
        self.seed()
        self.reset()

    def _wind_sensor(self):
        wind_degree_fluc = 15 #degree
        wind_speed_fluc = 1
        wind_dir = self.np_random.uniform(low=(self.wind_mean_phi-wind_degree_fluc)*math.pi/180, 
                                         high=(self.wind_mean_phi+wind_degree_fluc)*math.pi/180)
        wind_speed = self.np_random.uniform(low=self.wind_mean_speed-wind_speed_fluc, 
                                            high=self.wind_mean_speed+wind_speed_fluc)
        return wind_dir, wind_speed

    # [local flow velocity (x,y) [m/s] (maximum 100,100), current location (x,y), t-t_last, last action (only direction), concentration (max 100 mg/m^3), last highest conc]
    def _observation(self):
        self.wind_d, self.wind_s = self._wind_sensor() # wind direction & speed
        wind_x = math.cos(self.wind_d + math.pi/2)*self.wind_s
        wind_y = math.sin(self.wind_d + math.pi/2)*self.wind_s

        self.gas_measure = self._gas_measure(self.agent_x, self.agent_y)

        return np.array([float(wind_x), float(wind_y), float(self.dur_t), float(self.last_action), float(self.last_conc), float(self.gas_measure), float(self.last_highest_conc)])

    def _normalize_observation(self, obs):
        normalized_obs = []
        for i in range(0, obs.size):
            normalized_obs.append((obs[i]-self.obs_low_state[i])/(self.obs_high_state[i] - self.obs_low_state[i]))
        return np.array(normalized_obs)


    #  extra rewarding reaching the goal and learning to do this by few steps as possible
    def _reward_goal_reached(self):
        return 100

    def _reward_failed(self):
        return -100

    def _gas_measure(self, pos_x, pos_y):
        env_sig = 0.4 #0.4
        sensor_sig_m = 0.2 #0.2;
        conc = self._gas_conc(self.agent_x, self.agent_y)
        conc_env = self.np_random.normal(conc,env_sig)
        while conc_env < 0:
            conc_env = self.np_random.normal(conc,env_sig)
        gas_measure = self.np_random.normal(conc_env, conc_env*sensor_sig_m)
        while gas_measure < 0:
            gas_measure = self.np_random.normal(conc_env, conc_env*sensor_sig_m)

        return gas_measure

    def _step_reward(self):
#        print(self.gas_measure)
#        print(self.last_highest_conc)
        if self.gas_measure > self.last_highest_conc: # need to be adjusted for different source condition
            reward = 1
            self.dur_t = 0
        else:
            reward = 0
            self.dur_t += 1
        return reward

    def _border_reward(self):
        reward = -100
        return reward

    def step(self, action):
        self.outborder = False
        self.count_actions += 1
        self._calculate_position(action)
        obs = self._observation()

        self.last_action = action

        # done for step rewarding
        done = bool(self._distance(self.agent_x, self.agent_y) <= self.eps)
        rew = 0
        if not done:
            rew = self._step_reward()
        else: # Reach the source
            rew = self._reward_goal_reached()
        done = bool(self.count_actions >= self.max_step and self._distance(self.agent_x, self.agent_y) > self.eps)
        if done: # Reach the max_step without finding source
            rew = self._reward_failed()
        if self.outborder: # Agent get out to search area
            rew = self._border_reward()
        # break if more than max_step actions taken
        done = bool(self.count_actions >= self.max_step or self._distance(self.agent_x, self.agent_y) <= self.eps or self.outborder)

        # track, where agent was
        self.positions.append([self.agent_x, self.agent_y])
        self.last_x = self.agent_x
        self.last_y = self.agent_y
        self.last_conc = self.gas_measure

        if self.gas_measure > self.last_highest_conc:
            self.last_highest_conc = self.gas_measure

        norm_obs = []
        if self.normalization:
           norm_obs = self._normalize_observation(obs)
           info = "time step:" + str(self.count_actions) + ", act:" + str(
                   round(float(action)*180,2)) + ", local flow: (" + str(round(norm_obs[0],2)) + ", " + str(
                   round(norm_obs[1],2))  + "), dur_t from last capture: " + str(round(norm_obs[2],2)) + ", last action: " + str(
                   round(float(norm_obs[3])*180,2)) + ", last conc" + str(round(norm_obs[4]*self.conc_max,2)) + ", conc" + str(
                   round(norm_obs[5]*self.conc_max,2)) + ", last highest conc" + str(round(norm_obs[6]*self.conc_max,2)) + ", rew:" + str(
                   rew) + ", current pos: (" + str(round(self.agent_x,2)) + "," + str(round(self.agent_y,2)) + ")", "goal pos: (" + str(
                   self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)
           return norm_obs, rew, done, info
        else:
           info = "time step:" + str(self.count_actions) + ", act:" + str(
                   round(float(action)*180,2)) + ", local flow: (" + str(round(obs[0],2)) + ", " + str(
                   round(obs[1],2))  + "), dur_t from last capture: " + str(round(obs[2],2)) + ", last action: " + str(
                   round(float(obs[3])*180,2)) + ", last conc" + str(round(obs[4],2)) + ", conc" + str(
                   round(obs[5],2)) + ", last highest conc" + str(round(obs[6],2)) + ", rew:" + str(
                   rew) + ", current pos: (" + str(round(self.agent_x,2)) + "," + str(round(self.agent_y,2)) + ")", "goal pos: (" + str(
                   self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)

           return obs, rew, done, info


    def reset(self):
        self.count_actions = 0
        self.positions = []
        # set initial state randomly
        self.agent_x = self.np_random.uniform(low=0, high=self.court_lx)
        self.agent_y = self.np_random.uniform(low=0, high=self.court_ly)
        # self.agent_x = 5
        # self.agent_y = 10
        # self.goal_x = self.np_random.uniform(low=0, high=self.court_lx)
        # self.goal_y = self.np_random.uniform(low=0, high=self.court_lx)
        self.goal_x = 44
        self.goal_y = 44

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
    id='SteWithoutPoseEnv-v0',
    entry_point='gym_ste.envs:SteWithoutPoseEnv',
)

