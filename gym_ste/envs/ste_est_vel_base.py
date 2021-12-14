import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste.envs.ste_base import *

from gym.envs.registration import register

from datetime import datetime


class SteEstVelBaseEnv(SteBaseEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        SteBaseEnv.__init__(self)

        self.min_vel = 1
        self.max_vel = 10
        self.agent_min_dist = max(2,self.min_vel) * self.delta_t

        self.obs_low_state = np.array([-1,  0,            #wind sensor measure (dircetion, vel)
                                        0,                #durtion time
                                        0,  0,            #current pos
                                       -1,  0,  0,        #last action (direction, vel, existence_porb)
                                        0,  0,  0])       #last conc, conc, highest conc
        self.obs_high_state = np.array([1, 20,
                                        self.max_step,
                                        self.court_lx, self.court_ly,
                                        1, 1, 1,
                                        self.conc_max, self.conc_max, self.conc_max])
        
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)


        self.action_low = np.array([-1, 0, 0]) # angle, velocity, prob
        self.action_high = np.array([1, 1, 1])

        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)

        self.screen_width = 300
        self.screen_height = 300
        self.scale = self.screen_width/self.court_lx


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
        #moved_dist = math.sqrt(pow(self.last_x - self.agent_x,2) + pow(self.last_y - self.agent_y,2))
        self.gas_measure = self._gas_measure()
        #print(self.last_action)

        obs = np.array([float(self.wind_d/math.pi), float(self.wind_s),
                        float(self.dur_t),
                        float(self.agent_x), float(self.agent_y),
                        float(self.last_action[0]), float(self.last_action[1]), float(self.last_action[2]),
                        float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc)])

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
                   round(obs[8],2)) + ", highest conc:" + str(round(obs[9],2))  + ", current pos: (" + str(round(self.agent_x,2)) + "," + str(
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

        self.dur_t = 0
        self.last_action = np.array([0, 0, 0])
        self.last_measure = 0
        self.last_highest_conc = 0


    def _reward_goal(self):
        self.eps = 1
        dist = self._distance(self.agent_x, self.agent_y)
        return 1000/pow(max(dist, self.eps)/self.eps, 2)

    def _reward_failed(self):
        reward = -10
        return reward

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

        self.last_action = action

        #x_warning, y_warning = self._boundary_warning_sensor()

        # done for step rewarding
        prob_thre = 0.9
        flag_done = bool(action[2] > prob_thre)

        rew = 0
        if self.outborder: # Agent get out to search area
            rew += self._border_reward()
        if not flag_done: # Step rewarding
            rew += self._step_reward()
        else: # prob > thre
#            rew = self._reward_goal()
            self.eps = 0.4*self.court_lx
            print("dist: ", self._distance(self.agent_x, self.agent_y), ", threshold: ", self.eps)
            if bool(self._distance(self.agent_x, self.agent_y) <= self.eps): # near by source
                print("\n -----------------close enough----------------")
                rew = self._reward_goal()
#                rew = self._reward_goal_reached()
            else:
                rew = self._reward_failed()

        # break if more than max_step actions taken
        done = bool(self.count_actions >= self.max_step or action[2] > prob_thre or self.outborder)

        # break if more than max_step actions taken
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
        self.count_actions = 0
        self.positions = []
        self.measures = []
        self.agent_xs = []
        self.agent_ys = []
        # set initial state randomly
        self._set_init_state()

        if math.sqrt(pow(self.goal_y - self.agent_y,2) + pow(self.goal_x - self.agent_x,2) ) < self.court_lx*0.3:
            print("Reset")
            self.reset()
        self.positions.append([self.agent_x, self.agent_y])
        if self.debug:
            print("x/y  - x/y", self.agent_x, self.agent_y, self.goal_x, self.goal_y)
            print("scale x/y  - x/y", self.agent_x*self.scale, self.agent_y*self.scale, self.goal_x*self.scale, self.goal_y*self.scale)

        obs = self._observation()
#        print("==============================================")
        if self.normalization:
            return self._normalize_observation(obs)
        else:
            return obs

    def render(self, mode='human'):
        if mode == 'ansi':
            return self._observation()
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            #track the way, the agent has gone
            self.track_way = rendering.make_polyline(np.dot(self.positions, self.scale))
            self.track_way.set_linewidth(4)
            self.viewer.add_onetime(self.track_way)

            # draw the agent
            agent = rendering.make_circle(5)
            self.agent_trans = rendering.Transform()
            agent.add_attr(self.agent_trans)
            agent.set_color(0, 0, 255)
            self.viewer.add_onetime(agent)

            self.agent_trans.set_translation(self.agent_x * self.scale, self.agent_y * self.scale)

            goal = rendering.make_circle(5)
            goal.add_attr(rendering.Transform(translation=(self.goal_x*self.scale, self.goal_y*self.scale)))
            goal.set_color(0, 0, 0)
            self.viewer.add_onetime(goal)

#            print(self.measures)
            for i in range(len(self.measures)):
                measure = rendering.make_circle(math.pow(self.measures[i],1/3)*3)
                measure.add_attr(rendering.Transform(translation=(self.agent_xs[i]*self.scale, self.agent_ys[i]*self.scale)))
#                measure.add_attr(self.agent_trans)
                measure.set_color(255, 0, 0)
#                self.viewer.add_geom(measure)
                self.viewer.add_onetime(measure)


            return self.viewer.render(return_rgb_array=mode == 'rgb_array')

