import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
import pyglet
from matplotlib import cm

#from gym_ste_v2.envs.common.ste_base import *
from gym_ste_v2.envs.common.particle_filter import ParticleFilter
from gym.envs.registration import register

from datetime import datetime
import time

class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self): # necessary to draw somthing
        self.label.draw()

class DrawPatch:
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        r = round(color[0]*255)
        g = round(color[1]*255)
        b = round(color[2]*255)
        self.vertex_list = pyglet.graphics.vertex_list(4, 'v2f', 'c3B')
        self.vertex_list.vertices = [x, y,
                                     x + width, y,
                                     x + width, y + height,
                                     x, y + height]
        self.vertex_list.colors = [r, g, b,
                                   r, g, b,
                                   r, g, b,
                                   r, g, b]
        self.draw_mode = pyglet.gl.GL_QUADS
    def render(self):
        self.vertex_list.draw(self.draw_mode)



class StePFilterBaseEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        #--------------------------Common Env----------------------------
        self.debug = False

        self.last_measure = 0
        self.court_lx = 60              # the size of the environment
        self.court_ly = 60              # the size of the environment
        self.max_step = 300
        self.gmm_num = 0

        # gas sensing
        self.env_sig = 0.4 #0.4
        self.sensor_sig_m = 0.2 #0.2
        self.conc_eps = 0.2 # minimum conc
        self.conc_max = 100

        # rendering
        self.screen_height = 250
        self.screen_width = 250
        self.viewer = None                  # viewer for render()
        self.background_viewer = None       # viewer for background
        self.scale = self.screen_width/self.court_lx
        self.true_conc = np.zeros((self.court_lx, self.court_ly))

        #---------- ------Dummy Env (changed at each Env)-----------------------------
        self.gas_d = 10                 # diffusivity [10m^2/s]
        self.gas_t = 1000               # gas life time [1000sec]
        self.gas_q = 2000               # gas strength [mg/s]
        self.wind_mean_phi = 310        # mean wind direction [degree]
        self.wind_mean_speed = 2

        # agent
        self.agent_x = 0
        self.agent_y = 0
        self.positions = []                 # track agent positions for drawing
        # the goal
        self.goal_x = 0
        self.goal_y = 0

        #--------------------------Initial data-------------------------
        self.dur_t = 0                  # duration time of out of plume
        self.last_x = -1                # last sensing position x
        self.last_y = -1                # last sensing position y
        self.last_action = 0;           # last action
        self.last_highest_conc = 0

        self.gas_measure = -1;
        self.outborder = False          # in search area

        self.total_time = 0.
        self.update_count = 0

        self.CovXxp = 0.
        self.CovXyp = 0.
        self.CovXqp = 0.
        self.warning = False

        self.max_q = 5000

        #-------------------------Particle filter-------------------
        self.pf_num = 1000 #150??
        self.pf_low_state_x = np.zeros(self.pf_num) # particle filter (x1,x2,x3, ...)
        self.pf_low_state_y = np.zeros(self.pf_num) # particle filter (y1,y2,y3, ...)
        pf_low_state_wp = np.zeros(self.pf_num) # particle filter (q1,q2,q3, ...)

        self.pf_high_state_x = np.ones(self.pf_num)*self.court_lx
        self.pf_high_state_y = np.ones(self.pf_num)*self.court_ly
        pf_high_state_wp = np.ones(self.pf_num)

        self.Wps = np.ones(self.pf_num)/self.pf_num
        self.Wpnorms = self.Wps

        self.pf_center = None
        self.est_location = None

        #--------------------------Observation-----------------------
        self.normalization = True

        etc_low_state = np.array([-1, 0, 0, 0, 0, -1, 0, 0, 0]) # [wind_direction, wind speed (m/s), duration time, current pos (x,y), last action (direction), last concentration, concentration, highest conc
        etc_high_state = np.array([1, 20,  self.max_step, self.court_lx, self.court_ly, 1, self.conc_max, self.conc_max, self.conc_max])

        self.obs_low_state = np.concatenate((etc_low_state, self.pf_low_state_x, self.pf_low_state_y, pf_low_state_wp), axis=None)
        self.obs_high_state = np.concatenate((etc_high_state, self.pf_high_state_x, self.pf_high_state_y, pf_high_state_wp), axis=None)
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

        #---------------------------Action--------------------------
        self.delta_t = 1                # 1sec
        self.agent_v = 6                # 2m/s
        self.agent_dist = self.agent_v * self.delta_t
        self.action_angle_low = -1
        self.action_angle_high = 1
        self.action_space = spaces.Box(np.array([self.action_angle_low]), np.array([self.action_angle_high]), dtype=np.float32)

        #--------------------------Ending Criteria--------------------------------
        self.conv_eps = 1.0
        self.eps = 1.0
        self.conc_eps = 0.2 # minimum conc


        seed = self.seed(8201076236150)
        print("Seed: ", seed)
        self.particle_filter = ParticleFilter(self)


    def _distance(self, pose_x, pose_y):
        return math.sqrt(pow((self.goal_x - pose_x), 2) + pow(self.goal_y - pose_y, 2))

    def _wind_sensor(self):
        wind_degree_fluc = 5 #degree
        wind_speed_fluc = 0.1
        wind_dir = self.np_random.uniform(low=(self.wind_mean_phi-wind_degree_fluc)*math.pi/180, 
                                         high=(self.wind_mean_phi+wind_degree_fluc)*math.pi/180)
        # wind_dir [radian]
        wind_speed = self.np_random.uniform(low=self.wind_mean_speed-wind_speed_fluc, 
                                            high=self.wind_mean_speed+wind_speed_fluc)
        return wind_dir, wind_speed

    def _gas_conc(self, pos_x, pos_y): # true gas conectration
        if self.goal_x == pos_x and self.goal_y == pos_y: # to avoid divide by 0
            pos_x += 1e-10
            pos_y += 1e-10
        dist = self._distance(pos_x, pos_y)
        #print("true wind_d: ", self.wind_mean_phi*math.pi/180)
        y_n = -(pos_x - self.goal_x)*math.sin(self.wind_mean_phi*math.pi/180)+ \
                   (pos_y - self.goal_y)*math.cos(self.wind_mean_phi*math.pi/180)
        lambda_plume = math.sqrt(self.gas_d * self.gas_t / (1 + pow(self.wind_mean_speed,2) * self.gas_t/4/self.gas_d) )
        conc = self.gas_q/(4 * math.pi * self.gas_d * dist) * np.exp(-y_n * self.wind_mean_speed/(2*self.gas_d) - dist/lambda_plume) - self.conc_eps

        if conc < 0:
            conc = 0

        return conc

    def _gas_measure(self):
        conc = self._gas_conc(self.agent_x, self.agent_y)
        conc_env = self.np_random.normal(conc,self.env_sig)
        #print(self.sensor_sig_m)
        while conc_env < 0:
            conc_env = self.np_random.normal(conc,self.env_sig)
        gas_measure = self.np_random.normal(conc_env, conc_env*self.sensor_sig_m)
        while gas_measure < 0:
            gas_measure = self.np_random.normal(conc_env, conc_env*self.sensor_sig_m)

        return gas_measure


    def _observation(self):
        self.wind_d, self.wind_s = self._wind_sensor() # wind direction & speed
#        wind_x = math.cos(self.wind_d + math.pi/2)*self.wind_s
#        wind_y = math.sin(self.wind_d + math.pi/2)*self.wind_s
        moved_dist = math.sqrt(pow(self.last_x - self.agent_x,2) + pow(self.last_y - self.agent_y,2))
#        print("------------------------------------------------------")
        self.gas_measure = self._gas_measure()
        self.pf_x, self.pf_y, self.pf_q, self.Wpnorms = self.particle_filter._weight_update(self.gas_measure, self.agent_x, self.agent_y,
                                                                                            self.pf_x, self.pf_y, self.pf_q, self.Wpnorms,
                                                                                            self.wind_d, self.wind_s)

        self.CovXxp = np.var(self.pf_x)
        self.CovXyp = np.var(self.pf_y)
        self.CovXqp = np.var(self.pf_q)
        #x_warning, y_warning = self._boundary_warning_sensor()

        etc_state = np.array([float(self.wind_d/math.pi), float(self.wind_s), float(self.dur_t), float(self.agent_x), float(self.agent_y),
                              float(self.last_action), float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc)])


        return np.concatenate((etc_state, self.pf_x, self.pf_y, self.Wpnorms), axis=None)

    def _normalize_observation(self, obs):
        normalized_obs = []
        for i in range(0, obs.size):
            normalized_obs.append((obs[i]-self.obs_low_state[i])/(self.obs_high_state[i] - self.obs_low_state[i]))
        return np.array(normalized_obs)

    def _calculate_position(self, action):
        angle = (action) * math.pi
        # if angle > 2 * math.pi:
        #    angle -= 2 * math.pi
        # step_size = (action[1] + 1) / 2 * self.max_step_size
        step_size = self.agent_v * self.delta_t
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

    def _reward_goal_reached(self):
        return 100 #100

    def _reward_failed(self):
        return 0 #-100

    def _step_reward(self):
        if self.gas_measure > self.last_highest_conc: # need to be adjusted for different source condition
            reward = 0.1 #+1
            self.dur_t = 0
        else:
            reward = 0
            self.dur_t += 1
        return reward

    def _border_reward(self):
        reward = -1 #-100
        return reward

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

    def _info_function(self, obs, action, done, rew):
        mean_q = np.sum(self.pf_q * self.Wpnorms)

        return np.array([mean_q, self.gas_q])
    '''
    def _info_function(self, obs, action, done, rew):
        if self.normalization:
           obs = obs*(self.obs_low_state - self.obs_high_state) + self.obs_low_state
           info = "time step:" + str(self.count_actions) + ", act:" + str(
                   round(float(action)*180,2)) + ", local flow: (" + str(round(obs[0]*180,2)) + "degree, " + str(
                   round(obs[1],2)) + "), duration time: " + str(round(obs[2],2)) + ", current pos: (" + str(
                   round(obs[3],2)) + ", " + str(round(obs[4],2)) + "), last action: " + str(
                   round(float(obs[5])*180,2)) + ", last conc:" + str(round(obs[6],2)) + ", conc:" + str(
                   round(obs[7],2)) + ", highest conc:" + str(round(obs[8],2)) +", particles_x : " + str(
                   np.round(obs[9:9+self.pf_num],2)) + ", particles_y : " + str(
                   np.round(obs[9+self.pf_num:9+self.pf_num*2],2)) + ", particles_wp : " + str(
                   np.round(obs[9+self.pf_num*2:9+self.pf_num*3],2)) + ", current pos: (" + str(round(self.agent_x,2)) + "," + str(
                   round(self.agent_y,2)) + ")", "goal pos: (" + str(round(self.goal_x,2)) + "," + str(round(self.goal_y,2)) + "), done: " + str(done) + ", rew:" + str(rew)
        else:
           info = "time step:" + str(self.count_actions) + ", act:" + str(
                   round(float(action)*180,2)) + ", local flow: (" + str(round(obs[0],2)) + ", " + str(
                   round(obs[1],2)) + "), last pos: (" + str(round(obs[2],2)) + ", " + str(
                   round(obs[3],2)) + "), dur_t from last capture: " + str(round(obs[4],2)) + ", last action: " + str(
                   round(float(obs[5]),2)) + ", last conc" + str(round(obs[6],2)) + ", conc" + str(round(obs[7],2)) + ", last highest conc" + str(
                   round(obs[8],2)) + ", rew:" + str(rew) + ", current pos: (" + str(round(self.agent_x,2)) + "," + str(
                   round(self.agent_y,2)) + ")", "goal pos: (" + str(self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)
        info = "Not show data"

        return info
    '''

    def step(self, action):
        self.outborder = False
        self.count_actions += 1
        self._calculate_position(action)
        obs = self._observation()

        self.last_action = action

        rew = 0
        if self.outborder: # Agent get out to search area
            rew += self._border_reward()
        # done for step rewarding
        done = bool(self._distance(self.agent_x, self.agent_y) <= self.eps)
        if not done:
            rew += self._step_reward()
        else: # Reach the source
            rew = self._reward_goal_reached()
#        done = bool(self.count_actions >= self.max_step and self._distance(self.agent_x, self.agent_y) > self.eps)
#        if done: # Reach the max_step without finding source
#            rew = self._reward_failed()
        # break if more than max_step actions taken
        done = bool(self.count_actions >= self.max_step or self._distance(self.agent_x, self.agent_y) <= self.eps or self.outborder)
#        done = bool(self.count_actions >= self.max_step or self._distance(self.agent_x, self.agent_y) <= self.eps)

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
            print("Reset")
            self.reset()
        self.positions.append([self.agent_x, self.agent_y])
        if self.debug:
            print("x/y  - x/y", self.agent_x, self.agent_y, self.goal_x, self.goal_y)
            print("scale x/y  - x/y", self.agent_x*self.scale, self.agent_y*self.scale, self.goal_x*self.scale, self.goal_y*self.scale)


        self.est_location = np.array([0.5, 0.5])


        obs = self._observation()
#        print("==============================================")
        if self.normalization:
            return self._normalize_observation(obs)
        else:
            return obs

    def render_background(self, mode='human'):
        if mode == 'human':
            from gym.envs.classic_control import rendering
            if self.background_viewer is None:
                self.background_viewer = rendering.Viewer(self.screen_width, self.screen_height)
            max_conc = 100;
            width = 1*self.scale
            height = 1*self.scale
#            conc_mat = []

            for xx in range(self.court_lx):
#                conc_mat_temp = []
                for yy in range(self.court_ly):
                    conc = self._gas_conc(xx+0.5, yy+0.5)
                    conc = self.np_random.normal(conc, math.sqrt(pow(self.env_sig,2) + pow(conc*self.sensor_sig_m,2)) )
                    while conc < 0:
                        conc = self.np_random.normal(conc, math.sqrt(pow(self.env_sig,2) + pow(conc*self.sensor_sig_m,2)) )

                    x = xx*self.scale
                    y = yy*self.scale
                    if conc > max_conc: #set maximum value for visualization
                        conc = max_conc
                        color = cm.jet(255) # 255 is maximum number
                        self.background_viewer.add_geom(DrawPatch(x, y, width, height, color))
                    elif conc > self.conc_eps: #just for plot (_gas_conc already includes conc_eps)
                        color_cal = round( (math.exp(math.log(conc+1)/math.log(max_conc+1))-1) * 255)
#                        color_cal = conc/max_conc * 255
                        if color_cal < 0: color_cal = 0
                        color = cm.jet(color_cal)
#                        color = cm.jet(round((conc+1)/(max_conc+1)*255) )
                        self.background_viewer.add_geom(DrawPatch(x, y, width, height, color))
#                    conc_mat_temp.append(round(conc,2))
#                conc_mat.append(conc_mat_temp)
#            print(conc_mat)
            return self.background_viewer.render(return_rgb_array=mode == 'rgb_array')


    def render(self, mode='human'):
        if mode == 'ansi':
            return self._observation()
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

#            measure = rendering.make_circle(self.gas_measure)
#            measure.add_attr(self.agent_trans)
#            measure.set_color(255, 0, 0)
#            self.viewer.add_geom(measure)

            #track the way, the agent has gone
            track_way = rendering.make_polyline(np.dot(self.positions, self.scale))
            track_way.set_linewidth(4)
            self.viewer.add_onetime(track_way)

            # draw the agent
            agent = rendering.make_circle(5)
            agent_trans = rendering.Transform()
            agent.add_attr(agent_trans)
            agent.set_color(0, 0, 1)
            self.viewer.add_onetime(agent)

            agent_trans.set_translation(self.agent_x * self.scale, self.agent_y * self.scale)

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


#            for i in range(0,self.pf_num):
#                particle = rendering.make_circle(3)
#                particle.add_attr(rendering.Transform(translation=(self.pf_x[i]*self.scale, self.pf_y[i]*self.scale)))
#                particle.set_color(0,255,0)
#                self.viewer.add_onetime(particle)

            if self.gmm_num > 0:
                for i in range(0,self.gmm_num):
                    for j in range(0, np.shape(self.gmm_data[i])[0]):
                        particle = rendering.make_circle(3)
                        particle.add_attr(rendering.Transform(translation=(self.gmm_data[i][j][0]*self.scale,
                                                                           self.gmm_data[i][j][1]*self.scale)))
                        particle.set_color(i*.5, .3*i, .1*i)
                        self.viewer.add_onetime(particle)

#                    gmm_particle = rendering.make_circle(5)
                    gmm_particle = rendering.make_capsule(8, 4)
                    gmm_particle.add_attr(rendering.Transform(translation=(self.gmm_mean_x[i]*self.scale, self.gmm_mean_y[i]*self.scale)))
#                    gmm_particle.set_color(i*.5-0.2, .3*i-0.2, .1*i-0.2)
                    gmm_particle.set_color(0,1,0)
                    self.viewer.add_onetime(gmm_particle)

            else:
                for i in range(0,self.pf_num):
                    particle = rendering.make_circle(3)
                    particle.add_attr(rendering.Transform(translation=(self.pf_x[i]*self.scale, self.pf_y[i]*self.scale)))
                    particle.set_color(0,255,0)
                    self.viewer.add_onetime(particle)

            if not np.all(self.pf_center) == None:
                pf_center = rendering.make_circle(5)
                pf_center.add_attr(rendering.Transform(translation=(self.pf_center[0]*self.scale, self.pf_center[1]*self.scale)))
                pf_center.set_color(0.5, .3, .1)
                self.viewer.add_onetime(pf_center)

            if not np.all(self.est_location) == None:
                est_location = rendering.make_capsule(8, 4)
                est_location.add_attr(rendering.Transform(translation=(self.est_location[0]*self.scale, self.est_location[1]*self.scale)))
                est_location.set_color(0.1, .3, .5)
                self.viewer.add_onetime(est_location) 


#            text = 'This is a test but it is not visible'
#            label = pyglet.text.Label(text, font_size=36,
#                                      x=10, y=10, anchor_x='left', anchor_y='bottom',
#                                      color=(255, 123, 255, 255))
#            label.draw()
#            self.viewer.add_geom(DrawText(label))

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.background_viewer:
            self.background_viewer.close()
            self.background_viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

