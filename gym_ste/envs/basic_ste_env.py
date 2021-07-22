import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
import pyglet
from matplotlib import cm

from gym.envs.registration import register

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

class BasicSteEnv(gym.Env):
    # this is a list of supported rendering modes!
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        self.debug = False
        # define the environment and the observations
        self.court_lx = 55		# the size of the environment
        self.court_ly = 55		# the size of the environment
        self.max_step = 150
        self.delta_t = 1		# 1sec
        self.agent_v = 3		# 3m/s
        self.gas_d = 10 		# diffusivity [10m^2/s]
        self.gas_t = 1000 		# gas life time [1000sec]
        self.gas_q = 2000		# gas strength
        self.wind_mean_phi = 310        # mean wind direction [degree]
        self.wind_mean_speed = 2
        self.dur_t = 0			# duration time of out of plume
        self.last_x = -1		# last sensing position x
        self.last_y = -1		# last sensing position y
        self.last_action = 0;		# last action
        self.gas_measure = -1;

        # not normalize [local flow velocity (x,y) [m/s], last sampling location (x,y), t-t_last, last action (only direction)
        self.obs_low_state = np.array([-np.inf, -np.inf, 0, 0,  0, -1, 0])
        self.obs_high_state = np.array([np.inf,	np.inf,	self.court_lx,	self.court_ly,	self.max_step,	1, np.inf])
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

        # Action space should be bounded between (-1,1)
        # self.max_step_size = 10
        # action space: change direction in rad (discrete), run into this direction (Box)
        self.action_angle_low = -1
        self.action_angle_high = 1
        # self.action_step_low = -1
        # self.action_step_high = 1
        # self.action_space = spaces.Box(np.array([self.action_angle_low, self.action_step_low]), np.array([self.action_angle_high, self.action_step_high]), dtype=np.float32)
        self.action_space = spaces.Box(np.array([self.action_angle_low]), np.array([self.action_angle_high]), dtype=np.float32)

        self.count_actions = 0  # count actions for rewarding
        self.eps = 0.1*self.court_lx  # distance to goal, that has to be reached to solve env
        self.np_random = None  # random generator

        # agent
        self.agent_x = 0
        self.agent_y = 0
        self.positions = []                 # track agent positions for drawing

        # the goal
        self.goal_x = 0
        self.goal_y = 0

        # rendering
        self.screen_height = 600
        self.screen_width = 600
        self.viewer = None                  # viewer for render()
        self.background_viewer = None       # viewer for background
        self.agent_trans = None             # Transform-object of the moving agent
        self.track_way = None               # polyline object to draw the tracked way
        self.scale = self.screen_width/self.court_lx
        self.true_conc = np.zeros((self.court_lx, self.court_ly))

        # set a seed and reset the environment
        self.seed()
        self.reset()

    def _distance(self, pose_x, pose_y):
        return math.sqrt(pow((self.goal_x - pose_x), 2) + pow(self.goal_y - pose_y, 2))

    def _wind_sensor(self):
        wind_degree_fluc = 1.5 #degree
        wind_speed_fluc = 0.1
        wind_dir = self.np_random.uniform(low=(self.wind_mean_phi-wind_degree_fluc)*math.pi/180, 
                                         high=(self.wind_mean_phi+wind_degree_fluc)*math.pi/180)
        wind_speed = self.np_random.uniform(low=self.wind_mean_speed-wind_speed_fluc, 
                                            high=self.wind_mean_speed+wind_speed_fluc)
        return wind_dir, wind_speed

    #  extra rewarding reaching the goal and learning to do this by few steps as possible
    def _reward_goal_reached(self):
        return 100

    def _reward_failed(self):
        return -100

    def _gas_conc(self, pos_x, pos_y):
        if self.goal_x == pos_x and self.goal_y == pos_y: # to avoid divide by 0
            pos_x += 1e-3
            pos_y += 1e-3
        dist = self._distance(pos_x, pos_y)
        y_n = -(pos_x - self.goal_x)*math.sin(self.wind_d) + (pos_y - self.goal_y)*math.cos(self.wind_d)
        lambda_plume = math.sqrt(self.gas_d * self.gas_t / (1 + pow(self.wind_s,2) * self.gas_t/4/self.gas_d) )
        conc = self.gas_q/(4 * math.pi * self.gas_d * dist) * np.exp(-y_n * self.wind_s/(2*self.gas_d) - dist/lambda_plume)
        return conc

    def _gas_measure(self, pos_x, pos_y):
        env_sig = 0. #0.4
        sensor_sig_m = 0. #0.2;
        conc = self._gas_conc(self.agent_x, self.agent_y)
        conc_env = self.np_random.normal(conc,env_sig)
        while conc_env < 0:
            conc_env = self.np_random.normal(conc,env_sig)
        gas_measure = self.np_random.normal(conc_env, conc_env*sensor_sig_m)
        while gas_measure < 0:
            gas_measure = self.np_random.normal(conc_env, conc_env*sensor_sig_m)

        return gas_measure

    def _step_reward(self):
#        self.gas_measure = self._gas_measure(self.agent_x, self.agent_y)
#        print(gas_measure)
        if self.gas_measure > 0.3: # need to be adjusted for different source condition
            reward = 1
            self.dur_t = 1
        else:
            reward = 0
            self.dur_t += 1
        return reward

    # not normalize [local flow velocity (x,y) [m/s], last sampling location (x,y), t-t_last, last action (only direction)
    def _observation(self):
        self.wind_d, self.wind_s = self._wind_sensor() # wind direction & speed
        wind_x = math.cos(self.wind_d + math.pi/2)*self.wind_s
#        print(len(wind_x))
        wind_y = math.sin(self.wind_d + math.pi/2)*self.wind_s
#        print(str(self.last_action) + "\n")
#        print("OBS   wind_x:" + str(wind_x) + "wind_y:" + str(wind_y) + "last_x:" + str(self.last_x) )
        self.last_x = self.agent_x
        self.last_y = self.agent_y

        self.gas_measure = self._gas_measure(self.agent_x, self.agent_y)

        return np.array([float(wind_x), float(wind_y), float(self.last_x), float(self.last_y), float(self.dur_t), float(self.last_action), float(self.gas_measure)])

#    def _normalize_observation(self, obs):
#        normalized_obs = []
#        for i in range(0, 4):
#            normalized_obs.append(obs[i]/255*2-1)
#        normalized_obs.append(obs[-1]/360.62)
#        return np.array(normalized_obs)

    def _calculate_position(self, action):
        angle = (action) * math.pi
        if angle > 2 * math.pi:
            angle -= 2 * math.pi
        # step_size = (action[1] + 1) / 2 * self.max_step_size
        step_size = self.agent_v * self.delta_t
        # calculate new agent state
        self.agent_x = self.agent_x + math.cos(angle) * step_size
        self.agent_y = self.agent_y + math.sin(angle) * step_size

        # borders
        if self.agent_x < 0:
            self.agent_x = 0
        if self.agent_x > self.court_lx:
            self.agent_x = self.court_lx
        if self.agent_y < 0:
            self.agent_y = 0
        if self.agent_y > self.court_ly:
            self.agent_y = self.court_ly

    def step(self, action):
        self.count_actions += 1
        self._calculate_position(action)
        # calulate new observation
        obs = self._observation()

        self.last_action = action

        # done for step rewarding
        done = bool(self._distance(self.agent_x, self.agent_y) <= self.eps)
        rew = 0
        if not done:
            rew += self._step_reward()
        else: # Reach the source
            rew += self._reward_goal_reached()

        done = bool(self.count_actions >= self.max_step and self._distance(self.agent_x, self.agent_y) > self.eps)
        if done: # Reach the max_step without finding source
            rew += self._reward_failed()

        # break if more than max_step actions taken
        done = bool(self.count_actions >= self.max_step or self._distance(self.agent_x, self.agent_y) <= self.eps)

        # track, where agent was
        self.positions.append([self.agent_x, self.agent_y])
        self.last_x = self.agent_x
        self.last_y = self.agent_y

        # normalized_obs = self._normalize_observation(obs)

        info = "time step:" + str(self.count_actions) + ", act:" + str(
               round(float(action)*180,2)) + ", local flow: (" + str(round(obs[0],2)) + ", " + str(
               round(obs[1],2)) + "), last pos: (" + str(round(obs[2],2)) + ", " + str(
               round(obs[3],2)) + "), dur_t from last capture: " + str(round(obs[4],2)) + ", last action: " + str(
               round(float(obs[5])*180,2)) + ", rew:" + str(rew) + ", current pos: (" + str(round(self.agent_x,2)) + "," + str(
               round(self.agent_y,2)) + ")", "goal pos: (" + str(self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)

        # return normalized_obs, rew, done, info
        return obs, rew, done, info

    def reset(self):
        self.count_actions = 0
        self.positions = []
        # set initial state randomly
        # self.agent_x = self.np_random.uniform(low=0, high=self.court_lx)
        # self.agent_y = self.np_random.uniform(low=0, high=self.court_ly)
        self.agent_x = 5
        self.agent_y = 10
        # self.goal_x = self.np_random.uniform(low=0, high=self.court_lx)
        # self.goal_y = self.np_random.uniform(low=0, high=self.court_lx)
        self.goal_x = 45.5
        self.goal_y = 45.5

        # self.gas_d = self.np_random.uniform(low=0, high=20)                # diffusivity [10m^2/s]
        # self.gas_t = self.np_random.uniform(low=500, high=1500)            # gas life time [1000sec]
        # self.gas_q = self.np_random.uniform(low=1500, high=2500)           # gas strength
        # self.wind_mean_phi = self.np_random.uniform(low=0, high=360)        # mean wind direction
        self.gas_d = 10                 # diffusivity [10m^2/s]
        self.gas_t = 1000               # gas life time [1000sec]
        self.gas_q = 2000               # gas strength
        self.wind_mean_phi = 310        # mean wind direction [degree]


        if self.goal_y == self.agent_y or self.goal_x == self.agent_x:
            self.reset()
        self.positions.append([self.agent_x, self.agent_y])
        if self.debug:
            print("x/y  - x/y", self.agent_x, self.agent_y, self.goal_x, self.goal_y)
            print("scale x/y  - x/y", self.agent_x*self.scale, self.agent_y*self.scale, self.goal_x*self.scale, self.goal_y*self.scale)

        obs = self._observation()
        return obs

    def render_background(self, mode='human'):
        if mode == 'human':
            from gym.envs.classic_control import rendering
            if self.background_viewer is None:
                self.background_viewer = rendering.Viewer(self.screen_width, self.screen_height)
            max_conc = 100;
            width = 1*self.scale
            height = 1*self.scale
            for xx in range(self.court_lx - 1):
                for yy in range(self.court_ly - 1):
                    conc = self._gas_conc(xx+0.5, yy+0.5)
                    x = xx*self.scale
                    y = yy*self.scale
                    if conc > max_conc: #set maximum value for visualization
                        conc = max_conc
                        color = cm.jet(255) # 255 is maximum number
                        self.background_viewer.add_geom(DrawPatch(x, y, width, height, color))
                    elif conc > 0.1:
                        color = cm.jet(round(math.log(conc+1)/math.log(max_conc+1)*255))
                        self.background_viewer.add_geom(DrawPatch(x, y, width, height, color))

            return self.background_viewer.render(return_rgb_array=mode == 'rgb_array')


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
            self.viewer.add_geom(self.track_way)

            # draw the agent
            car = rendering.make_circle(5)
            self.agent_trans = rendering.Transform()
            car.add_attr(self.agent_trans)
            car.set_color(0, 0, 255)
            self.viewer.add_geom(car)

            self.agent_trans.set_translation(self.agent_x * self.scale, self.agent_y * self.scale)

            goal = rendering.make_circle(5)
            goal.add_attr(rendering.Transform(translation=(self.goal_x*self.scale, self.goal_y*self.scale)))
            goal.set_color(255, 0, 0)
            self.viewer.add_geom(goal)


#            text = 'This is a test but it is not visible'
#            label = pyglet.text.Label(text, font_size=36,
#                                      x=10, y=10, anchor_x='left', anchor_y='bottom',
#                                      color=(255, 123, 255, 255))
#            label.draw()
#            self.viewer.add_geom(DrawText(label))

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        # elif mode == "rgb_array":
        #     super(Nav2dEnv, self).render(mode=mode)
        # else:
        #     super(Nav2dEnv, self).render(mode=mode)

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

register(
    id='BasicSteEnv-v0',
    entry_point='gym_ste.envs:BasicSteEnv',
)
