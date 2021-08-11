import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste.envs.basic_ste_env import *

from gym.envs.registration import register

from datetime import datetime


class StePFilterBaseEnv3(BasicSteEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        BasicSteEnv.__init__(self)

        self.agent_dist = self.agent_v * self.delta_t

        self.last_measure = 0
        self.court_lx = 60              # the size of the environment
        self.court_ly = 60              # the size of the environment
        self.max_step = 1000

        self.pf_num = 20
        self.pf_low_state_x = np.zeros(self.pf_num) # particle filter (x1,x2,x3, ...)
        self.pf_low_state_y = np.zeros(self.pf_num) # particle filter (y1,y2,y3, ...)
        pf_low_state_wp = np.zeros(self.pf_num) # particle filter (q1,q2,q3, ...)
#        etc_low_state = np.array([-1, -20, 0, -1, 0, 0, 0]) # [wind_direction, wind speed (m/s), last action (only direction), last concentration, concentration (max 100 mg/m^3), last highest conc]
#        etc_low_state = np.array([-1, 0, -1, -1, -1, 0, 0, 0]) # [wind_direction, wind speed (m/s), boundary warning (x,y), last action (only direction), last concentration, concentration (max 100 mg/m^3)]
#        etc_low_state = np.array([-1, 0, -1, 0, 0, 0]) # [wind_direction, wind speed (m/s), moved distance, last action (only direction), last concentration, highest concentration]
#        etc_low_state = np.array([-1, 0, 0, -1, -1, -1, 0, 0, 0]) # [wind_direction, wind speed (m/s), duration time, boundary (x,y), last action (only direction), last concentration, concentration (max 100 mg/m^3), highest conc]
#        etc_low_state = np.array([-1, 0, 0, -1, 0, 0, 0]) # [wind_direction, wind speed (m/s), duration time, last action (only direction), last concentration, concentration (max 100 mg/m^3), highest conc]
        etc_low_state = np.array([-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0])
        # [wind_direction, wind speed (m/s), duration time, current location, last action (only direction), last concentration, concentration (max 100 mg/m^3), highest concentration, weighted sum (x, y), variance (xx, yy)])

        self.obs_low_state = etc_low_state
#        self.obs_low_state = np.concatenate((etc_low_state, self.pf_low_state_x, self.pf_low_state_y, pf_low_state_wp), axis=None)

        self.conc_max = 100
        self.pf_high_state_x = np.ones(self.pf_num)*self.court_lx
        self.pf_high_state_y = np.ones(self.pf_num)*self.court_ly
        pf_high_state_wp = np.ones(self.pf_num)
#        etc_high_state = np.array([1, 20,  self.max_step,  1, self.conc_max, self.conc_max, self.conc_max])
        etc_high_state = np.array([1, 20,  self.max_step, self.court_lx, self.court_ly, 1, self.conc_max, self.conc_max, self.conc_max, self.court_lx, self.court_ly, pow(self.court_lx,2), pow(self.court_ly,2)])
#        etc_high_state = np.array([1, 20,  self.max_step, 1, 1, 1, self.conc_max, self.conc_max, self.conc_max])

        self.obs_high_state = etc_high_state
#        self.obs_high_state = np.concatenate((etc_high_state, self.pf_high_state_x, self.pf_high_state_y, pf_high_state_wp), axis=None)
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

        self.Wps = np.ones(self.pf_num)/self.pf_num
        self.Wpnorms = self.Wps

        self.eps = 0.05*self.court_lx
        self.conc_eps = 0.2 # minimum conc
        self.last_highest_conc = self.conc_eps
        self.normalization = True

        self.env_sig = 0.1 #0.05
        self.sensor_sig_m = 0.05 #0.02;

        self.screen_width = 300
        self.screen_height = 300
        self.scale = self.screen_width/self.court_lx

        self.total_time = 0.

        self.CovXxp = 0.
        self.CovXyp = 0.
        self.warning = False
        # set a seed and reset the environment
        self.seed()
#        self.reset()

    def _wind_sensor(self):
        wind_degree_fluc = 5 #degree
        wind_speed_fluc = 0.1
        wind_dir = self.np_random.uniform(low=(self.wind_mean_phi-wind_degree_fluc)*math.pi/180, 
                                         high=(self.wind_mean_phi+wind_degree_fluc)*math.pi/180)
        wind_speed = self.np_random.uniform(low=self.wind_mean_speed-wind_speed_fluc, 
                                            high=self.wind_mean_speed+wind_speed_fluc)
        return wind_dir, wind_speed

    def _pf_gas_conc(self, pos_x, pos_y, source_x, source_y, source_q): # true gas conectration
        avoid_zero = (np.sqrt(pow(source_x - pos_x,2) + pow(source_y - pos_y,2) ) < 1e-10) 
        source_x[avoid_zero] += 1e-10
        source_y[avoid_zero] += 1e-10

        dist = np.sqrt(pow((source_x - pos_x), 2) + pow(source_y - pos_y, 2))
        y_n = -(pos_x - source_x)*np.sin(self.wind_d*math.pi/180)+ \
               (pos_y - source_y)*math.cos(self.wind_d*math.pi/180)
        lambda_plume = math.sqrt(self.gas_d * self.gas_t / (1 + pow(self.wind_s,2) * self.gas_t/4/self.gas_d) )
        conc = source_q/(4 * math.pi * self.gas_d * dist) * np.exp(-y_n * self.wind_mean_speed/(2*self.gas_d) - dist/lambda_plume)
        return conc

    def _weight_calculate(self, pf_x, pf_y, pf_q):
        pf_conc = self._pf_gas_conc(self.agent_x, self.agent_y, pf_x, pf_y, pf_q)
        pdetSig = math.sqrt( pow((self.gas_measure*self.sensor_sig_m),2) + pow(self.env_sig,2) )
        if pdetSig < 1e-10: pdetSig = 1e-10
        pdetSig_sq = pow(pdetSig, 2)
        gauss_val = (self.gas_measure - pf_conc)/pdetSig
        gauss_new = 1/(math.sqrt(2*math.pi)*pdetSig_sq)*np.exp(-pow(gauss_val,2)/2)
        gauss_new[gauss_new < 1e-20] = 1e-20
        
        return gauss_new
        
    def _particle_resample(self, gauss_new):
            N = self.Wpnorms.size
            M = N
            indx = np.ones(N)*-1
            Q = np.cumsum(self.Wpnorms)
            indx = np.zeros(N)
            T = np.arange(N)/N + self.np_random.uniform(low=np.zeros(N), high=np.ones(N)/N)
            i=1
            j=1
            while(i<N and j<M):
                while(Q[j] < T[i]):
                    j = j+1
                indx[i]=j
                i=i+1

            indx = np.int64(indx)
            
            self.pf_x = self.pf_x[indx]
            self.pf_y = self.pf_y[indx]
            self.pf_q = self.pf_q[indx]

            mm = 2
            A=pow(4/(mm+2), 1/(mm+4) )
            cx = 4*math.pi/3
            hopt = A*pow(A,-1/(mm+4))
            for _ in range(3):
                CovXxp = np.var(self.pf_x)
                CovXyp = np.var(self.pf_y)
                CovXqp = np.var(self.pf_q)

                dkXxp = math.sqrt(CovXxp)+0.5
                dkXyp = math.sqrt(CovXyp)+0.5
                dkXqp = math.sqrt(CovXqp)+0.5

                nXxp = self.pf_x + (hopt*dkXxp*np.random.normal(0,1,self.pf_num) )
                nXxp[nXxp>self.court_lx] = self.court_lx # out of area
                nXxp[nXxp<0] = 0 # out of area
                nXyp = self.pf_y + (hopt*dkXyp*np.random.normal(0,1,self.pf_num) )
                nXyp[nXyp>self.court_ly] = self.court_ly # out of area
                nXyp[nXyp<0] = 0 # out of area
                nXqp = self.pf_q + (hopt*dkXqp*np.random.normal(0,1,self.pf_num) )
                nXqp[nXqp<0] = 0 # out of range

                n_new = self._weight_calculate(nXxp, nXyp, nXqp)
                alpha = n_new/gauss_new[indx]
                mcrand = np.random.uniform(0,1,self.pf_num)
#                print(alpha > mcrand)
                new_point_bool = alpha > mcrand
                self.pf_x[new_point_bool] = nXxp[new_point_bool]
                self.pf_y[new_point_bool] = nXyp[new_point_bool]
                self.pf_q[new_point_bool] = nXqp[new_point_bool]
            self.Wpnorms = np.ones(self.pf_num)/self.pf_num
            

    def _particle_filter(self):
        Wp_sum = 0
        resample_true = False
        gauss_new = self._weight_calculate(self.pf_x, self.pf_y, self.pf_q)
        if (gauss_new == np.ones(self.pf_num)/self.pf_num).all(): resample_true = True
        self.Wps = self.Wpnorms * gauss_new
        Wp_sum = np.sum(self.Wps)

        if Wp_sum==0:
            Wp_sum = self.pf_num
            self.Wps = np.ones(self.pf_num)
  #          print(Wp_sum)
 #           print(self.Wps)
        self.Wpnorms = self.Wps/Wp_sum

        if 1/sum(pow(self.Wpnorms,2)) < self.pf_num*0.5 or resample_true: # 1 for every time
            self._particle_resample(gauss_new)
            self.CovXxp = np.var(self.pf_x)
            self.CovXyp = np.var(self.pf_y)

    def _boundary_warning_sensor(self):
        dist_x = self.court_lx - self.agent_x
#        if dist_x == 0: dist_x = 1e-2
        dist_y = self.court_ly - self.agent_y
#        if dist_y == 0: dist_y = 1e-2
        x_warning = 0
        y_warning = 0
        if dist_x < self.agent_dist*0.99:
            x_warning = 1
            self.warning = True
        if dist_y < self.agent_dist*0.99:
            y_warning = 1
            self.warning = True
        dist_x = 0 - self.agent_x
#        if dist_x == 0: dist_x = -1e-2
        dist_y = 0 - self.agent_y
#        if dist_y == 0: dist_y = -1e-2
        if abs(dist_x) < self.agent_dist*0.99:
            x_warning = -1
            self.warning = True
        if abs(dist_y) < self.agent_dist*0.99:
            y_warning = -1
            self.warning = True
        
        return x_warning, y_warning

    def _observation(self):
        self.wind_d, self.wind_s = self._wind_sensor() # wind direction & speed
#        wind_x = math.cos(self.wind_d + math.pi/2)*self.wind_s
#        wind_y = math.sin(self.wind_d + math.pi/2)*self.wind_s
        moved_dist = math.sqrt(pow(self.last_x - self.agent_x,2) + pow(self.last_y - self.agent_y,2))
#        print("------------------------------------------------------")
        self.gas_measure = self._gas_measure(self.agent_x, self.agent_y)
        self._particle_filter()
        weighted_x = sum(self.pf_x*self.Wpnorms)
        weighted_y = sum(self.pf_y*self.Wpnorms)
#        var_x = np.var(self.pf_x)
#        var_y = np.var(self.pf_y)
#        x_warning, y_warning = self._boundary_warning_sensor()
#        print("x", x_warning)
#        print("y", y_warning)
#        etc_state = np.array([float(wind_x), float(wind_y), float(self.last_x), float(self.last_y), float(self.dur_t), float(self.last_action), float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc)])
#        etc_state = np.array([float(self.wind_d/math.pi), float(self.wind_s), x_warning, y_warning, float(self.last_action), float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc)])
#        etc_state = np.array([float(self.wind_d/math.pi), float(self.wind_s), float(moved_dist), float(self.last_action), float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc)])
#        etc_state = np.array([float(self.wind_d/math.pi), float(self.wind_s), float(self.dur_t), float(x_warning), float(y_warning), float(self.last_action), float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc)])
#        etc_state = np.array([float(self.wind_d/math.pi), float(self.wind_s), float(self.dur_t), float(self.last_action), float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc)])
        etc_state = np.array([float(self.wind_d/math.pi), float(self.wind_s), float(self.dur_t), float(self.agent_x), float(self.agent_y), float(self.last_action), float(self.last_measure), float(self.gas_measure), float(self.last_highest_conc), float(weighted_x), float(weighted_y), float(self.CovXxp), float(self.CovXyp)])

        return etc_state
#        return np.concatenate((etc_state, self.pf_x, self.pf_y, self.Wpnorms), axis=None)


    def _normalize_observation(self, obs):
        normalized_obs = []
        for i in range(0, obs.size):
            normalized_obs.append((obs[i]-self.obs_low_state[i])/(self.obs_high_state[i] - self.obs_low_state[i]))
        return np.array(normalized_obs)

    def _gas_measure(self, pos_x, pos_y):
        conc = self._gas_conc(self.agent_x, self.agent_y)
        conc_env = self.np_random.normal(conc,self.env_sig)
        while conc_env < 0:
            conc_env = self.np_random.normal(conc,self.env_sig)
        gas_measure = self.np_random.normal(conc_env, conc_env*self.sensor_sig_m)
        while gas_measure < 0:
            gas_measure = self.np_random.normal(conc_env, conc_env*self.sensor_sig_m)

        return gas_measure

    def _reward_goal_reached(self):
        return 10 #100

    def _reward_failed(self):
        return 0 #-100

    def _step_reward(self):
        if self.gas_measure > self.last_highest_conc: # need to be adjusted for different source condition
            reward = 1 #+1
            self.dur_t = 0
        else:
            reward = 0
            self.dur_t += 1
        return reward

    def _border_reward(self):
        reward = -10 #-100
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

    def _info_function(self, norm_obs, action, done, rew):
        if self.normalization:
#           warning_x_text = str(norm_obs[2]*(self.agent_dist*2)-self.agent_dist)
#           warning_y_text = str(norm_obs[3]*(self.agent_dist*2)-self.agent_dist)
#           print(norm_obs[3])
#           print(norm_obs[4])
           info = "time step:" + str(self.count_actions) + ", act:" + str(
                   round(float(action)*180,2)) + ", local flow: (" + str(round(norm_obs[0]*180,2)) + "degree, " + str(
                   round(norm_obs[1]*20,2)) + "), duration time: " + str(round(norm_obs[2]*self.max_step,2)) + ", current pos: (" + str(
                   round(norm_obs[3]*self.court_lx,2)) + ", " + str(round(norm_obs[4]*self.court_ly,2)) + "), last action: " + str(
                   round(float(norm_obs[5])*180/math.pi,2)) + ", last conc:" + str(round(norm_obs[6]*self.conc_max,2)) + ", conc:" + str(
                   round(norm_obs[7]*self.conc_max,2)) + ", highest conc:" + str(round(norm_obs[8]*self.conc_max,2)) +", particle center: (" + str(
                   np.round(norm_obs[9]*self.court_lx,2)) + ", " + str(
                   np.round(norm_obs[10]*self.court_ly,2)) + "), variance: (" + str(
                   np.round(norm_obs[11]*pow(self.court_lx,2),2)) + ", " + str(round(norm_obs[12]*pow(self.court_ly,2),2)) + ", agent current pos: (" + str(round(self.agent_x,2)) + ", " + str(
                   round(self.agent_y,2)) + ")", "goal pos: (" + str(round(self.goal_x,2)) + "," + str(round(self.goal_y,2)) + "), done: " + str(done) + ", rew:" + str(rew)
        else:
           info = "time step:" + str(self.count_actions) + ", act:" + str(
                   round(float(action)*180,2)) + ", local flow: (" + str(round(obs[0],2)) + ", " + str(
                   round(obs[1],2)) + "), last pos: (" + str(round(obs[2],2)*self.court_lx) + ", " + str(
                   round(obs[3],2)*self.court_ly) + "), dur_t from last capture: " + str(round(obs[4],2)) + ", last action: " + str(
                   round(float(obs[5])*180,2)) + ", last conc" + str(round(obs[6],2)) + ", conc" + str(round(obs[7],2)) + ", last highest conc" + str(
                   round(obs[8],2)) + ", rew:" + str(rew) + ", current pos: (" + str(round(self.agent_x,2)) + "," + str(
                   round(self.agent_y,2)) + ")", "goal pos: (" + str(self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)
        return info


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
        # set initial state randomly
        self._set_init_state()

        self.last_x = self.agent_x-self.agent_dist
        self.last_y = self.agent_y

        self.dur_t = 0
        self.last_highest_conc = self.conc_eps

        self.pf_x = self.np_random.uniform(low=self.pf_low_state_x, high=self.pf_high_state_x)
        self.pf_y = self.np_random.uniform(low=self.pf_low_state_y, high=self.pf_high_state_y)
        self.pf_q = self.np_random.uniform(low=np.zeros(self.Wpnorms.size), high=np.ones(self.Wpnorms.size)*3000)
        self.Wps = np.ones(self.pf_num)/self.pf_num
        self.Wpnorms = self.Wps

        self.CovXxp = np.var(self.pf_x)
        self.CovXyp = np.var(self.pf_y)
        self.cov_val = np.sqrt(pow(self.CovXxp,2) + pow(self.CovXyp,2))
        self.cov_last_highest = self.cov_val

        if math.sqrt(pow(self.goal_y - self.agent_y,2) + pow(self.goal_x == self.agent_x,2) ) < self.court_lx*0.3:
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
            goal.set_color(255, 0, 0)
            self.viewer.add_geom(goal)

            for i in range(0,self.pf_num):
                particle = rendering.make_circle(3)
                particle.add_attr(rendering.Transform(translation=(self.pf_x[i]*self.scale, self.pf_y[i]*self.scale)))
                particle.set_color(0,255,0)
                self.viewer.add_onetime(particle)

#            text = 'This is a test but it is not visible'
#            label = pyglet.text.Label(text, font_size=36,
#                                      x=10, y=10, anchor_x='left', anchor_y='bottom',
#                                      color=(255, 123, 255, 255))
#            label.draw()
#            self.viewer.add_geom(DrawText(label))

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')
