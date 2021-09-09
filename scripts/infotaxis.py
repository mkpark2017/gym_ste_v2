import gym
import time
import matplotlib.pyplot as plt
import numpy as np
import math

#import gym_ste.envs

DEBUG = True
env = gym.make('gym_ste:StePFilterConvInfotaxisVeryHardEnv-v0')

def _pf_gas_conc(self, pos_x, pos_y, source_x, source_y, source_q): # true gas conectration
        avoid_zero = (np.sqrt(pow(source_x - pos_x,2) + pow(source_y - pos_y,2) ) < 1e-50) 
        source_x[avoid_zero] += 1e-50
        source_y[avoid_zero] += 1e-50

        dist = np.sqrt(pow((source_x - pos_x), 2) + pow(source_y - pos_y, 2))
        y_n = -(pos_x - source_x)*np.sin(self.wind_d*math.pi/180)+ \
               (pos_y - source_y)*math.cos(self.wind_d*math.pi/180)
        lambda_plume = math.sqrt(self.gas_d * self.gas_t / (1 + pow(self.wind_s,2) * self.gas_t/4/self.gas_d) )
        conc_com_1 = source_q/(4 * math.pi * self.gas_d * dist) 
        conc_com_2 = np.exp( -y_n * self.wind_mean_speed/(2*self.gas_d) - dist/lambda_plume)
        conc = conc_com_1 * conc_com_2
        return conc

def _weight_calculate(self, pf_x, pf_y, pf_q):
        pf_conc = self._pf_gas_conc(self.agent_x, self.agent_y, pf_x, pf_y, pf_q)
        #mean_pf_conc = sum(pf_conc*self.Wpnorms)
        #mean_conc = (mean_pf_conc + self.gas_measure)/2
        mean_conc = (pf_conc + self.gas_measure)/2
        pdetSig = np.sqrt( pow((mean_conc*self.sensor_sig_m),2) + pow(self.env_sig,2) )
        #if pdetSig < 1e-100: pdetSig = 1e-100
        pdetSig[pdetSig < 1e-100] = 1e-100
        pdetSig_sq = pow(pdetSig, 2)
        gauss_val = (self.gas_measure - pf_conc)/pdetSig
        gauss_new = 1/(math.sqrt(2*math.pi)*pdetSig_sq)*np.exp(-pow(gauss_val,2)/2)
        
        gauss_new[gauss_new != gauss_new] = 1e-200
        gauss_new[gauss_new < 1e-200] = 1e-200
        return gauss_new



def particle_filter(x, y, pf_x, pf_y, pf_q, wpnorms):
    self.update_count += 1
    Wp_sum = 0
    resample_true = False
    gauss_new = self._weight_calculate(self.pf_x, self.pf_y, self.pf_q)
#        if (gauss_new == np.ones(self.pf_num)/self.pf_num).all(): resample_true = True
    sort_g = np.sort(gauss_new)
    if (sort_g[self.pf_num-1] == sort_g[0] or self.update_count == 10): resample_true = True
    self.Wps = self.Wpnorms * gauss_new
    Wp_sum = np.sum(self.Wps)

    Wpnorms = self.Wps/Wp_sum

    return pf_x, pf_y, pf_q, Wpnorms


for e in range(100):
    obs = env.reset()
    cumulated_reward = 0
    i = 0
    done = False
    env.render_background(mode='human')
    pf_num = 150
    etc_num = 9
    while not done and i <= 300:
    #while 1:
        i += 1
        new_util = np.ones(4)*np.nan
        #obs_temp = obs
        for i in range(4):
            angle = (-1 + 0.5*i)*math.pi
            [new_agent_x, new_agent_y] = np.ones(2)*np.nan
            new_agent_x = obs[6] + round(math.cos(angle))*obs[0]
            new_agent_y = obs[7] + round(math.sin(angle))*obs[0]

            pf_x, pf_y, pf_q, Wpnorm = particle_filter(new_agent_x, new_agent_y, obs[12:12+pf_num], obs[12+pf_num:12+pf_num*2],
                                                       obs[12+pf_num*2:12+pf_num*3], obs[12+pf_num*3:12+pf_num*4])


            
        act = 1
        obs, rew, done, info = env.step(act)     # take a random action
    #    env.render_background(mode='human')

        env.render(mode='human')
        cumulated_reward += rew
        time.sleep(0.1)
        if DEBUG:
            print(info)
#        env.close()
    if DEBUG and done:
        time.sleep(3)

    print("episode ended with cumulated rew", cumulated_reward, "and done:", done)
    env.close()
