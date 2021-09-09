import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from gym_ste.envs.ste_pf_base_3 import *

from gym.envs.registration import register

from datetime import datetime


class StePFilterConvBaseEnv3(StePFilterBaseEnv3):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        StePFilterBaseEnv3.__init__(self)

    def _reward_goal_reached(self):
        return 10000

    def step(self, action):
        self.warning = False
        self.outborder = False
        self.count_actions += 1
        self._calculate_position(action)
        obs = self._observation()

        self.last_action = action

        x_warning, y_warning = self._boundary_warning_sensor()

        # done for step rewarding
        agent_dist = self.agent_v*self.delta_t
        self.cov_val = np.sqrt(self.CovXxp + self.CovXyp)
        converge_done = bool(self.cov_val < agent_dist)
        done = bool(self._distance(self.agent_x, self.agent_y) <= self.eps)

        rew = 0
        if self.outborder: # Agent get out to search area
           rew += self._border_reward()
        reach_source_done = bool(self._distance(self.agent_x, self.agent_y) <= self.eps)
        if reach_source_done:
           rew += 100

        pf_center = np.array([np.mean(self.pf_x), np.mean(self.pf_y)])
        nearby = math.sqrt( pow(pf_center[0]-self.goal_x,2) + pow(pf_center[1]-self.goal_y,2) )
        if not converge_done:
            rew += self._step_reward()
        else: # particle filter is converged
            nearby_bool = bool(nearby<agent_dist*2)
            if nearby_bool:
                rew = self._reward_goal_reached()
#            else:
#                rew = self._reward_goal_reached()/2
#            else:
#                rew = self._reward_failed()

#        source_done = bool(self._distance(self.agent_x, self.agent_y) <= self.eps)
#        if source_done:
#            rew = self._reward_goal_reached()

        terminate_done = bool(self.count_actions >= self.max_step)
#        if terminate_done: # Reach the max_step without finding source
#            rew = self._reward_failed()

        # break if more than max_step actions taken
        done = bool(self.count_actions >= self.max_step or converge_done or self.outborder)
#        done = bool(self.count_actions >= self.max_step or converge_done)

        # track, where agent was
        self.positions.append([self.agent_x, self.agent_y])
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



# register(
#     id='StePFilterConvEasyEnv-v0',
#     entry_point='gym_ste.envs:StePFilterConvEasyEnv',
# )

