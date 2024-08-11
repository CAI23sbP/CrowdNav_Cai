from crowd_sim.envs.base_sim import BaseSim
from gymnasium import spaces
import numpy as np 
from copy import deepcopy
from typing import Deque
from collections import deque

class ExampleSimScan(BaseSim):
    metadata = {'render_modes': ['human'],
                "render_fps": 50,}
    render_mode = 'human'

    def __init__(self):
        super().__init__()
        d = {}
        d['ranges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(4, 360), dtype = np.float64)  
        d['robotstate_obs'] = spaces.Box(low=-np.inf, high=np.inf, shape=(4, 5), dtype = np.float64)  
        action_space = np.array([0.6, 0.6])
        self.action_space = spaces.Box(low=-action_space, high=action_space, shape = (2,), dtype = np.float64)
        self.sequence = 4
        self.observation_space = spaces.Dict(d)
        self.stack_scan = deque(maxlen=self.sequence)
        self.stack_ro_info = deque(maxlen=self.sequence)

    def _stack_obs(self, obs):
        ranges = obs['ranges']
        robotstate_obs = obs['robotstate_obs']
        self.stack_scan.append(ranges)
        self.stack_ro_info.append(robotstate_obs)
        length = self.sequence-len(self.stack_scan)
        if len(self.stack_scan) < self.sequence:
            gd = deepcopy(self.stack_scan)
            gd1 = np.stack([gd[0] for i in range(length)])
            stacked_scan = deepcopy(np.concatenate([gd1, gd]))

            gd = deepcopy(self.stack_ro_info)
            gd1 = np.stack([gd[0] for i in range(length)])
            stacked_ro_info = deepcopy(np.concatenate([gd1, gd]))
        
        else:
            stacked_scan = deepcopy(np.stack(self.stack_scan))
            stacked_ro_info = deepcopy(np.stack(self.stack_ro_info))

        obs['ranges'] = stacked_scan
        obs['robotstate_obs'] = stacked_ro_info     
        return obs 

    def reset(self, phase='train', test_case=None, seed = None, options = None):
        obs = super().reset(phase, test_case, seed , options)
        info = {}
        self.stack_scan.clear()
        self.stack_ro_info.clear()

        obs = self._stack_obs(obs)
        del obs['angles']

        return obs, info
    
    def step(self, action):
        obs, reward, done, info = super().step(action,update=False)
        obs = self._stack_obs(obs)
        del obs['angles']

        return obs, reward, done, False, info
    
    def render(self):
        super().render()