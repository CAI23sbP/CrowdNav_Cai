import numpy as np
from numpy.linalg import norm
import abc
import logging
from crowd_sim.envs.utils.action import ActionXY, ActionRot, ActionXYRot
from crowd_sim.envs.utils.state import ObservableState, FullState, ObservableState_noV
from crowd_sim.envs.policy.policy_factory import policy_factory
from omegaconf import OmegaConf


class Agent(object):
    def __init__(self, config:OmegaConf, section:str):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        subconfig = config.env_config.robot if section == 'robot' else config.env_config.humans
        self.visible = subconfig.visible
        self.v_pref = subconfig.v_pref
        self.radius = subconfig.radius
        if subconfig.policy in ['orca','social_force']:
            self.policy = policy_factory[subconfig.policy](config)
        else:
            self.policy = policy_factory[subconfig.policy](config)
        self.FOV = np.pi * subconfig.FOV
        # for humans: we only have holonomic kinematics; for robot: depend on config
        self.kinematics = 'holonomic' if section == 'humans' else config.env_config.robot.kinematics
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.ax = None
        self.ay = None
        self.theta = None
        self.time_step = config.env_config.sim.time_step
        self.policy.time_step = config.env_config.sim.time_step

        self.max_v_pref = config.env_config.humans.max_v_pref
        self.min_v_pref = config.env_config.humans.min_v_pref

    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))


    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return: 로봇 속도 값 +- 0.2
        """
        self.v_pref = np.random.uniform(self.max_v_pref, self.min_v_pref)  
        # self.radius = np.random.uniform(0.3, 0.4) 
        self.radius = 0.3

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta

        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    # self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta
    def set_list(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.radius = radius
        self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_observable_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius]

    def get_observable_state_noV(self):
        return ObservableState_noV(self.px, self.py, self.radius)

    def get_observable_state_list_noV(self):
        return [self.px, self.py, self.theta]

    def get_special_state_list(self):
        if self.kinematics == 'holonomic':
            # poseX    poseY    goalX   goalY     velX     velY      angle       radius
            return [self.px, self.py, self.gx, self.gy, self.vx, self.vy , self.theta, self.radius]

        else:
            return [self.px, self.py, self.gx, self.gy, self.vx, self.vy , self.theta, self.radius]
            
    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_full_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta]

    def get_full_state_list_noV(self):
        return [self.px, self.py, self.radius, self.gx, self.gy, self.v_pref, self.theta]
        # return [self.px, self.py, self.radius, self.gx, self.gy, self.v_pref]

    def get_traj_state_list(self):
        return [self.px, self.py, self.theta ,self.vx, self.vy]
        #[x(m)0, y(m)1, theta(rad)2, v(m/s)3, omega(rad/s)4]

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy

    def get_start_position(self):
        return self.px, self.py

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    def set_policy(self, policy):
        self.policy = policy

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """

    def check_validity(self, action):
        if self.kinematics == 'unicycle':
            assert isinstance(action, ActionRot)
            
        elif self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t

        else: # unicycle
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t
        return px, py

    def compute_velocity(self, action):
        
        if self.kinematics == 'holonomic':
            vx = action.vx
            vy = action.vy
        else:
            # self.theta = (self.theta + action.r) 
            self.theta = (self.theta + action.r) % (2 * np.pi)
            vx = action.v * np.cos(self.theta)
            vy = action.v * np.sin(self.theta)

        return vx, vy
    

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        last_x, last_y = self.px, self.py
        last_vx, last_vy = self.vx, self.vy
        pos = self.compute_position(action, self.time_step)
        vel = self.compute_velocity(action)

        self.px, self.py = pos
        self.vx, self.vy = vel
        self.ax = (self.vx-last_vx) / self.time_step
        self.ay = (self.vy-last_vy) / self.time_step 

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

