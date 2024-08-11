from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import JointState
from omegaconf import OmegaConf
import numpy as np 

class Robot(Agent):
    def __init__(self, config: OmegaConf, section:str = 'robot'):
        super().__init__( config, section)
        self.max_range = config.obs_config.scan.max_range
        self.robot_fov = config.env_config.robot.FOV
        self.v_pref = config.env_config.robot.v_pref

    def act(self, ob, global_map=None):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        
        state = JointState(self.get_full_state(), ob)
        if global_map is not None:
            action = self.policy.predict(state, global_map, self)
        else:
            action = self.policy.predict(state)
        return action
    
    def transform_to_rot(self, action: ActionXY) -> ActionRot:
        linear_vel = np.linalg.norm([action.vx, action.vy])
        angular_vel = (np.arctan2(
            action.vy, action.vx) - self.theta)
        linear_vel = np.clip(linear_vel, -self.v_pref, self.v_pref)
        angular_vel = np.clip(angular_vel, -self.v_pref, self.v_pref)
        angular_vel = np.clip(angular_vel, -self.v_pref, self.v_pref)

        action = ActionRot(linear_vel,angular_vel )
        return action