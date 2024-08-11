from numpy.linalg import norm
import numpy as np 
from omegaconf import OmegaConf
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.action import ActionXY, ActionRot, ActionXYRot
from typing import Optional

class RewardManager():
    def __init__(self,config: OmegaConf):
        self.config = config 
        self.discomfort_dist = config.env_config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.env_config.reward.discomfort_penalty_factor
        self.time_limit = config.env_config.sim.time_limit
        self.time_step = config.env_config.sim.time_step
        self.robot_policy = config.env_config.robot.policy
        self.success_reward = config.env_config.reward.success_reward
        self.collision_penalty = config.env_config.reward.collision_penalty
        self.timeout_penalty = config.env_config.reward.timeout_penalty
        self.global_time = 0

    def reward_reset(self):
        self.global_time = 0

    def calc_reward(
        self, 
        robot:Robot, 
        raw_scan:np.ndarray,
        approach :bool= False, 
        potential: Optional[np.ndarray]=None, 
        ):
        collision = False
        min_scan = np.min(raw_scan)
        robot_radius = robot.radius
        if min_scan <= robot_radius + 0.05:
            collision = True 

        # check if reaching the goal
        reaching_goal = norm(np.array(robot.get_position()) - np.array(robot.get_goal_position())) <= robot_radius
        self.global_time += self.time_step 
        reward = 0

        if self.global_time >= self.time_limit - 1:
            done = True 
            episode_info = Timeout()
            reward = self.timeout_penalty
                
        elif collision:
            reward = self.collision_penalty
            done = True
            episode_info = Collision()
            
        elif reaching_goal:
            done = True 
            reward = self.success_reward
            episode_info = ReachGoal()

        elif min_scan < self.discomfort_dist:
            reward = (min_scan - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False 
            episode_info = Discomfort(min_scan)

        else:
            if approach:
                potential_cur = np.linalg.norm(
                    np.array([robot.px, robot.py]) - np.array(robot.get_goal_position()))
                reward = 2 * (-abs(potential_cur) - potential)
                potential = -abs(potential_cur)
                
            done = False
            episode_info = Nothing()
    
        return reward, done, episode_info, potential 