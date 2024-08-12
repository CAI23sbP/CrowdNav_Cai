import numpy as np
import gymnasium as gym 
from crowd_sim.envs.env_manager.obstacle_manager import ObstacleManager 
from crowd_sim.envs.env_manager.visualize_manager import VisualizeManager 
from crowd_sim.envs.env_manager.base_managers.observation_manager import ObervationManager 
from crowd_sim.envs.env_manager.base_managers.reward_manager import RewardManager 
from crowd_sim.envs.env_manager.map_manager import MapManager
from crowd_sim.envs.utils.robot import Robot
from numpy.linalg import norm
import random
from omegaconf import OmegaConf

class BaseSim(gym.Env):
    metadata = {'render_modes':["human", "rgb_array"],
                "render_fps": 50,}
    render_mode = 'rgb_array'

    def __init__(self):
        super(BaseSim,self).__init__()
        self.thisSeed = None 
        self.nenv = None 
        self.potential = None ## for sf human , not ORCA
        self.phase = None   
        self.test_case = None
        self.states = [] ## for render trj
        self.time_limit = None
        self.sending_frame_info = False
        
    def configure(
        self, 
        config: OmegaConf, 
        nenv:int, 
        phase:str, 
        )->None:

        self.nenv = nenv
        self.phase = phase 
        self.time_limit = config.env_config.sim.time_limit
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.env_config.env.val_size,
                            'test': config.env_config.env.test_size}
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        self.random_goal_changing = config.env_config.humans.random_goal_changing
        self.end_goal_changing = config.env_config.humans.end_goal_changing
        self.time_step = config.env_config.sim.time_step
        self.human_num = config.env_config.sim.human_num
        self.wall_num = config.env_config.map.num_walls + config.env_config.map.num_circles
        self.total_obs_num = self.human_num + self.wall_num
        self.robot_actions = []

        self.set_robot(config)
        self.set_manager(config)
        ## default manager
        self.MapManager = MapManager(config)
        self.VisManager = VisualizeManager(config)
        self.ObManager = ObstacleManager(config, self.VisManager)
        self.path = None

    def set_manager(self, config):
        self.ObsManager = ObervationManager(config)
        self.RewardManager = RewardManager(config)
        
    def set_robot(self, config: OmegaConf):
        self.robot = Robot(config, 'robot')
        self.humans = []

    def step(self, action, update=False):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        self.sim_time += self.time_step
        if self.robot.policy.name != 'ORCA':
            action = self.robot.policy.clip_action(action)
        else:
            ## get action by ORCA
            orca_obs = []
            for human in self.humans:
                if self.VisManager.detect_visible(self.robot, human):
                    orca_obs.append(human.get_observable_state())
            action = self.robot.act(orca_obs, self.sim_map.obstacle_vertices)
            action = self.robot.transform_to_rot(action)

        reward, done, episode_info, self.potential = self.RewardManager.calc_reward(
                                                                                    robot = self.robot,                
                                                                                    potential = self.potential,
                                                                                    raw_scan = self.lidar_scan,
                                                                                    approach = True,
                                                                                    )

        info = {'info': episode_info}
        self.humans, self.robot = self.ObManager.step_agent(action = action, 
                                                            humans = self.humans, 
                                                            sim_map = self.sim_map,
                                                            robot = self.robot)
        
        obs_dict, self.distances_travelled_in_base_frame = self.ObsManager.get_observation(robot = self.robot, 
                                                                                            humans=self.humans, 
                                                                                            distances_travelled_in_base_frame=self.distances_travelled_in_base_frame, 
                                                                                            )
        self.lidar_scan = obs_dict['ranges']
        self.lidar_angles = obs_dict['angles']
        self.which_visible = obs_dict['which_visible']
        del obs_dict['which_visible']

        self.global_time += self.time_step # max episode length=time_limit/time_step
        self._update_human_goal()
        # # Update all humans' goals randomly midway through episode
        if update:
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],
                            [human.id for human in self.humans]]) ## visualize
            
        return obs_dict, reward, done, info

    def _update_human_goal(self):

        if self.random_goal_changing:
            if self.global_time % 1 == 0:
                self.ObManager.update_human_goals_randomly(self.robot, self.humans)

        # # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for human in self.humans:
                if not human.isObstacle and human.v_pref != 0 and norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.ObManager.update_human_goal(self.robot, human, self.humans)


    def reset(self, phase='train', test_case=None, seed = 0 , options = None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case=self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate case
        self.global_time = 0
        self.sim_time = 0

        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}
        if self.thisSeed is None:
            self.thisSeed = seed 

        seed = counter_offset[phase] + self.case_counter[phase] + self.thisSeed
        self.set_seed(seed)
        self.robot, self.humans = self.ObManager.generate_robot_humans(self.robot, phase, self.human_num)
        self.sim_map = self.MapManager.generate_map(self.robot, self.humans)
        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        # get current observation
        obs_dict, self.distances_travelled_in_base_frame, self.path = self.ObsManager.reset_observation(self.robot, 
                                                                                            self.humans, 
                                                                                            self.sim_map)
        self.lidar_scan = obs_dict['ranges']
        self.lidar_angles = obs_dict['angles']
        self.which_visible = obs_dict['which_visible']
        del obs_dict['which_visible']
        self.RewardManager.reward_reset()
        # initialize potential
        self.potential = -abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))
        
        return obs_dict
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

    def render(self):
        self.VisManager.render(robot = self.robot, 
                                humans = self.humans ,
                                lidar_angles = self.lidar_angles ,
                                lidar_scan= self.lidar_scan,
                                obstacle_vertices = self.sim_map.obstacle_vertices, 
                                sim_time = self.sim_time,
                                which_visible = self.which_visible,
                                path = self.path ,
                                phase = self.phase
                                )
 
