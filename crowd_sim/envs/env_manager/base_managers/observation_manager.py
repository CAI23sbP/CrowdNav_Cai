from crowd_sim.envs.utils.state import *
import numpy as np
from omegaconf import OmegaConf
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.map import Map
from pose2d import (
apply_tf_to_vel, 
inverse_pose2d, 
apply_tf_to_pose
)
from CMap2D import flatten_contours, render_contours_in_lidar,  CSimAgent
from CMap2D import CMap2D as cm2d
from typing import (
List,
Tuple,
Dict,
Any
)
from copy import copy
from collections import deque
import CMap2D

class ObervationManager():
    def __init__(self, config: OmegaConf):
        self.scan_increment = config.obs_config.scan.increment
        self.scan_min_angle = config.obs_config.scan.min_angle
        self.scan_max_angle = config.obs_config.scan.max_angle + self.scan_increment
        self.n_angles = config.obs_config.scan.n_angles
        self.max_range = config.obs_config.scan.max_range
        self.converter_cmap2d = cm2d()
        self.converter_cmap2d.set_resolution(1.)
        self.time_step = config.env_config.sim.time_step
        self.lidar_legs = config.obs_config.scan.lidar_legs
        self.leg_radius = config.obs_config.scan.leg_radius
        self.human_num = config.env_config.sim.human_num
        self.wall_num = config.env_config.map.num_walls + config.env_config.map.num_circles
        
    def _cb_lidar(
        self, 
        robot: Robot, 
        humans: List[Human,], 
        flat_contours: np.ndarray,
        distances_travelled_in_base_frame: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[bool]]:
        """
        create lidar
        """
        lidar_pos = np.array([robot.px, robot.py, robot.theta], dtype=np.float32)
        ranges = np.ones((self.n_angles,), dtype=np.float32) * self.max_range
        angles = np.linspace(self.scan_min_angle,
                             self.scan_max_angle-self.scan_increment,
                             self.n_angles) + lidar_pos[2]
        render_contours_in_lidar(ranges, angles, flat_contours, lidar_pos[:2])
        other_agents = []
        for i, human in enumerate(humans):
            pos = np.array([human.px, human.py, human.theta], dtype=np.float32)
            dist = distances_travelled_in_base_frame[i].astype(np.float32)
            vel = np.array([human.vx, human.vy], dtype=np.float32)
            if self.lidar_legs:
                agent = CSimAgent(pos, dist, vel, type_="legs", radius = self.leg_radius)
            else:
                agent = CSimAgent(pos, dist, vel, type_="trunk", radius=human.radius)
            other_agents.append(agent)
        self.converter_cmap2d.render_agents_in_lidar(ranges, angles, other_agents, lidar_pos[:2])
        which_visible = [agent.get_agent_which_visible() for agent in other_agents]
        return ranges, angles, distances_travelled_in_base_frame, which_visible

    def _cb_robot_pose(
        self, 
        robot: Robot
        ) -> np.ndarray:
        """
        transform robot tf
        """
        baselink_in_world = np.array([robot.px, robot.py, robot.theta])
        world_in_baselink = inverse_pose2d(baselink_in_world)
        robotvel_in_world = np.array([robot.vx, robot.vy, 0])  
        robotvel_in_baselink = apply_tf_to_vel(robotvel_in_world, world_in_baselink)
        goal_in_world = np.array([robot.gx, robot.gy, 0])
        goal_in_baselink = apply_tf_to_pose(goal_in_world, world_in_baselink)
        robotstate_obs = np.hstack([goal_in_baselink[:2], robotvel_in_baselink[:2], robot.radius])
        return robotstate_obs

    def _cb_human_distance(
        self,
        humans: List[Human],
        distances_travelled_in_base_frame: np.ndarray
    ) -> np.ndarray:
        """
        transform human tf and get distancefrom human
        """
        for i, human in enumerate(humans):
            vrot = 0.
            if len(self.states) > 1:
                vrot = (self.states[-1][i].theta
                        - self.states[-2][i].theta) / self.time_step
            # transform world vel to base vel
            baselink_in_world = np.array([human.px, human.py, human.theta])
            world_in_baselink = inverse_pose2d(baselink_in_world)
            vel_in_world_frame = np.array([human.vx, human.vy, vrot])
            vel_in_baselink_frame = apply_tf_to_vel(vel_in_world_frame, world_in_baselink)
            distances_travelled_in_base_frame[i, :] += vel_in_baselink_frame * self.time_step
        
        return distances_travelled_in_base_frame
    
    def _cb_dijkstra_path_planning(self, robot:Robot, sim_map:Map) -> np.ndarray:
        """
        might be useful at testing
        """
        copy_cmap2d = copy(self.converter_cmap2d)
        copy_cmap2d.from_closed_obst_vertices(contours = sim_map.obstacle_vertices,
                                              resolution = 0.01,
                                              )
        occupancy_map = copy_cmap2d.occupancy() 
        map_sdf = copy_cmap2d.as_tsdf(1)
        occupancy_map = np.rot90(occupancy_map)
        end_point = [robot.gx,robot.gy]
        ij = copy_cmap2d.xy_to_ij(np.array([end_point], dtype=np.float32)
                                            , clip_if_outside= True)
        cost = copy_cmap2d.dijkstra(ij[0],extra_costs=1/(0.0000001 + map_sdf), inv_value=-1, connectedness=32)

        start_point = [robot.px, robot.py]
        ij = copy_cmap2d.xy_to_ij(np.array([start_point], dtype=np.float32)
                                            ,clip_if_outside= True)
        path, _ = CMap2D.path_from_dijkstra_field(cost, ij[0])
        path = copy_cmap2d.ij_to_xy(np.array(path))
        return path
    
    def reset_observation(
        self, 
        robot: Robot, 
        humans: List[Human],
        sim_map: Map,
        use_path: bool = False 
        )-> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        ## must be calling def reset function
        distances_travelled_in_base_frame = np.zeros((len(humans), 3))
        self.flat_contours = flatten_contours(sim_map.obstacle_vertices)
        self.states = deque(maxlen=2) 
        self.states.append([human.get_full_state() for human in humans])
        path = None
        if use_path:
            path =  self._cb_dijkstra_path_planning(robot, sim_map)
        obs, distances_travelled_in_base_frame = self.get_observation(robot, humans, distances_travelled_in_base_frame)
        return obs, distances_travelled_in_base_frame, path
    
    def get_observation(    
        self, 
        robot: Robot, 
        humans: List[Human],
        distances_travelled_in_base_frame: np.ndarray,
        ) -> Tuple[Dict[str, Any], np.ndarray]:
        distances_travelled_in_base_frame = self._cb_human_distance(humans=humans,
                            distances_travelled_in_base_frame = distances_travelled_in_base_frame
                            )
        ranges, angles, distances_travelled_in_base_frame, which_visible = self._cb_lidar(robot=robot, 
                       humans=humans,
                       flat_contours=self.flat_contours,
                       distances_travelled_in_base_frame=distances_travelled_in_base_frame)
        robotstate_obs = self._cb_robot_pose(robot=robot)
        self.states.append([human.get_full_state() for human in humans])
        return {'ranges':ranges,
                'angles':angles,
                'robotstate_obs':robotstate_obs,
                'which_visible':which_visible,
                } , distances_travelled_in_base_frame
