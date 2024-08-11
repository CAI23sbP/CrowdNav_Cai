from omegaconf import OmegaConf
import numpy as np 
import collections
import random 

from crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.map import Map
import math
from typing import List, Tuple
Obstacle = collections.namedtuple(
    "Obstacle", ["location_x", "location_y", "dim", "patch"])
from numpy.linalg import norm

class MapManager():
    def __init__(self, config:OmegaConf):
        ## initialize map data
        self.sim_map = Map(map_resolution = config.env_config.map.resolution,
                            map_size_m = config.env_config.map.map_size_m,
                            map_data = np.zeros([1,1]),
                            map_as_obs=[],
                            obstacle_vertices=[],
                            border = [],
                            BORDER_OFFSET = 1., 
                            )
        self.num_circles = config.env_config.map.num_circles
        self.num_walls = config.env_config.map.num_walls
        self.discomfort_dist = config.env_config.map.discomfort_dist
        self.circle_inflation_rate_il = config.env_config.map.circle_inflation_rate_il
        self.wall_inflation_rate_il = config.env_config.map.wall_inflation_rate_il
        self.apply_map = config.env_config.map.apply_map

    def generate_map(
        self, 
        robot: Robot, 
        humans: List[Human],
        ) -> Map:
        grid_size = int(round(self.sim_map.map_size_m / self.sim_map.map_resolution))
        self.sim_map.map_data = np.ones((grid_size, grid_size))
        max_locations = int(round(grid_size))
        obstacles = []
        obstacle_vertices = []

        def generate_circle_points(center_x, center_y, radius, num_points=36):
            circle_points = []
            angle_increment = 2 * math.pi / num_points
            for i in range(num_points):
                angle = i * angle_increment
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                circle_points.append((x, y))
            return circle_points
        
        for _ in range(self.num_circles):
            while True:
                location_x = np.random.randint(
                    -max_locations / 2.0, max_locations / 2.0)
                location_y = np.random.randint(
                    -max_locations / 2.0, max_locations / 2.0)
                circle_radius = (np.random.random() + 0.5) * 0.7
                dim = (int(round(2 * circle_radius / self.sim_map.map_resolution)),
                       int(round(2 * circle_radius / self.sim_map.map_resolution)))
                patch = np.zeros([dim[0], dim[1]])

                location_x_m = location_x * self.sim_map.map_resolution
                location_y_m = location_y * self.sim_map.map_resolution

                collide = False
                if norm(
                    (location_x_m - robot.px,
                     location_y_m - robot.py)) < circle_radius + robot.radius + self.discomfort_dist or norm(
                    (location_x_m - robot.gx,
                     location_y_m - robot.gy)) < circle_radius + robot.radius + self.discomfort_dist:
                    collide = True
                if not collide:
                    break
            obstacles.append(Obstacle(int(round(location_x + grid_size / 2.0)),
                                      int(round(location_y + grid_size / 2.0)), dim, patch))
            circle_radius_inflated = self.circle_inflation_rate_il * circle_radius
            ## make sure circle
            circle_vertices = generate_circle_points(location_x_m, location_y_m, circle_radius_inflated)
            obstacle_vertices.append(circle_vertices)

        for _ in range(self.num_walls):
            while True:
                location_x = np.random.randint(
                    -max_locations / 2.0, max_locations / 2.0)
                location_y = np.random.randint(
                    -max_locations / 2.0, max_locations / 2.0)
                if np.random.random() > 0.5:
                    x_dim = np.random.randint(2, 4)
                    y_dim = 1
                else:
                    y_dim = np.random.randint(2, 4)
                    x_dim = 1
                dim = (int(round(x_dim / self.sim_map.map_resolution)),
                       int(round(y_dim / self.sim_map.map_resolution)))
                patch = np.zeros([dim[0], dim[1]])

                location_x_m = location_x * self.sim_map.map_resolution
                location_y_m = location_y * self.sim_map.map_resolution

                collide = False

                if (abs(location_x_m -
                        robot.px) < x_dim /
                    2.0 +
                    robot.radius +
                    self.discomfort_dist and abs(location_y_m -
                                                 robot.py) < y_dim /
                    2.0 +
                    robot.radius +
                    self.discomfort_dist) or (abs(location_x_m -
                                                  robot.gx) < x_dim /
                                              2.0 +
                                              robot.radius +
                                              self.discomfort_dist and abs(location_y_m -
                                                                           robot.gy) < y_dim /
                                              2.0 +
                                              robot.radius +
                                              self.discomfort_dist):
                    collide = True
                if not collide:
                    break

            obstacles.append(Obstacle(int(round(location_x + grid_size / 2.0)),
                                      int(round(location_y + grid_size / 2.0)), dim, patch))
            x_dim_inflated = (self.wall_inflation_rate_il + np.random.uniform(-0.2, 0.3)) * x_dim
            y_dim_inflated = (self.wall_inflation_rate_il + np.random.uniform(-0.2, 0.3)) * y_dim
            obstacle_vertices.append([(location_x_m +
                                            x_dim_inflated /
                                            2.0, location_y_m +
                                            y_dim_inflated /
                                            2.0), (location_x_m -
                                                   x_dim_inflated /
                                                   2.0, location_y_m +
                                                   y_dim_inflated /
                                                   2.0), (location_x_m -
                                                          x_dim_inflated /
                                                          2.0, location_y_m -
                                                          y_dim_inflated /
                                                          2.0), (location_x_m +
                                                                 x_dim_inflated /
                                                                 2.0, location_y_m -
                                                                 y_dim_inflated /
                                                                 2.0)])

        for obstacle in obstacles:
            if obstacle.location_x > obstacle.dim[0] / 2.0 and \
                    obstacle.location_x < grid_size - obstacle.dim[0] / 2.0 and \
                    obstacle.location_y > obstacle.dim[1] / 2.0 and \
                    obstacle.location_y < grid_size - obstacle.dim[1] / 2.0:

                start_idx_x = int(
                    round(
                        obstacle.location_x -
                        obstacle.dim[0] /
                        2.0))
                start_idx_y = int(
                    round(
                        obstacle.location_y -
                        obstacle.dim[1] /
                        2.0))
                self.sim_map.map_data[start_idx_x:start_idx_x +
                         obstacle.dim[0], start_idx_y:start_idx_y +
                         obstacle.dim[1]] = np.minimum(self.sim_map.map_data[start_idx_x:start_idx_x +
                                                                obstacle.dim[0], start_idx_y:start_idx_y +
                                                                obstacle.dim[1]], obstacle.patch)

            else:
                for idx_x in range(obstacle.dim[0]):
                    for idx_y in range(obstacle.dim[1]):
                        shifted_idx_x = idx_x - obstacle.dim[0] / 2.0
                        shifted_idx_y = idx_y - obstacle.dim[1] / 2.0
                        submap_x = int(
                            round(
                                obstacle.location_x +
                                shifted_idx_x))
                        submap_y = int(
                            round(
                                obstacle.location_y +
                                shifted_idx_y))
                        if submap_x > 0 and submap_x < grid_size and submap_y > 0 and submap_y < grid_size:
                            self.sim_map.map_data[submap_x,
                                                    submap_y] = obstacle.patch[idx_x, idx_y]
        
        if robot.policy.name in ['cadrl','sarl','lstm','srnn','gcn']:
            map_as_obs = self.create_observation_from_static_obstacles(obstacles, obstacle_vertices)
        else:
            map_as_obs = []
        self.sim_map.obstacle_vertices = obstacle_vertices
        self.sim_map.map_as_obs = map_as_obs
        self.sim_map = self._add_border_obstacle(self.sim_map ,robot, humans)
        return self.sim_map

    
    def create_observation_from_static_obstacles(
        self, 
        obstacles: List[Obstacle],
        obstacle_vertices: List
        )-> List[ObservableState]:
        if len(obstacle_vertices)>0:
            static_obstacles_as_pedestrians = []
            for index, obstacle in enumerate(obstacles):
                if obstacle.dim[0] == obstacle.dim[1]:  # Obstacle is a square
                    px = (
                        obstacle_vertices[index][0][0] + obstacle_vertices[index][2][0]) / 2.0
                    py = (
                        obstacle_vertices[index][0][1] + obstacle_vertices[index][2][1]) / 2.0
                    radius = (
                        obstacle_vertices[index][0][0] - px) * np.sqrt(2)
                    static_obstacles_as_pedestrians.append(
                        ObservableState(px, py, 0, 0, radius))
                elif obstacle.dim[0] > obstacle.dim[1]:  # Obstacle is rectangle
                    py = (
                        obstacle_vertices[index][0][1] + obstacle_vertices[index][2][1]) / 2.0
                    radius = (
                        obstacle_vertices[index][0][1] - py) * np.sqrt(2)
                    px = obstacle_vertices[index][1][0] + radius
                    while px < obstacle_vertices[index][0][0]:
                        static_obstacles_as_pedestrians.append(
                            ObservableState(px, py, 0, 0, radius))
                        px = px + 2 * radius
                else:  # Obstacle is rectangle
                    px = (
                        obstacle_vertices[index][0][0] + obstacle_vertices[index][2][0]) / 2.0
                    radius = (
                        obstacle_vertices[index][0][0] - px) * np.sqrt(2)
                    py = obstacle_vertices[index][2][1] + radius
                    while py < obstacle_vertices[index][0][1]:
                        static_obstacles_as_pedestrians.append(
                            ObservableState(px, py, 0, 0, radius))
                        py = py + 2 * radius
        return static_obstacles_as_pedestrians
    
    def _add_border_obstacle(
        self,
        sim_map: Map,
        robot: Robot,
        humans: List[Human],
        
        ):
        all_agents = [robot] + humans 
        x_agents = [a.px for a in all_agents]
        y_agents = [a.py for a in all_agents]
        x_goals = [a.gx for a in all_agents]
        y_goals = [a.gy for a in all_agents]
        if len(sim_map.obstacle_vertices)>0:
            all_vertices = np.concatenate(sim_map.obstacle_vertices).reshape((-1,2))
            x_vertices = list(all_vertices[:,0])
            y_vertices = list(all_vertices[:,1])
            sim_map.BORDER_OFFSET = 1.
            x_min = min(x_agents + x_goals + x_vertices) - sim_map.BORDER_OFFSET
            x_max = max(x_agents + x_goals + x_vertices) + sim_map.BORDER_OFFSET
            y_min = min(y_agents + y_goals + y_vertices) - sim_map.BORDER_OFFSET
            y_max = max(y_agents + y_goals + y_vertices) + sim_map.BORDER_OFFSET
        else:
            sim_map.BORDER_OFFSET = 1.
            x_min = min(x_agents + x_goals ) - sim_map.BORDER_OFFSET
            x_max = max(x_agents + x_goals ) + sim_map.BORDER_OFFSET
            y_min = min(y_agents + y_goals ) - sim_map.BORDER_OFFSET
            y_max = max(y_agents + y_goals ) + sim_map.BORDER_OFFSET
        if self.apply_map:
            border_vertices = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)][::-1]
            sim_map.obstacle_vertices.append(border_vertices)
            sim_map.border = [(x_min, x_max), (y_min, y_max)]
        else:
            sim_map.obstacle_vertices.append([])

        return sim_map
