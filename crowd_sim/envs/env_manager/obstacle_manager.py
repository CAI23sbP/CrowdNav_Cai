import numpy as np
from numpy.linalg import norm
import logging, random,copy
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.map import Map
from typing import List, Tuple, Optional, Union
from omegaconf import OmegaConf
from crowd_sim.envs.env_manager.visualize_manager import VisualizeManager

class ObstacleManager():
    def __init__(self, config: OmegaConf, VisManager: VisualizeManager):
        self.discomfort_dist = config.env_config.reward.discomfort_dist
        self.VisManager = VisManager

        self.goal_pose = config.env_config.robot.goal_pose
        self.random_unobservability = config.env_config.humans.random_unobservability
        if self.random_unobservability:
            self.unobservable_chance = config.env_config.humans.unobservable_chance
        
        self.random_goal_changing = config.env_config.humans.random_goal_changing
        if self.random_goal_changing:
            self.goal_change_chance = config.env_config.humans.goal_change_chance

        self.end_goal_changing = config.env_config.humans.end_goal_changing
        if self.end_goal_changing:
            self.end_goal_change_chance = config.env_config.humans.end_goal_change_chance

        self.random_radii = config.env_config.humans.random_radii
        self.random_v_pref = config.env_config.humans.random_v_pref

        self.dummy_human = Human(config, 'humans')
        self.dummy_human.set(7, 7, 7, 7, 0, 0, 0) # (7, 7, 7, 7, 0, 0, 0)
        self.dummy_human.time_step = config.env_config.sim.time_step
        self.dummy_robot = Robot(config, 'robot')
        self.dummy_robot.set(7, 7, 7, 7, 0, 0, 0)
        self.dummy_robot.time_step = config.env_config.sim.time_step
        self.dummy_robot.kinematics = 'holonomic'
        self.dummy_robot.policy = ORCA(config)

        self.randomize_attributes = config.env_config.sim.randomize_attributes
        self.circle_radius = config.env_config.sim.circle_radius
        self.square_width = config.env_config.sim.square_width
        self.human_num = config.env_config.sim.human_num
        self.group_human = config.env_config.sim.group_human

        self.step_v_pref = config.env_config.humans.step_v_pref
        self.a_min_pref = config.env_config.humans.a_min_pref
        self.a_max_pref = config.env_config.humans.a_max_pref
        self.config = config 

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
            
    def generate_circle_group_obstacle(
        self,
        robot: Robot,
        humans: List[Human],
        circum_num:int,
        ) -> Tuple[List[Human], List]:

        group_circumference = self.config.env_config.humans.radius * 2 * circum_num
        group_radius = group_circumference / (2 * np.pi)

        while True:
            rand_cen_x = np.random.uniform(-3, 3)
            rand_cen_y = np.random.uniform(-3, 3)
            success = True
            for i, group in enumerate(self.circle_groups):
                dist_between_groups = np.sqrt((rand_cen_x - group[1]) ** 2 + (rand_cen_y - group[2]) ** 2)
                sum_radius = group_radius + group[0] + 2 * self.config.env_config.humans.radius
                if dist_between_groups < sum_radius:
                    success = False
                    break
            if success:
                break
        self.circle_groups.append((group_radius, rand_cen_x, rand_cen_y))
        arc = 2 * np.pi / circum_num
        for i in range(circum_num):
            angle = arc * i
            curr_x = rand_cen_x + group_radius * np.cos(angle)
            curr_y = rand_cen_y + group_radius * np.sin(angle)
            point = (curr_x, curr_y)
            curr_human = self.generate_circle_static_obstacle(robot, humans, point)
            curr_human.isObstacle = True
            humans.append(curr_human)

        return humans

    def generate_circle_static_obstacle(
        self, 
        robot: Robot,
        humans : List[Human],
        position=None
        ) -> Human:
        human = Human(self.config, 'humans')
        # For fixed position
        if position:
            px, py = position
        # For random position
        else:
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                px_noise = (np.random.random() - 0.5) * v_pref
                py_noise = (np.random.random() - 0.5) * v_pref
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                collide = False
                for agent in [robot] + humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                                    norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break

        human.set(px, py, px, py, 0, 0, 0, v_pref=0)
        return human


    def generate_square_crossing_human(
        self,
        robot: Robot,
        humans: List[Human]
        )->Human:
        human = Human(self.config,'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1

        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [robot] + humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [robot] + humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break

        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def generate_circle_crossing_human(
        self,
        robot: Robot,
        humans: List[Human],
        ):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        
        radius = copy.deepcopy(self.circle_radius)
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            v_pref = 0. if human.v_pref == 0 else human.v_pref
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for i, agent in enumerate([robot] + humans):
                # keep human at least 3 meters away from robot
                if robot.kinematics == 'unicycle':
                    min_dist = self.circle_radius / 2 
                else:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                self.circle_radius = radius
                break
            else: 
                self.circle_radius -= random.random() # to prevent stucking

        # gx_noise = np.random.uniform(-v_pref/2, v_pref/2)
        # gy_noise = np.random.uniform(-v_pref/2, v_pref/2)
        # gx = -px + gx_noise
        # gy = -py + gy_noise

        human.set(px, py, -px, -py, 0, 0, 0)
        return human
    
    def generate_random_human_position(
        self, 
        robot: Robot,
        humans: List[Human],
        human_num: int, 
        ) -> List[Human]:
        """
        Generate human position: generate start position on a circle, goal position is at the opposite side
        :param human_num:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if self.config.env_config.sim.test_scenario == 'circle_crossing':
            for i in range(human_num):
                humans.append(self.generate_circle_crossing_human(robot, humans))
        elif self.config.env_config.sim.test_scenario == 'square_crossing':
            for i in range(human_num):
                humans.append(self.generate_square_crossing_human(robot, humans))
        return humans
    
    def check_collision_group(
        self,
        humans:List[Human], 
        pos:Tuple[float, float], 
        radius:float,
        )->bool:
        # check circle groups
        for r, x, y in self.circle_groups:
            if np.linalg.norm([pos[0] - x, pos[1] - y]) <= r + radius + 2 * 0.5: # use 0.5 because it's the max radius of human
                return True

        for human in humans:
            if human.isObstacle:
                pass
            else:
                if np.linalg.norm([pos[0] - human.px, pos[1] - human.py]) <= human.radius + radius:
                    return True
        return False

    def check_collision_group_goal(
        self, 
        pos:Tuple[float, float], 
        radius: float,
        ) -> bool:
        collision = False
        # check circle groups
        for r, x, y in self.circle_groups:
            if np.linalg.norm([pos[0] - x, pos[1] - y]) <= r + radius + 4 * 0.5: # use 0.5 because it's the max radius of human
                collision = True
        return collision


    def generate_robot_humans(
        self, 
        robot: Robot,
        phase:str, 
        human_num: Optional[int]=None, 
        ) -> Tuple [Robot, List[Human]]: # it will be used in env reset
        humans = []
        if human_num is None:
            human_num = self.human_num
        # for Group environment
        
        if self.group_human:
            # set the robot in a dummy far away location to avoid collision with humans
            robot.set(10, 10, 10, 10, 0, 0, np.pi / 2)

            # generate humans
            self.circle_groups = []
            humans_left = human_num

            while humans_left > 0:
                if humans_left <= 5:
                    if phase in ['train', 'val']:
                        humans = self.generate_random_human_position(robot, 
                                                                     humans,
                                                                     human_num=humans_left)
                    else:
                        humans = self.generate_random_human_position(robot,
                                                                     humans,
                                                                     human_num=humans_left)
                    humans_left = 0
                else:
                    if humans_left < 8:
                        max_rand = humans_left
                    else:
                        max_rand = 8
                    circum_num = np.random.randint(4, max_rand)
                    humans = self.generate_circle_group_obstacle(robot,
                                                                humans, 
                                                                circum_num)
                    humans_left -= circum_num

            # randomize starting position and goal position while keeping the distance of goal to be > 6
            # set the robot on a circle with radius 5.5 randomly
            rand_angle = np.random.uniform(0, np.pi * 2)
            increment_angle = 0.0
            while True:
                px_r = np.cos(rand_angle + increment_angle) * 5.5
                py_r = np.sin(rand_angle + increment_angle) * 5.5
                # check whether the initial px and py collides with any human
                collision = self.check_collision_group(humans,
                                                       (px_r, py_r), 
                                                       robot.radius)
                # if the robot goal does not fall into any human groups, the goal is okay, otherwise keep generating the goal
                if not collision:
                    break
                increment_angle = increment_angle + 0.2

            increment_angle = increment_angle + np.pi # start at opposite side of the circle
            while True:
                gx = np.cos(rand_angle + increment_angle) * 5.5
                gy = np.sin(rand_angle + increment_angle) * 5.5
                # check whether the goal is inside the human groups
                collision = self.check_collision_group_goal((gx, gy), 
                                                            robot.radius)
                if not collision:
                    break
                increment_angle = increment_angle + 0.2
      
            # vx = np.random.uniform(-self.config.env_config.robot.v_pref, self.config.env_config.robot.v_pref)
            # vy = np.sqrt(self.config.env_config.robot.v_pref**2-vx**2)
            robot.set(px_r, py_r, gx, gy, 0., 0., np.pi / 2)

        # for FoV environment
        else:
            while True:
                # Make the goal closer to the center so that the robot encounters obstructed agents more often
                if self.goal_pose == 'center':
                    start_circle = 5
                    px, py = np.random.uniform(-start_circle, start_circle, 2)
                    gx, gy = 0, 0 

                elif self.goal_pose == 'far_away':
                    start_circle = 5
                    px, py, gx, gy = np.random.uniform(-self.circle_radius, self.circle_radius, 4)

                elif self.goal_pose == 'random':
                    rand = np.random.random()
                    if rand >= 0.6:
                        px, py, gx, gy = np.random.uniform(-self.circle_radius, self.circle_radius, 4)
                    else:
                        start_circle = 5
                        px, py = np.random.uniform(-start_circle, start_circle, 2)
                        gx, gy = 0, 0

                if np.linalg.norm([px - gx, py - gy]) >= 6:
                    break
            # vx = np.random.uniform(-self.config.env_config.robot.v_pref, self.config.env_config.robot.v_pref)
            # vy = np.sqrt(self.config.env_config.robot.v_pref**2-vx**2)

            circle_radius = self.circle_radius * \
                min(robot.v_pref * 5, 1) * (1 + np.random.random() * 2)
            if circle_radius > 9:
                circle_radius = 9
            robot.set(
                0, -circle_radius, 0, circle_radius, 0, 0, np.pi / 2)
            # generate humans
            humans = self.generate_random_human_position(robot, humans, human_num=human_num)
        humans = self.sort_humans(robot, humans)
        return robot, humans
    
    def sort_humans(
        self,
        robot: Robot,
        humans: List[Human]
        ):
                ## sorting human by descending_order for make Occupancy 
        distances = []
        for human in humans:
            dist = np.linalg.norm([human.px- robot.px, human.py - robot.py])
            distances.append(dist)
        descending_order = np.array(distances).argsort()
        humans = np.array(humans)[descending_order]
        humans = humans.tolist()
        return humans 
    
    def step_agent(
        self, 
        action,
        humans:List[Human], 
        sim_map: Map,
        robot:Robot, 
        ) -> Tuple[List[Human], Robot]:
        humans = self.sort_humans(robot, humans)
        human_actions = self.get_human_actions(robot, humans, sim_map.obstacle_vertices)
        for i, human_action in enumerate(human_actions):
            humans[i].step(human_action)
        robot.step(action) 
        return humans, robot

    def update_human_goals_randomly(
        self,
        robot: Robot,
        humans: List[Human],
        )->List[Human]:
        # Update humans' goals randomly
        for human in humans:
            if human.isObstacle or human.v_pref == 0:
                continue
            if np.random.random() <= self.goal_change_chance:
                if not self.group_human: # to improve the runtime
                    humans_copy = []
                    for h in humans:
                        if h != human:
                            humans_copy.append(h)
                # Produce valid goal for human in case of circle setting
                while True:
                    angle = np.random.random() * np.pi * 2
                    # add some noise to simulate all the possible cases robot could meet with human
                    v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                    gx_noise = (np.random.random() - 0.5) * v_pref
                    gy_noise = (np.random.random() - 0.5) * v_pref
                    gx = self.circle_radius * np.cos(angle) + gx_noise
                    gy = self.circle_radius * np.sin(angle) + gy_noise
                    collide = False

                    if self.group_human:
                        collide = self.check_collision_group(humans,(gx, gy), human.radius)
                    else:
                        for agent in [robot] + humans_copy:
                            min_dist = human.radius + agent.radius + self.discomfort_dist
                            if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                                    norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                                collide = True
                                break
                    if not collide:
                        break
                # Give human new goal
                human.gx = gx
                human.gy = gy
        return humans
    
    def update_robot_goal(
        self, 
        robot: Robot
        ) -> Robot:
        current_goalx = robot.gx
        current_goaly = robot.gy
        
        while True:
            # Make the goal closer to the center so that the robot encounters obstructed agents more often
        
            start_circle = 5
          
            gx, gy = np.random.uniform(-start_circle, start_circle, 2)

            if np.linalg.norm([current_goalx - gx, current_goaly - gy]) >= 6:
                break
        robot.gx = gx
        robot.gy = gy 
        return robot
    
    def update_human_goal(
        self, 
        robot: Robot,
        human: Human,
        humans: List[Human]
        ) -> Human:
        # Update human's goals randomly
        if np.random.random() <= self.end_goal_change_chance:
            if not self.group_human:
                humans_copy = []
                for h in humans:
                    if h != human:
                        humans_copy.append(h)

            # Update human's radius now that it's reached goal
            if self.random_radii:
                human.radius += np.random.uniform(-0.1, 0.1)

            # Update human's v_pref now that it's reached goal
            if self.random_v_pref:
                if human.v_pref == 0.0 :
                    human.v_pref += np.random.uniform(self.step_v_pref, self.a_max_pref)
                    human.v_pref = np.clip(human.v_pref, a_min = self.a_min_pref, a_max = self.a_max_pref)
                else:
                    human.v_pref += np.random.uniform(-self.step_v_pref, self.step_v_pref)
                    human.v_pref = np.clip(human.v_pref, a_min = self.a_min_pref, a_max = self.a_max_pref) 

            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                gx_noise = (np.random.random() - 0.5) * v_pref
                gy_noise = (np.random.random() - 0.5) * v_pref
                gx = self.circle_radius * np.cos(angle) + gx_noise
                gy = self.circle_radius * np.sin(angle) + gy_noise
                collide = False
                if self.group_human:
                    collide = self.check_collision_group(humans,(gx, gy), human.radius)
                else:
                    for agent in [robot] + humans_copy:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                        if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                                norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                            collide = True
                            break
                if not collide:
                    break

            # Give human new goal
            human.gx = gx
            human.gy = gy
        return human 
    
    def get_human_actions(
        self,
        robot: Robot,  
        humans: List[Human], 
        global_map: List, 
        ) -> List:
        # step all humans
        human_actions = []  # a list of all humans' actions
        for i, human in enumerate(humans):
            # observation for humans is always coordinates
            ob = []
            for other_human in humans:
                if other_human != human:
                    # Chance for one human to be blind to some other humans
                    if self.random_unobservability and i == 0:
                        if np.random.random() <= self.unobservable_chance or not self.VisManager.detect_visible(human,
                                                                                                    other_human):
                            ob.append(self.dummy_human.get_observable_state())
                        else:
                            ob.append(other_human.get_observable_state())

                    elif self.VisManager.detect_visible(human, other_human):
                        ob.append(other_human.get_observable_state())
                    else:
                        ob.append(self.dummy_human.get_observable_state())

            if robot.visible:
                # Chance for one human to be blind to robot
                if self.random_unobservability and i == 0:
                    if np.random.random() <= self.unobservable_chance or not self.VisManager.detect_visible(human, robot):
                        ob += [self.dummy_robot.get_observable_state()]
                    else:
                        ob += [robot.get_observable_state()]
                # Else human will always see visible robots
                elif self.VisManager.detect_visible(human, robot):
                    ob += [robot.get_observable_state()]
                else:
                    ob += [self.dummy_robot.get_observable_state()]

            human_actions.append(human.act(ob, global_map)) ## only think ORCA human
        return human_actions

    # def randomize_human_policies(
    #     self,
    #     humans: List[Human]
    #     )->List[Human]:
    #     """
    #     Randomize the moving humans' policies to be either orca or social force
    #     """
    #     for human in humans:
    #         if not human.isObstacle:
    #             new_policy = random.choice(['orca','social_force'])
    #             new_policy = policy_factory[new_policy](self.config)
    #             human.set_policy(new_policy)
    #     return humans 