#!/usr/bin/env python3
import numpy as np
import rvo2
from crowd_sim.envs.policy.policy import Policy
from omegaconf import OmegaConf
from crowd_sim.envs.utils.action import ActionRot, ActionXY

class ORCA(Policy):
    def __init__(self, config:OmegaConf):
        super().__init__(config)
        self.name = 'ORCA'
        self.radius = None
        self.max_speed = 1 # the ego agent assumes that all other agents have this max speed
        self.sim = None
        self.time_step = config.env_config.sim.time_step
        self.safety_space = config.orca_config.safety_space

        self.time_horizon_obst = config.orca_config.time_horizon_obst
        self.time_horizon = config.orca_config.time_horizon
        self.neighbor_dist = config.orca_config.neighbor_dist

    def predict(self, state, global_map, agent):
        robot_state = state.robot_state
        self.max_neighbors = len(state.human_states) 
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        # create sim with static obstacles if they don't exist
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None

        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, agent.radius, agent.v_pref)
            for obstacle in global_map:
                if len(obstacle) !=0:
                    self.sim.addObstacle(obstacle)
            self.sim.processObstacles()
            self.sim.addAgent(robot_state.position, *params, robot_state.radius + 0.01 + self.safety_space,
                              robot_state.v_pref, robot_state.velocity)
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, *params, human_state.radius + 0.01 + self.safety_space,
                                  self.max_speed, human_state.velocity)
        else:
            self.sim.setAgentPosition(0, robot_state.position)
            self.sim.setAgentVelocity(0, robot_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        velocity = np.array((robot_state.gx - robot_state.px, robot_state.gy - robot_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        # for i, human_state in enumerate(human_states_in_FOV):
        for i, human_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))
        agent.last_state = state

        return action