from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from omegaconf import OmegaConf

class Human(Agent):
    def __init__(self,  config: OmegaConf, section: str = 'humans'):
        super().__init__( config, section)
        self.id = None
        self.isObstacle = False # whether the human is a static obstacle (part of wall) or a moving agent

    def act(self, ob=None, global_map=None, local_map=None):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        if ob is None:
            return self.policy.predict(self)
        state = JointState(self.get_full_state(), ob)
        if global_map is not None:
            action = self.policy.predict(state, global_map, self)
        elif local_map is not None:
            action = self.policy.predict(state, local_map, self)
        else:
            action = self.policy.predict(state)
        return action
