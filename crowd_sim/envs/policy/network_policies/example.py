import numpy as np

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from omegaconf import OmegaConf

class Example(Policy):
	def __init__(self, config: OmegaConf):
		super().__init__(config)
		self.name = 'Example'
		self.trainable = True
		self.multiagent_training = True
		self.kinematics = config.env_config.robot.kinematics
		self.v_pref = config.env_config.robot.v_pref
		self.time_step = config.env_config.sim.time_step

	# clip the self.raw_action and return the clipped action
	def clip_action(self, raw_action):
		action = ActionRot(raw_action[0], raw_action[1])
		return action 


