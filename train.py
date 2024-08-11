from drl_utils.env_utils.make_env import make_vec_env
from drl_utils.algorithms.extractors.example_extractor import ExampleExtractor
from stable_baselines3.ppo import PPO
from omegaconf import OmegaConf
import torch as th
import numpy as np 
import pathlib
import hydra
import os
from typing import Deque
from crowd_sim.envs.utils.info import *

MODEL_NAME = 'My_Alogrithm'
DEVICE = 'cuda:0'

class TrainWorkSpace():
    def __init__(self, config: OmegaConf):
        
        th.manual_seed(config.default.seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(config.default.seed)
            th.cuda.manual_seed(config.default.seed)
        rng = np.random.default_rng(config.default.seed)

        self.policy_kwargs = dict(
                        features_extractor_class = ExampleExtractor,
                        features_extractor_kwargs = dict(features_dim=128),
                        net_arch=[dict(pi=[128,128], vf=[128, 64])],
                        optimizer_class= th.optim.AdamW,
                        share_features_extractor = True, 
                        )

        vecenv = make_vec_env(
                    env_name=f'{config.default.env_name}',
                    n_envs = 1,
                    rng = rng,
                    parallel = True, 
                    max_episode_steps = int(config.system.time_limit/ config.system.time_step),
                    config = config,
                    phase='train',
                    )  
        
        ppo = PPO(
                    policy_kwargs = self.policy_kwargs,
                    policy = 'MultiInputPolicy',
                    env = vecenv,
                    n_steps = 1024,
                    batch_size = 256,
                    learning_rate = 2.5e-4,
                    tensorboard_log = os.path.join(os.getcwd(), f'tensorboard_log/{MODEL_NAME}')
                    )
        
        ppo.learn(total_timesteps=100_000) 
        ppo.save(os.path.join(os.getcwd(), f'model_weight/{MODEL_NAME}'))

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'crowd_nav', 'configs')),
    config_name='base_config.yaml'
)

def main(cfg):
    TrainWorkSpace(cfg)

if __name__ == "__main__":
    main()
