import functools
import os
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
)
import gymnasium as gym
import numpy as np
from stable_baselines3.common import monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from drl_utils.algorithms.crowd_common.vec_env.envs import TimeLimitMask, VecPyTorch
from drl_utils.algorithms.crowd_common.vec_env.shmem_vec_env import ShmemVecEnv
from drl_utils.algorithms.crowd_common.vec_env.dummy_vec_env import DummyVecEnv as DumVecEnv
from drl_utils.algorithms.crowd_common.vec_env.monitor import Monitor
from omegaconf import OmegaConf

from imitation.util.util import make_seeds

def make_vec_env(
    env_name: str,
    *,
    rng: np.random.Generator,
    n_envs: int = 8,
    parallel: bool = False,
    log_dir: Optional[str] = None,
    phase : str = 'train',
    max_episode_steps: Optional[int] = None,
    post_wrappers: Optional[Sequence[Callable[[gym.Env, int], gym.Env]]] = None,
    env_make_kwargs: Optional[Mapping[str, Any]] = None,
    config:OmegaConf,
) -> VecEnv:
    tmp_env = gym.make(env_name)
    tmp_env.close()
    spec = tmp_env.spec
    env_make_kwargs = env_make_kwargs or {}

    def make_env(i: int, this_seed: int, config: OmegaConf, n_envs = 1) -> gym.Env:
        assert env_make_kwargs is not None  # Note: to satisfy mypy
        assert spec is not None  # Note: to satisfy mypy
        env = gym.make(spec, max_episode_steps=max_episode_steps, **env_make_kwargs)

        env.configure(config = config, 
                        nenv = n_envs, 
                        phase = phase, 
                        )
        ## setting CrowdWs
        
        env.reset(seed=int(this_seed+i), phase = phase)
        log_path = None
        if log_dir is not None:
            log_subdir = os.patorch.join(log_dir, "monitor")
            os.makedirs(log_subdir, exist_ok=True)
            log_path = os.patorch.join(log_subdir, f"mon{i:03d}")
        
        env = monitor.Monitor(env, log_path)

        if post_wrappers:
            for wrapper in post_wrappers:
                env = wrapper(env, i)
        return env

    env_seeds = make_seeds(rng, n_envs)

    env_fns: List[Callable[[], gym.Env]] = [
        functools.partial(make_env, i, s, config, n_envs) for i, s in enumerate(env_seeds)
    ]

    if parallel:
        envs = SubprocVecEnv(env_fns, start_method="forkserver")
        return VecNormalize(envs, 
                    #  training=True if phase in ['train'] else False, 
                     norm_obs=True,
                     norm_obs_keys=['robotstate_obs','ranges'],
                     norm_reward=False,
                     gamma=config.env_config.reward.gamma
                     )
    else:
        envs = DummyVecEnv(env_fns)
        return VecNormalize(envs, 
                    #  training=True if phase in ['train'] else False, 
                     norm_obs=True,
                     norm_obs_keys=['robotstate_obs','ranges'],
                     norm_reward=False,
                     gamma=config.env_config.reward.gamma
                     )

def make_env(
        env_name, 
        seed, 
        rank, 
        allow_early_resets, 
        max_episode_steps, 
        envConfig=None, 
        envNum=1, 
        phase=None):

    def _thunk():
        env = gym.make(env_name, max_episode_steps=max_episode_steps)
        env.configure(envConfig, nenv = envNum, phase = phase)

        envSeed = seed + rank if seed is not None else None
        env.reset(seed=int(envSeed), phase = phase)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        env = Monitor(
            env,
            None,
            allow_early_resets=allow_early_resets)
        return env
    return _thunk


def make_vec_envs(env_name: str,
                  device,
                  rng: np.random.Generator,
                  allow_early_resets: bool,
                  n_envs: int = 8,
                  phase : str = 'train',
                  max_episode_steps: Optional[int] = None,
                  config=None,
                  ):
    env_seeds = make_seeds(rng, n_envs)
    envs = [
        make_env(env_name=env_name, seed = seed, rank = i, allow_early_resets= allow_early_resets, envConfig=config,
                 envNum=n_envs,phase=phase, max_episode_steps = max_episode_steps)
        for i, seed in enumerate(env_seeds)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DumVecEnv(envs)

    envs = VecPyTorch(envs, device)
    return envs
