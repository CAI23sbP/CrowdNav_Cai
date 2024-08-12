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
                     norm_obs=False,
                     norm_obs_keys=['robotstate_obs','ranges'],
                     norm_reward=False,
                     gamma=config.env_config.reward.gamma
                     )
    else:
        envs = DummyVecEnv(env_fns)
        return VecNormalize(envs, 
                     norm_obs=False,
                     norm_obs_keys=['robotstate_obs','ranges'],
                     norm_reward=False,
                     gamma=config.env_config.reward.gamma
                     )