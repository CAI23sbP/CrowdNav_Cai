import warnings
import gymnasium as gym
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import trange 
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, is_vecenv_wrapped, SubprocVecEnv
"""
mimic stable-baselines3
"""
from crowd_sim.envs.utils.info import Collision, ReachGoal, Timeout
from typing import Deque
from collections import deque

def callback_info(infos, dones, info_deque: Deque):
    if dones:
        for info in infos:
            if isinstance(info['info'], Collision):
                info_deque.append('Collision')
            elif isinstance(info['info'], Timeout):
                info_deque.append('Timeout')
            elif isinstance(info['info'], ReachGoal):
                info_deque.append('ReachGoal')
                
def evaluate_policy(
    model: Union["type_aliases.PolicyPredictor"]  ,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    warn: bool = True,
    phase:str ='test',
    test_case: int = -1 ,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )
    info_deque : Deque = deque(maxlen=n_eval_episodes)
    n_envs = env.num_envs
    if n_envs >1:
        raise ValueError('only one env are able to test.')
    
    for episode in trange(n_eval_episodes):
        states = None 
        is_done = False
        if test_case>-1:
            env.set_attr('test_case', test_case , 0)
        else:
            env.set_attr('test_case', episode, 0)

        observations = env.reset()
        
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        while not is_done:
            actions, states = model.predict(
                observations,  
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
                
            if render and phase in ['test', 'val']:
                env.render()
            new_observations, rewards, dones, infos = env.step(actions)
            observations = new_observations
            if dones:
                is_done = True
                
            if callback:
                callback(infos , is_done, info_deque)
                

    collision_cnt = info_deque.count('Collision')
    timeout_cnt = info_deque.count('Timeout')
    reachgoal_cnt = info_deque.count('ReachGoal')
    info_dict = {
                f'Total_{phase}': n_eval_episodes,
                'Success Rate' : round((reachgoal_cnt/n_eval_episodes)*100,2),
                'Collision Rate': round((collision_cnt/n_eval_episodes)*100,2),
                'TimeOut Rate': round((timeout_cnt/n_eval_episodes)*100,2),
                }
    return info_dict








