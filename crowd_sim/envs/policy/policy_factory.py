from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.network_policies.example import Example 



def none_policy():
    return None

policy_factory = dict()
policy_factory['LINEAR'] = Linear
policy_factory['ORCA'] = ORCA
policy_factory['Example'] = Example
policy_factory['NONE'] = none_policy
