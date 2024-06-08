from pogema import pogema_v0
from pogema import AnimationMonitor, AnimationConfig
from pogema.integrations.sample_factory import AutoResetWrapper

from pomapf_env.wrappers import RewardShaping, MultiMapWrapper


def make_pomapf(grid_config, with_animations=False):
    grid_config.auto_reset = False
    env = pogema_v0(grid_config)
    if with_animations:
        env = AnimationMonitor(env, AnimationConfig(egocentric_idx=0))
    env = RewardShaping(env)

    env = MultiMapWrapper(env)
    env = AutoResetWrapper(env)
    return env
