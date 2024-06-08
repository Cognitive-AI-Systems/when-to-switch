from abc import abstractmethod
from typing import Optional

import numpy as np
from pogema_toolbox.algorithm_config import AlgoBase
from pydantic import BaseModel, Extra

from agents.epom import EpomConfig, EPOM
from agents.replan import RePlanConfig, RePlan


class SwitcherBaseConfig(AlgoBase, extra=Extra.forbid):
    planning: RePlanConfig = RePlanConfig(fix_loops=True, add_none_if_loop=True, no_path_random=False,
                                          use_best_move=False, fix_nones=False)
    learning: EpomConfig = EpomConfig()
    num_process: int = 10
    clear_hidden_after_switch: bool = False
    seed: Optional[int] = 0


class SwitcherBase:
    def __init__(self, algo_cfg):
        self.cfg = algo_cfg

        self.planning = RePlan(algo_cfg.planning)
        self.learning = EPOM(algo_cfg.learning)

        self.learning_used = 0
        self.planning_used = 0

        self._rnd = np.random.RandomState(seed=algo_cfg.seed)

    @abstractmethod
    def get_learning_use_mask(self, planning_actions, learning_actions, observations):
        raise NotImplementedError

    def act(self, observations, rewards=None, dones=None, infos=None):
        if infos is None:
            infos = [{'is_active': True} for _ in range(len(observations))]
        planning = self.planning.act(observations, rewards, dones, infos)
        learning = self.learning.act(observations, rewards, dones, infos)
        masks = self.get_learning_use_mask(planning_actions=planning, learning_actions=learning,
                                           observations=observations)

        self.update_usage(masks, infos, dones)
        if self.cfg.clear_hidden_after_switch:
            for agent_idx, mask in enumerate(masks):
                if not mask:
                    self.learning.clear_hidden(agent_idx)

        return [learning[idx] if masks[idx] else planning[idx] for idx, _ in enumerate(masks)]

    def after_step(self, dones):
        self.planning.after_step(dones)
        self.learning.after_step(dones)

    def after_reset(self):
        self.planning.after_reset()
        self.learning.after_reset()

        self._rnd = np.random.RandomState(seed=self.cfg.seed)

    def reset_states(self):
        self.after_reset()

    def update_usage(self, mask, infos, dones):

        for idx, info in enumerate(infos):
            if not info['is_active']:
                continue
            if mask[idx]:
                self.learning_used += 1
            else:
                self.planning_used += 1
