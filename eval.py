from pogema import BatchAStarAgent
from pogema_toolbox.create_env import create_env_base, Environment
from pogema_toolbox.evaluator import evaluation

from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results

from pathlib import Path
import wandb

import yaml

from pogema_toolbox.registry import ToolboxRegistry

from agents.assistant_switcher import AssistantSwitcher, ASwitcherConfig
from agents.epom import EPOM, EpomConfig
from agents.heuristic_switcher import HeuristicSwitcher, HSwitcherConfig
from agents.learnable_switcher import LSwitcherConfig, LearnableSwitcher
from agents.replan import RePlan, RePlanConfig

PROJECT_NAME = 'pogema-toolbox'
BASE_PATH = Path('experiments')


def main(disable_wandb=True):
    ToolboxRegistry.register_env('Pogema-v0', create_env_base, Environment)

    ToolboxRegistry.register_algorithm('ASwitcher', AssistantSwitcher, ASwitcherConfig)
    ToolboxRegistry.register_algorithm('HSwitcher', HeuristicSwitcher, HSwitcherConfig)
    ToolboxRegistry.register_algorithm('LSwitcher', LearnableSwitcher, LSwitcherConfig)

    ToolboxRegistry.register_algorithm('EPOM', EPOM, EpomConfig)
    ToolboxRegistry.register_algorithm('RePlan', RePlan, RePlanConfig)

    ToolboxRegistry.register_algorithm('A*', BatchAStarAgent)

    with open("pomapf_env/maps.yaml", 'r') as f:
        maps_to_register = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps_to_register)

    with open("pomapf_env/test-maps.yaml", 'r') as f:
        maps_to_register = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps_to_register)

    folder_names = [
        '01-random-20x20',
    ]

    for folder in folder_names:
        config_path = BASE_PATH / folder / f"{Path(folder).name}.yaml"
        eval_dir = BASE_PATH / folder

        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)
        if folder == 'eval-fast':
            disable_wandb = True

        initialize_wandb(evaluation_config, eval_dir, disable_wandb, PROJECT_NAME)
        evaluation(evaluation_config, eval_dir=eval_dir)
        save_evaluation_results(eval_dir)
        wandb.finish()


if __name__ == '__main__':
    main()
