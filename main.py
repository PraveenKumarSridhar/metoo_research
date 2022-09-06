import os
import logging

import hydra
import importlib

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

@hydra.main(config_name="config", config_path = ".", version_base = None)
def go(config):

    if config["mode"] == "remote":
        os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
        os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    root_path = hydra.utils.get_original_cwd()

    steps = config["main"]["steps"]
    to_run = steps.split(",") if steps != "all" else config["components"].keys()

    for component, params in config["components"].items():
        if component in to_run:
            logger.info(f"\n====> Running component: {component}\n")
            # print(params)
            module_name = 'components.'+component
            module = importlib.import_module(module_name)
            func = getattr(module,'go')
            func(params)
            # alt way to use subprocesses
            # subprocess.call([f'./components/{component}.py', config['mode']] + list(params.values()))


if __name__ == "__main__":
    go()