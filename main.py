import os
import logging

import hydra
import importlib
import os
from pathlib import Path
import shutil

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

def create_artifact_folder(path, all_flag=True):
    artifact_path = Path(path) / "artifacts"
    if not os.path.exists(str(artifact_path)):
        logger.info(f"\Trying to create artifacts folder {artifact_path}\n")
        os.makedirs(artifact_path)
        logger.info(f"\nSuccessfully created artifacts folder\n")
    return

@hydra.main(config_name="config", config_path = ".", version_base = None)
def go(config):

    if config["mode"] == "remote":
        os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
        os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    root_path = hydra.utils.get_original_cwd()

    steps = config["main"]["steps"]
    to_run = steps.split(",") if steps != "all" else config["components"].keys()
    logger.info(f"\nGoing to run {steps}\n")
    create_artifact_folder('components', steps=='all')
    for component, params in config["components"].items():
        if component in to_run:
            logger.info(f"\n====> Running component: {component}\n")
            module_name = 'components.'+component
            module = importlib.import_module(module_name)
            func = getattr(module,'go')
            func(params)
            # alt way to use subprocesses
            # subprocess.call([f'./components/{component}.py', config['mode']] + list(params.values()))
    logger.info(f"\n====> Completed running : {steps} successfully\n")


if __name__ == "__main__":
    go()