import os
import logging

import hydra
import importlib
import os
from pathlib import Path
import shutil

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

def create_artifact_folder(path):
    artifact_path = Path(path) / "artifacts"
    if os.path.exists(str(artifact_path)):
        shutil.rmtree(str(artifact_path))

    os.makedirs(artifact_path)
    return

@hydra.main(config_name="config", config_path = ".", version_base = None)
def go(config):

    if config["mode"] == "remote":
        os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
        os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    root_path = hydra.utils.get_original_cwd()

    steps = config["main"]["steps"]
    to_run = steps.split(",") if steps != "all" else config["components"].keys()

    create_artifact_folder('components')
    for component, params in config["components"].items():
        if component in to_run:
            logger.info(f"\n====> Running component: {component}\n")
            module_name = 'components.'+component
            module = importlib.import_module(module_name)
            func = getattr(module,'go')
            func(params)
            # alt way to use subprocesses
            # subprocess.call([f'./components/{component}.py', config['mode']] + list(params.values()))


if __name__ == "__main__":
    go()