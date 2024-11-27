import os
import pydevd_pycharm

import tensorflow as tf

from trainers.trainer import Trainer

def main():
    config_path = 'config/training/model_v28_config.yml'

    trainer = Trainer(config_path=config_path)

    trainer.run()

if __name__ == "__main__":
    main()
