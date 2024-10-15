#!/usr/bin/env python3

import os
import pydevd_pycharm

import tensorflow as tf

from trainers.trainer import Trainer

pydevd_pycharm.settrace('localhost', port=63342, stdoutToServer=True, stderrToServer=True, suspend=False)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

tf.config.run_functions_eagerly(True)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def main():

    # Path to the configuration file
    config_path = 'config/training/model_v18_config.yml'

    # Initialize the Trainer with the configuration
    trainer = Trainer(config_path=config_path)

    # Run the training pipeline
    trainer.run()
if __name__ == "__main__":
    main()