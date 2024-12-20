#!/usr/bin/env python3

import os
import pydevd_pycharm
import argparse

import tensorflow as tf

from trainers.trainer import Trainer

pydevd_pycharm.settrace('localhost', port=63342, stdoutToServer=True, stderrToServer=True, suspend=False)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

tf.config.run_functions_eagerly(True)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train seq2seq model via configuration file.")
    parser.add_argument('--training_config_file_path', type=str, required=True, help='Path to training configuration file.')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    training_config_file_path: str = args.training_config_file_path

    # Initialize the Trainer with the configuration
    trainer = Trainer(config_path=training_config_file_path)

    # Run the training pipeline
    trainer.run()

if __name__ == "__main__":
    main()
