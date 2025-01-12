#!/usr/bin/env python3

import os
import pydevd_pycharm
import argparse

import tensorflow as tf

from trainers.trainer import Trainer

# pydevd_pycharm.settrace('localhost', port=63342, stdoutToServer=True, stderrToServer=True, suspend=False)

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train seq2seq model via configuration file.")
    parser.add_argument(
        '--training_config_filepath',
        type=str,
        required=True,
        help='Path to training configuration file.'
    )

    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Initialize the Trainer with the configuration
    trainer = Trainer(config_path=args.training_config_filepath)

    # Run the training pipeline
    trainer.run()

if __name__ == "__main__":
    main()
