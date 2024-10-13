#!/usr/bin/env python3

from trainers.trainer import Trainer

# Path to the configuration file
config_path = 'config/training/model_v17_config.yml'

# Initialize the Trainer with the configuration
trainer = Trainer(config_path=config_path)

# Run the training pipeline
trainer.run()