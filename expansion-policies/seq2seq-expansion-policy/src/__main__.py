import logging
from trainers.trainer import Trainer

# Configure logging to display debug information
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("aizynthfinder")
logger.setLevel(logging.DEBUG)


def main():
    # Path to the configuration file
    config_path = 'config/training/model_v25_config.yml'

    # Initialize the Trainer with the configuration
    trainer = Trainer(config_path=config_path)

    # Run the training pipeline
    trainer.run()

if __name__ == "__main__":
    main()