from trainers.trainer import Trainer

def main():
    # Path to the configuration file
    config_path = 'config/training/model_v14_config.yml'

    # Initialize the Trainer with the configuration
    trainer = Trainer(config_path=config_path)

    # Run the training pipeline
    trainer.run()

if __name__ == '__main__':
    main()