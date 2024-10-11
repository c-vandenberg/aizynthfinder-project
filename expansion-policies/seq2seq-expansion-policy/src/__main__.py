import pydevd_pycharm

from trainers.trainer import Trainer

pydevd_pycharm.settrace('localhost', port=63342, stdoutToServer=True, stderrToServer=True)

def main():
    # Path to the configuration file
    config_path = 'config/model_v6_config.yml'

    # Initialize the Trainer with the configuration
    trainer = Trainer(config_path=config_path)

    # Run the training pipeline
    trainer.run()

if __name__ == '__main__':
    main()