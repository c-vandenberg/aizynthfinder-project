import logging

from trainers.trainer import Trainer

def main():
    logging.basicConfig(level=logging.INFO)

    config_path = 'config/training/model_v28_config.yml'

    trainer = Trainer(config_path=config_path)

    trainer.run()

if __name__ == "__main__":
    main()
