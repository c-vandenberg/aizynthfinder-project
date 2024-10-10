import os

import yaml
from trainers.trainer import Trainer
from models.utils import Seq2SeqModelUtils

def main():
    # Path to the configuration file
    config_path = 'config/training/model_v15_config.yml'

    # Initialize the Trainer with the configuration
    trainer = Trainer(config_path=config_path)

    config = trainer.load_config(config_path)
    model_save_path = config['training']['model_save_path']
    model_save_dir = config['training']['model_save_dir']

    Seq2SeqModelUtils.convert_saved_model_to_hdf5(
        model_save_path,
        os.path.join(model_save_dir, 'hdf5')
    )

    test = 'test'

if __name__ == '__main__':
    main()