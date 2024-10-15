import os
import pydevd_pycharm

import tensorflow as tf

from trainers.trainer import Trainer

from data.utils.tokenization import SmilesTokenizer

pydevd_pycharm.settrace('localhost', port=63342, stdoutToServer=True, stderrToServer=True, suspend=False)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def main():
    original_smiles = "COC(=O)CCC(=O)c1ccc(OC2CCCCO2)cc1O"  # Benzene
    tokenizer = SmilesTokenizer(max_sequence_tokens=140, reverse_input_sequence=True)
    tokenized = tokenizer.tokenize(original_smiles, is_input_sequence=True)
    tokenizer.adapt(tokenized)
    sequences = tokenizer.texts_to_sequences([tokenized])
    reconstructed = tokenizer.sequences_to_texts(sequences)
    assert original_smiles == reconstructed[0], "Mismatch in tokenization/detokenization"

    # Path to the configuration file
    config_path = 'config/training/model_v18_config.yml'

    # Initialize the Trainer with the configuration
    trainer = Trainer(config_path=config_path)

    # Run the training pipeline
    trainer.run()
if __name__ == "__main__":
    main()