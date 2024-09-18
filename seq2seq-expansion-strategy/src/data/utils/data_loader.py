import tensorflow as tf
import numpy as np
from typing import Tuple


class DataLoader:
    def __init__(self, encoder_inputs, decoder_inputs, decoder_targets, batch_size: int, buffer_size: int = 10000):
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def get_dataset(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((
            (self.encoder_inputs, self.decoder_inputs),
            self.decoder_targets
        ))
        dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_train_valid_datasets(self, validation_split=0.1) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        total_samples = len(self.encoder_inputs)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        valid_size = int(total_samples * validation_split)
        train_indices = indices[valid_size:]
        valid_indices = indices[:valid_size]

        train_dataset = tf.data.Dataset.from_tensor_slices((
            (self.encoder_inputs[train_indices], self.decoder_inputs[train_indices]),
            self.decoder_targets[train_indices]
        )).shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        valid_dataset = tf.data.Dataset.from_tensor_slices((
            (self.encoder_inputs[valid_indices], self.decoder_inputs[valid_indices]),
            self.decoder_targets[valid_indices]
        )).batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        return train_dataset, valid_dataset

    def load_smiles_data_from_csv(self, csv_file):
        """
        Load SMILES data from a CSV file.

        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            List[str]: A list of SMILES strings.
        """
        import pandas as pd

        df = pd.read_csv(csv_file)

        smiles_col = self.smiles_column.lower()

        if smiles_col not in df.columns.str.lower():
            raise ValueError(f"Column '{self.smiles_column}' not found in the CSV file.")

        # Drop NaN values and convert to a list
        smiles_list = df[self.smiles_column].dropna().tolist()

        return smiles_list
