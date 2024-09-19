import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataPreprocessor:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def preprocess_smiles(self, tokenized_smiles_list):
        """
        Preprocess tokenized SMILES strings into padded sequences of integers.

        Args:
            tokenized_smiles_list (List[List[str]]): A list of tokenized SMILES strings.

        Returns:
            tf.Tensor: A tensor of padded sequences.
        """
        # Convert tokens to sequences of integers
        sequences = self.tokenizer.texts_to_sequences(tokenized_smiles_list)

        # Pad sequences
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_seq_length,
            padding='post',
            truncating='post'
        )
        return tf.constant(padded_sequences, dtype=tf.int32)