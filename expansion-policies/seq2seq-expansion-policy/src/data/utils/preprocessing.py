from typing import List

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from data.utils.tokenization import SmilesTokenizer


class DataPreprocessor:
    """
    Preprocesses tokenized SMILES strings into padded sequences of integers.

    Parameters
    ----------
    tokenizer : Tokenizer
        An instance of a tokenizer that can convert text to sequences.
    max_seq_length : int
        The maximum sequence length for padding.

    Methods
    -------
    preprocess_smiles(tokenized_smiles_list)
        Converts tokenized SMILES strings into padded integer sequences.
    """
    def __init__(
        self,
        smiles_tokenizer: SmilesTokenizer,
        tokenizer: Tokenizer,
        max_seq_length: int
    ) -> None:
        self.smiles_tokenizer = smiles_tokenizer
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def preprocess_smiles(self, tokenized_smiles_list: List[List[str]]) -> tf.Tensor:
        """
        Converts tokenized SMILES strings into padded sequences of integers.

        Parameters
        ----------
        tokenized_smiles_list : List[List[str]]
            A list of tokenized SMILES strings.

        Returns
        -------
        tf.Tensor
            A tensor of padded sequences of shape (num_sequences, max_seq_length).
        """
        sequences = self.tokenizer.texts_to_sequences(tokenized_smiles_list)

        # Pad sequences
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_seq_length,
            padding='post',
            truncating='post'
        )
        return tf.constant(padded_sequences, dtype=tf.int32)
