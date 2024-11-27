from typing import List

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data.utils.tokenization import SmilesTokenizer


class SmilesDataPreprocessor:
    """
    Preprocesses tokenized SMILES strings into padded sequences of integers for model training and evaluation.

    This class leverages a provided `SmilesTokenizer` to convert tokenized SMILES strings into
    integer sequences and applies padding to ensure uniform sequence lengths across the dataset.

    Attributes
    ----------
    smiles_tokenizer : SmilesTokenizer
        The tokenizer used for converting SMILES strings to integer sequences.
    max_seq_length : int
        The maximum sequence length for padding sequences.

    Methods
    -------
    preprocess_smiles(tokenized_smiles_list)
        Converts a list of tokenized SMILES strings into a tensor of padded integer sequences.
    """
    def __init__(
        self,
        smiles_tokenizer: SmilesTokenizer,
        max_seq_length: int
    ) -> None:
        if not isinstance(smiles_tokenizer, SmilesTokenizer):
            raise TypeError("smiles_tokenizer must be an instance of SmilesTokenizer.")
        if not isinstance(max_seq_length, int) or max_seq_length <= 0:
            raise ValueError("max_seq_length must be a positive integer.")

        self.smiles_tokenizer = smiles_tokenizer
        self.max_seq_length = max_seq_length

    def preprocess_smiles(self, tokenized_smiles_list: List[str]) -> tf.Tensor:
        """
        Converts a list of tokenized SMILES strings into a tensor of padded integer sequences.

        This method transforms each tokenized SMILES string into a sequence of integers using the
        provided tokenizer. It then pads or truncates these sequences to ensure they all have the
        same length (`max_seq_length`).

        Padding is applied post-sequence to maintain the start of the sequence intact, which is
        crucial for models such as seq2seq models that predict the next token in a sequence.

        Parameters
        ----------
        tokenized_smiles_list : List[str]
            A list of tokenized SMILES strings. Each string should be a space-separated sequence of tokens.

        Returns
        -------
        tf.Tensor
            A tensor of shape `(num_sequences, max_seq_length)` containing the padded integer sequences.
            Each row corresponds to a padded sequence of integers representing a SMILES string.

        Raises
        ------
        ValueError
            If `tokenized_smiles_list` is empty.
            If any of the SMILES strings in `tokenized_smiles_list` are not strings.
        """
        if not tokenized_smiles_list:
            raise ValueError("tokenized_smiles_list must contain at least one SMILES string.")
        if not all(isinstance(smiles, str) for smiles in tokenized_smiles_list):
            raise ValueError("All elements in tokenized_smiles_list must be strings.")

        sequences: tf.Tensor = self.smiles_tokenizer.texts_to_sequences(tokenized_smiles_list)

        # Pad sequences
        padded_sequences: tf.Tensor = pad_sequences(
            sequences,
            maxlen=self.max_seq_length,
            padding='post',
            truncating='post'
        )
        return tf.constant(padded_sequences, dtype=tf.int32)
