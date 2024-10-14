from typing import Dict, List, Union, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer

class SmilesTokenizer:
    """
    Tokenizer for SMILES strings.

    This tokenizer converts SMILES strings into individual character tokens,
    adds optional start and end tokens, and uses a Keras Tokenizer internally
    to convert tokens to indices.

    Parameters
    ----------
    start_token : str, optional
        Token representing the start of a sequence (default is '<START>').
    end_token : str, optional
        Token representing the end of a sequence (default is '<END>').
    oov_token : str, optional
        Token representing out-of-vocabulary tokens (default is '<OOV>').

    Attributes
    ----------
    _start_token : str
        Start token.
    _end_token : str
        End token.
    oov_token : str
        Out-of-vocabulary token.

    Methods
    -------
    tokenize(smiles)
        Tokenizes a single SMILES string.
    tokenize_list(smiles_list)
        Tokenizes a list of SMILES strings.
    create_tokenizer(tokenized_smiles_list)
        Creates and fits the internal tokenizer on the tokenized SMILES list.
    texts_to_sequences(texts)
        Converts tokenized texts to sequences of token indices.
    sequences_to_texts(sequences)
        Converts sequences of token indices back to texts.
    """
    def __init__(
        self,
        start_token: str = '<START>',
        end_token: str = '<END>',
        oov_token: str = '<OOV>',
        max_sequence_tokens: int = 140,
        reverse_input_sequence: bool = False
    ) -> None:
        self._start_token = start_token
        self._end_token = end_token
        self.oov_token = oov_token
        self.reverse_input_sequence = reverse_input_sequence

        # Initialize TextVectorization layer
        self.text_vectorization = TextVectorization(
            standardize=None,
            split='whitespace',
            output_mode='int',
            output_sequence_length=None,
            max_tokens=max_sequence_tokens,
            pad_to_max_tokens=False,
        )

    @property
    def start_token(self):
        return self._start_token

    @property
    def end_token(self):
        return self._end_token

    def tokenize(self, smiles: str, is_input_sequence: bool) -> str:
        """
        Tokenizes a single SMILES string into individual characters.

        Parameters
        ----------
        smiles : str
            A SMILES string.
        is_input_sequence : bool, optional
            Is input SMILES string boolean.

        Returns
        -------
        List[str]
            A list of character tokens with start and end tokens.
        """
        basic_smiles_tokenizer = BasicSmilesTokenizer()
        tokenized_smiles = basic_smiles_tokenizer.tokenize(smiles)

        # Add start and end tokens
        tokens = [self.start_token] + tokenized_smiles + [self.end_token]

        # Reverse the SMILES tokens if required (excluding special tokens)
        if self.reverse_input_sequence and is_input_sequence:
            # Reverse only the SMILES tokens, keep start and end tokens in place
            tokens = [self.start_token] + tokenized_smiles[::-1] + [self.end_token]

        # Join tokens back into a string separated by spaces (required for TextVectorization)
        return ' '.join(tokens)

    def tokenize_list(self, smiles_list: List[str], is_input_sequence = False) -> List[List[str]]:
        """
        Tokenizes a list of SMILES strings.

        Parameters
        ----------
        smiles_list : List[str]
            A list of SMILES strings.
        is_input_sequence : bool, optional
            Boolean declaring whether SMILES list is sequence input or not.

        Returns
        -------
        List[List[str]]
            A list of token lists.
        """
        return [self.tokenize(smiles, is_input_sequence) for smiles in smiles_list]

    def adapt(self, tokenized_smiles_list: List[str]) -> None:
        """
        Adapts the TextVectorization layer to the preprocessed SMILES list.

        Parameters
        ----------
        tokenized_smiles_list : List[str]
            A list of preprocessed SMILES strings.
        """
        # Convert list to TensorFlow dataset
        self.text_vectorization.adapt(tf.constant(tokenized_smiles_list))

    def texts_to_sequences(self, texts: List[str]) -> tf.Tensor:
        """
        Converts tokenized texts to sequences of token indices.

        Parameters
        ----------
        texts : List[str]
            List of tokenized texts.

        Returns
        -------
        tf.Tensor
            A tensor of token indices.
        """
        return self.text_vectorization(tf.constant(texts))

    def sequences_to_texts(self, sequences: Union[tf.Tensor, np.ndarray]) -> List[str]:
        """
        Converts sequences of token indices back to texts.

        Parameters
        ----------
        sequences : Union[tf.Tensor, np.ndarray]
            A tensor or NumPy array of token indices.

        Returns
        -------
        List[str]
            A list of SMILES strings.
        """
        # Create a reverse mapping from indices to tokens
        vocab = self.text_vectorization.get_vocabulary()
        inverse_vocab = {idx: word for idx, word in enumerate(vocab)}
        texts = []

        # Convert tf.Tensor to NumPy array if necessary
        if isinstance(sequences, tf.Tensor):
            sequences = sequences.numpy()
        elif not isinstance(sequences, np.ndarray):
            raise TypeError("Input must be a tf.Tensor or np.ndarray")

        for sequence in sequences:
            tokens = [inverse_vocab.get(idx, self.oov_token) for idx in sequence if idx != 0]
            # Remove start and end tokens
            if tokens and tokens[0] == self.start_token:
                tokens = tokens[1:]
            if tokens and tokens[-1] == self.end_token:
                tokens = tokens[:-1]
            # Reverse back if original was reversed
            if self.reverse_input_sequence:
                tokens = tokens[::-1]
            texts.append(''.join(tokens))
        return texts

    @property
    def word_index(self) -> Dict[str, int]:
        """
        Gets the word index mapping.

        Returns
        -------
        dict
            Word to index mapping.
        """
        vocab = self.text_vectorization.get_vocabulary()
        return {word: idx for idx, word in enumerate(vocab)}

    @property
    def vocab_size(self):
        """
        Gets the size of the vocabulary.

        Returns
        -------
        int
            Vocabulary size.
        """
        return len(self.text_vectorization.get_vocabulary())
