from typing import Dict, List, Optional

from tensorflow.keras.preprocessing.text import Tokenizer


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
    start_token : str
        Start token.
    end_token : str
        End token.
    oov_token : str
        Out-of-vocabulary token.
    tokenizer : Tokenizer, optional
        Keras Tokenizer instance used for converting tokens to sequences.

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
        oov_token: str = '<OOV>'
    ) -> None:
        self.start_token = start_token
        self.end_token = end_token
        self.oov_token = oov_token
        self.tokenizer: Optional[Tokenizer] = None

    def tokenize(self, smiles: str) -> List[str]:
        """
        Tokenizes a single SMILES string into individual characters.

        Parameters
        ----------
        smiles : str
            A SMILES string.

        Returns
        -------
        List[str]
            A list of character tokens with start and end tokens.
        """
        tokens = [self.start_token] + list(smiles) + [self.end_token]
        return tokens

    def tokenize_list(self, smiles_list: List[str]) -> List[List[str]]:
        """
        Tokenizes a list of SMILES strings.

        Parameters
        ----------
        smiles_list : List[str]
            A list of SMILES strings.

        Returns
        -------
        List[List[str]]
            A list of token lists.
        """
        return [self.tokenize(smiles) for smiles in smiles_list]

    def create_tokenizer(self, tokenized_smiles_list: List[List[str]]) -> Tokenizer:
        """
        Creates and fits a Keras Tokenizer on the tokenized SMILES list.

        Parameters
        ----------
        tokenized_smiles_list : List[List[str]]
            A list of tokenized SMILES strings.

        Returns
        -------
        Tokenizer
            The fitted Keras Tokenizer instance.
        """

        self.tokenizer = Tokenizer(
            filters='',
            lower=False,
            char_level=False,
            oov_token=self.oov_token
        )
        self.tokenizer.fit_on_texts(tokenized_smiles_list)
        return self.tokenizer

    def texts_to_sequences(self, texts: List[List[str]]) -> List[List[int]]:
        """
        Converts tokenized texts to sequences of token indices.

        Parameters
        ----------
        texts : List[List[str]]
            List of tokenized texts.

        Returns
        -------
        List[List[int]]
            Sequences of token indices.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been created. Call 'create_tokenizer' first.")
        return self.tokenizer.texts_to_sequences(texts)

    def sequences_to_texts(self, sequences: List[List[int]]) -> List[str]:
        """
        Converts sequences of token indices back to texts.

        Parameters
        ----------
        sequences : List[List[int]]
            Sequences of token indices.

        Returns
        -------
        List[str]
            List of texts.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been created. Call 'create_tokenizer' first.")
        return self.tokenizer.sequences_to_texts(sequences)

    @property
    def word_index(self) -> Dict[str, int]:
        """
        Gets the word index mapping.

        Returns
        -------
        dict
            Word to index mapping.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been created. Call 'create_tokenizer' first.")
        return self.tokenizer.word_index

    @property
    def vocab_size(self):
        """
        Gets the size of the vocabulary.

        Returns
        -------
        int
            Vocabulary size.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been created. Call 'create_tokenizer' first.")
        # +1 because Keras Tokenizer reserves index 0
        return len(self.tokenizer.word_index) + 1
