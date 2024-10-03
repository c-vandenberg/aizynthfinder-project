from tensorflow.keras.preprocessing.text import Tokenizer


class SmilesTokenizer(Tokenizer):
    def __init__(
        self,
        start_token='<START>',
        end_token='<END>'
    ) -> None:
        self.start_token = start_token
        self.end_token = end_token
        self.tokenizer = None  # Will be initialized later if needed

    def tokenize(self, smiles):
        """
        Tokenize a single SMILES string into individual characters.

        Args:
            smiles (str): A SMILES string.

        Returns:
            List[str]: A list of character tokens.
        """
        tokens = [self.start_token] + list(smiles) + [self.end_token]
        return tokens

    def tokenize_list(self, smiles_list):
        """
        Tokenize a list of SMILES strings.

        Args:
            smiles_list (List[str]): A list of SMILES strings.

        Returns:
            List[List[str]]: A list of token lists.
        """
        return [self.tokenize(smiles) for smiles in smiles_list]

    def create_tokenizer(self, tokenized_smiles_list):
        """
        Create and fit a Keras tokenizer on the tokenized SMILES list.

        Args:
            tokenized_smiles_list (List[List[str]]): A list of tokenized SMILES strings.
        """

        self.tokenizer = Tokenizer(filters='', lower=False, char_level=False, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(tokenized_smiles_list)
        return self.tokenizer

    def texts_to_sequences(self, texts):
        """
        Convert texts to sequences of token indices.

        Args:
            texts (List[List[str]]): List of tokenized texts.

        Returns:
            List[List[int]]: Sequences of token indices.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been created. Call 'create_tokenizer' first.")
        return self.tokenizer.texts_to_sequences(texts)

    def sequences_to_texts(self, sequences):
        """
        Convert sequences of token indices back to texts.

        Args:
            sequences (List[List[int]]): Sequences of token indices.

        Returns:
            List[str]: List of texts.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been created. Call 'create_tokenizer' first.")
        return self.tokenizer.sequences_to_texts(sequences)

    @property
    def word_index(self):
        """
        Get the word index mapping.

        Returns:
            Dict[str, int]: Word to index mapping.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been created. Call 'create_tokenizer' first.")
        return self.tokenizer.word_index

    @property
    def vocab_size(self):
        """
        Get the size of the vocabulary.

        Returns:
            int: Vocabulary size.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been created. Call 'create_tokenizer' first.")
        # +1 because Keras Tokenizer reserves index 0
        return len(self.tokenizer.word_index) + 1
