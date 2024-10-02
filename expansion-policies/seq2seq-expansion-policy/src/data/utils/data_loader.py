import os
from typing import List, Tuple, Optional

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.types.data import DatasetV2

from data.utils.tokenization import SmilesTokenizer
from data.utils.preprocessing import DataPreprocessor

class DataLoader:
    DEFAULT_MAX_SEQ_LENGTH = 150
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_BUFFER_SIZE = 10000
    DEFAULT_TEST_SIZE = 0.3
    DEFAULT_RANDOM_STATE = 42

    def __init__(
        self,
        products_file: str,
        reactants_file: str,
        products_valid_file: str,
        reactants_valid_file: str,
        num_samples: Optional[int] = None,
        max_encoder_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        max_decoder_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        self.products_file = products_file
        self.reactants_file = reactants_file
        self.products_valid_file = products_valid_file
        self.reactants_valid_file = reactants_valid_file
        self.num_samples = num_samples
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.test_size = test_size
        self.random_state = random_state

        self.smiles_tokenizer = SmilesTokenizer()
        self._tokenizer = None

        self.encoder_data_processor = None
        self.decoder_data_processor = None

        self.train_data = None
        self.valid_data = None
        self.test_data = None

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.tokenizer.word_index) + 1  # +1 for padding token

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise ValueError("Tokenizer has not been created yet.")
        return self._tokenizer

    def load_and_prepare_data(self) -> None:
        """Loads and preprocesses the datasets."""
        self._load_datasets()
        self._tokenize_datasets()
        self._split_datasets()
        self._create_tokenizer()
        self._preprocess_datasets()

    def _load_datasets(self) -> None:
        """Loads the datasets from the provided file paths."""
        self.products_x_dataset = self._load_smiles_from_file(self.products_file)
        self.reactants_y_dataset = self._load_smiles_from_file(self.reactants_file)
        self.products_x_valid_dataset = self._load_smiles_from_file(self.products_valid_file)
        self.reactants_y_valid_dataset = self._load_smiles_from_file(self.reactants_valid_file)

        # Ensure datasets have the same length
        if len(self.products_x_dataset) != len(self.reactants_y_dataset):
            raise ValueError("Mismatch in dataset lengths.")
        if len(self.products_x_valid_dataset) != len(self.reactants_y_valid_dataset):
            raise ValueError("Mismatch in validation dataset lengths.")

        # Limit the number of samples if specified
        if self.num_samples is not None:
            self.products_x_dataset = self.products_x_dataset[:self.num_samples]
            self.reactants_y_dataset = self.reactants_y_dataset[:self.num_samples]
            self.products_x_valid_dataset = self.products_x_valid_dataset[:self.num_samples]
            self.reactants_y_valid_dataset = self.reactants_y_valid_dataset[:self.num_samples]

        # Reverse the source SMILES strings before tokenization to prevent data leakage
        self.products_x_dataset = [smiles[::-1] for smiles in self.products_x_dataset]
        self.products_x_valid_dataset = [smiles[::-1] for smiles in self.products_x_valid_dataset]

    def _tokenize_datasets(self) -> None:
        """Tokenizes the datasets using the SMILES tokenizer."""
        self.tokenized_products_x_dataset = self.smiles_tokenizer.tokenize_list(self.products_x_dataset)
        self.tokenized_reactants_y_dataset = self.smiles_tokenizer.tokenize_list(self.reactants_y_dataset)
        self.tokenized_products_x_valid_dataset = self.smiles_tokenizer.tokenize_list(self.products_x_valid_dataset)
        self.tokenized_reactants_y_valid_dataset = self.smiles_tokenizer.tokenize_list(self.reactants_y_valid_dataset)

    def _split_datasets(self) -> None:
        """Splits the datasets into training and test sets."""
        (self.tokenized_products_x_train_data, self.tokenized_products_x_test_data,
         self.tokenized_reactants_y_train_data, self.tokenized_reactants_y_test_data) = train_test_split(
            self.tokenized_products_x_dataset,
            self.tokenized_reactants_y_dataset,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def _create_tokenizer(self) -> None:
        """Creates and fits the tokenizer on the training data."""
        train_tokenized_smiles = self.tokenized_products_x_train_data + self.tokenized_reactants_y_train_data
        self._tokenizer = self.smiles_tokenizer.create_tokenizer(train_tokenized_smiles)

        # Initialize DataPreprocessors
        self.encoder_data_processor = DataPreprocessor(self.tokenizer, self.max_encoder_seq_length)
        self.decoder_data_processor = DataPreprocessor(self.tokenizer, self.max_decoder_seq_length)

    def _preprocess_datasets(self) -> None:
        """Preprocesses the training, validation, and test datasets."""
        # Preprocess training data
        self.train_data = self._preprocess_data_pair(
            self.tokenized_products_x_train_data,
            self.tokenized_reactants_y_train_data
        )

        # Preprocess test data
        self.test_data = self._preprocess_data_pair(
            self.tokenized_products_x_test_data,
            self.tokenized_reactants_y_test_data
        )

        # Preprocess validation data
        self.valid_data = self._preprocess_data_pair(
            self.tokenized_products_x_valid_dataset,
            self.tokenized_reactants_y_valid_dataset
        )

    def _preprocess_data_pair(self,encoder_data: List[str],
                              decoder_data: List[str]) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """Preprocesses a pair of encoder and decoder data.

        Parameters
        ----------
        encoder_data : List[str]
            Tokenized encoder data.
        decoder_data : List[str]
            Tokenized decoder data.

        Returns
        -------
        Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]
            A tuple containing:
                - A tuple of encoder input and decoder input tensors.
                - Decoder target tensor.
        """
        encoder_input = self.encoder_data_processor.preprocess_smiles(encoder_data)
        decoder_full = self.decoder_data_processor.preprocess_smiles(decoder_data)
        decoder_input = decoder_full[:, :-1]
        decoder_target = decoder_full[:, 1:]
        return (encoder_input, decoder_input), decoder_target

    def get_dataset(self, data: Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor],
                    training: bool =True) -> tf.data.Dataset:
        """
        Creates a tf.data.Dataset from the given data.

        Parameters
        ----------
        data : Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]
            The data to create the dataset from.
        training : bool, optional
            Whether the dataset is for training (default is True).

        Returns
        -------
        tf.data.Dataset
            A TensorFlow dataset.
        """
        (encoder_input, decoder_input), decoder_target = data

        dataset = tf.data.Dataset.from_tensor_slices((
            (encoder_input, decoder_input),
            decoder_target
        ))

        if training:
            dataset = dataset.shuffle(self.buffer_size)

        # Use drop_remainder=True during training to ensure consistent batch sizes
        dataset = dataset.batch(self.batch_size, drop_remainder=training)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_train_dataset(self) -> tf.data.Dataset:
        """Returns the training dataset."""
        return self.get_dataset(self.train_data, training=True)

    def get_valid_dataset(self) -> tf.data.Dataset:
        """Returns the validation dataset."""
        return self.get_dataset(self.valid_data, training=False)

    def get_test_dataset(self) -> tf.data.Dataset:
        """Returns the test dataset."""
        return self.get_dataset(self.test_data, training=False)

    @staticmethod
    def _load_smiles_from_file(file_path: str) -> List[str]:
        """Loads SMILES strings from a file.

        Parameters
        ----------
        file_path : str
            The path to the file containing SMILES strings.

        Returns
        -------
        List[str]
            A list of SMILES strings.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, 'r') as file:
            smiles_list = [line.strip() for line in file if line.strip()]
        return smiles_list
