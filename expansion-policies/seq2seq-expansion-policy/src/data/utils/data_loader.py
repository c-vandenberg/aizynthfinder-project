import logging
from typing import List, Tuple, Optional

import tensorflow as tf
from sklearn.model_selection import train_test_split

from data.utils.tokenisation import SmilesTokeniser
from data.utils.preprocessing import TokenisedSmilesPreprocessor, SmilesDataPreprocessor
from data.utils.file_utils import load_smiles_from_file


class DataLoader:
    """
    Responsible for loading, preprocessing, and managing datasets for training, validation,
    and testing of SMILES-based Retrosynthesis models.

    N.B. The sum of `train_split`, `test_split`, and `validation_split` must equal 1.

    Methods
    -------
    vocab_size
        Returns the size of the tokeniser's vocabulary.
    smiles_tokeniser
        Returns the SMILES tokeniser instance.
    load_and_prepare_data()
        Loads and preprocesses the datasets.
    get_dataset(data, training=True)
        Creates a TensorFlow dataset from the given data.
    get_train_dataset()
        Returns the training dataset.
    get_valid_dataset()
        Returns the validation dataset.
    get_test_dataset()
        Returns the test dataset.
    """
    DEFAULT_MAX_SEQ_LENGTH = 140
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_BUFFER_SIZE = 10000
    DEFAULT_TEST_SIZE = 0.3
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_MAX_TOKENS = 150
    DEFAULT_REVERSE_INPUT_SEQ_BOOL = True

    def __init__(
        self,
        products_file: str,
        reactants_file: str,
        test_split: float,
        validation_split: float,
        logger: logging.Logger,
        num_samples: Optional[int] = None,
        max_encoder_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        max_decoder_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
        reverse_input_sequence: bool = DEFAULT_REVERSE_INPUT_SEQ_BOOL
    ) -> None:
        self.products_file = products_file
        self.reactants_file = reactants_file
        self.test_split = test_split
        self.validation_split = validation_split
        self._logger = logger
        self.num_samples = num_samples
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.random_state = random_state

        total_split = self.test_split + self.validation_split
        if not (0 < self.test_split < 1 and 0 < self.validation_split < 1):
            raise ValueError("test_split and validation_split must be between 0 and 1.")
        if not (0 < total_split < 1):
            raise ValueError("The sum of test_split and validation_split must be between 0 and 1.")

        self.train_split = 1.0 - total_split

        self._smiles_tokeniser = SmilesTokeniser(
            logger=self._logger,
            reverse_input_sequence=reverse_input_sequence
        )

        self.encoder_data_processor = None
        self.decoder_data_processor = None

        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.test_dataset_size = None

        self.token_counts = None

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the tokeniser's vocabulary.

        Returns
        -------
        int
            Vocabulary size.
        """
        return self.smiles_tokeniser.vocab_size

    @property
    def smiles_tokeniser(self) -> 'SmilesTokeniser':
        """
        Returns the SMILES tokeniser instance.

        Returns
        -------
        SmilesTokeniser
            The SMILES tokeniser used for tokenising SMILES strings.
        """
        return self._smiles_tokeniser

    def load_and_prepare_data(self) -> None:
        """
        Loads datasets from files, tokenises them, splits into training and
        test sets, and preprocesses the data for model consumption.

        This method orchestrates the entire data preparation pipeline, including
        loading raw SMILES strings, tokenising, splitting into training/testing
        sets, and preprocessing for model input.

        Returns
        -------
        None
        """
        self._load_datasets()
        self._tokenise_datasets()
        self._split_datasets()
        self._preprocess_tokenised_datasets()

    def _load_datasets(self) -> None:
        """
        Loads the datasets from the provided file paths.

        Reads SMILES strings from product and reactant files for both training
        and validation datasets. Ensures that the loaded datasets have matching
        lengths and optionally limits the number of samples.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If any of the specified files do not exist.
        ValueError
            If there is a mismatch in the lengths of the datasets.
        """
        self.products_x_dataset: List[str] = load_smiles_from_file(self.products_file)
        self.reactants_y_dataset: List[str] = load_smiles_from_file(self.reactants_file)

        # Ensure datasets have the same length
        if len(self.products_x_dataset) != len(self.reactants_y_dataset):
            raise ValueError("Mismatch in dataset lengths.")

        # Limit the number of samples if specified
        if self.num_samples is not None:
            self.products_x_dataset = self.products_x_dataset[:self.num_samples]
            self.reactants_y_dataset = self.reactants_y_dataset[:self.num_samples]

        # Canonicalise SMILES strings
        smiles_preprocessor: SmilesDataPreprocessor = SmilesDataPreprocessor()
        self.products_x_dataset = [
            smiles_preprocessor.canonicalise_smiles(smi) for smi in self.products_x_dataset
        ]
        self.reactants_y_dataset = [
            smiles_preprocessor.canonicalise_smiles(smi) for smi in self.reactants_y_dataset
        ]

    def _tokenise_datasets(self) -> None:
        """
        Tokenises the datasets using the SMILES tokeniser for training
        validation and testing datasets.

        Returns
        -------
            None
        """
        self.tokenised_products_x_dataset: List[str] = self.smiles_tokeniser.tokenise_list(
            self.products_x_dataset,
            is_input_sequence=True
        )
        self.tokenised_reactants_y_dataset = self.smiles_tokeniser.tokenise_list(
            self.reactants_y_dataset,
            is_input_sequence=False
        )

    def _split_datasets(self) -> None:
        """
        Splits the datasets into training, testing, and validation sets.

        Utilises scikit-learn's `train_test_split()` method to partition the tokenised
        product and reactant datasets into training, testing, and validation subsets based on
        the specified split ratios and random state. Adapts the tokeniser only on
        the training data to prevent data leakage.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the tokenised datasets are empty or invalid.
        """
        # First split: Train vs Temp (Test + Validation)
        (self.tokenised_products_x_train_data, self.tokenised_products_x_temp_data,
         self.tokenised_reactants_y_train_data, self.tokenised_reactants_y_temp_data) = train_test_split(
            self.tokenised_products_x_dataset,
            self.tokenised_reactants_y_dataset,
            test_size=(self.test_split + self.validation_split),
            random_state=self.random_state,
            shuffle=True
        )

        # Calculate the proportion of validation split relative to the temp data
        validation_ratio = self.validation_split / (self.test_split + self.validation_split)

        # Second split: Validation vs Test from Temp
        (self.tokenised_products_x_valid_data, self.tokenised_products_x_test_data,
         self.tokenised_reactants_y_valid_data, self.tokenised_reactants_y_test_data) = train_test_split(
            self.tokenised_products_x_temp_data,
            self.tokenised_reactants_y_temp_data,
            test_size=(1-validation_ratio),
            random_state=self.random_state,
            shuffle=True
        )

        # Store test dataset size for partial test evaluation
        self.test_size = len(self.tokenised_products_x_test_data)

        # Adapt the tokeniser on the training data only
        combined_tokenised_train_data = self.tokenised_products_x_train_data + self.tokenised_reactants_y_train_data
        self.smiles_tokeniser.adapt(combined_tokenised_train_data)

        # Extract token frequencies
        self.token_counts = self._smiles_tokeniser.get_token_frequencies(combined_tokenised_train_data)

    def _preprocess_tokenised_datasets(self) -> None:
        """
        Preprocesses the tokenised training, validation, and test datasets.

        Initialises data preprocessors for encoder and decoder, and applies
        preprocessing to the respective datasets to prepare them for model training
        and evaluation.

        Returns
        -------
        None
        """
        # Initialise DataPreprocessors
        self.encoder_data_processor = TokenisedSmilesPreprocessor(
            self.smiles_tokeniser,
            self.max_encoder_seq_length
        )
        self.decoder_data_processor = TokenisedSmilesPreprocessor(
            self.smiles_tokeniser,
            self.max_decoder_seq_length
        )

        # Preprocess training data
        self.train_data = self._preprocess_data_pair(
            self.tokenised_products_x_train_data,
            self.tokenised_reactants_y_train_data
        )

        # Preprocess test data
        self.test_data = self._preprocess_data_pair(
            self.tokenised_products_x_test_data,
            self.tokenised_reactants_y_test_data
        )

        # Preprocess validation data
        self.valid_data = self._preprocess_data_pair(
            self.tokenised_products_x_valid_data,
            self.tokenised_reactants_y_valid_data
        )

    def _preprocess_data_pair(
        self,encoder_data: List[str],
        decoder_data: List[str]
    ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Preprocesses a pair of encoder and decoder data.

        Parameters
        ----------
        encoder_data : list of str
            Tokenised encoder data.
        decoder_data : list of str
            Tokenised decoder data.

        Returns
        -------
        tuple
            A tuple containing:
            - (encoder_input, decoder_input) : tuple of tf.Tensor
                Encoder and decoder input tensors.
            - decoder_target : tf.Tensor
                Decoder target tensor.
        """
        encoder_input = self.encoder_data_processor.preprocess_smiles(encoder_data)
        decoder_full = self.decoder_data_processor.preprocess_smiles(decoder_data)
        decoder_input = decoder_full[:, :-1]
        decoder_target = decoder_full[:, 1:]
        return (encoder_input, decoder_input), decoder_target

    def get_dataset(
        self,
        data: Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor],
        training: bool =True
    ) -> tf.data.Dataset:
        """
        Creates a TensorFlow dataset from the given data.

        Converts the preprocessed data into a `tf.data.Dataset` object and applies
        shuffling, batching, and prefetching as per the configuration.

        Parameters
        ----------
        data : tuple
            A tuple ((encoder_input, decoder_input), decoder_target) where each element is a tf.Tensor.
        training : bool, optional
            Whether the dataset is for training. If True, shuffling and dropping the remainder
            of batches is applied to ensure consistent batch sizes. Default is True.

        Returns
        -------
        tf.data.Dataset
            A TensorFlow dataset ready for model consumption.
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
        """
        Returns the training dataset.

        Utilises the `get_dataset` method with training-specific configurations.

        Returns
        -------
        tf.data.Dataset
            The training dataset.
        """
        if self.train_data is None:
            raise ValueError("Training data has not been loaded and preprocessed.")

        return self.get_dataset(self.train_data, training=True)

    def get_valid_dataset(self) -> tf.data.Dataset:
        """
        Returns the validation dataset.

        Utilises the `get_dataset` method with validation-specific configurations.

        Returns
        -------
        tf.data.Dataset
            The validation dataset.
        """
        if self.valid_data is None:
            raise ValueError("Validation data has not been loaded and preprocessed.")

        return self.get_dataset(self.valid_data, training=False)

    def get_test_dataset(self) -> tf.data.Dataset:
        """
        Returns the test dataset.

        Utilises the `get_dataset` method with test-specific configurations.

        Returns
        -------
        tf.data.Dataset
            The test dataset.
        """
        if self.test_data is None:
            raise ValueError("Test data has not been loaded and preprocessed.")

        return self.get_dataset(self.test_data, training=False)
