import os
from typing import List, Tuple, Optional

import tensorflow as tf
from rdkit import Chem
from sklearn.model_selection import train_test_split

from data.utils.tokenization import SmilesTokenizer
from data.utils.preprocessing import SmilesDataPreprocessor


class DataLoader:
    """
    DataLoader is responsible for loading, preprocessing, and managing
    datasets for training, validation, and testing of SMILES-based Retrosynthesis models.

    Attributes
    ----------
    DEFAULT_MAX_SEQ_LENGTH : int
        Default maximum sequence length for encoder and decoder.
    DEFAULT_BATCH_SIZE : int
        Default batch size for training.
    DEFAULT_BUFFER_SIZE : int
        Default buffer size for shuffling datasets.
    DEFAULT_TEST_SIZE : float
        Default proportion of the dataset to include in the test split.
    DEFAULT_RANDOM_STATE : int
        Default random state for reproducibility.
    DEFAULT_MAX_TOKENS : int
        Default maximum number of tokens.
    DEFAULT_REVERSE_INPUT_SEQ_BOOL : bool
        Default flag to reverse input sequences.

    Methods
    -------
    vocab_size:
        Returns the size of the tokenizer's vocabulary.
    load_and_prepare_data():
        Loads and preprocesses the datasets.
    get_dataset(data, training=True):
        Creates a TensorFlow dataset from the given data.
    get_train_dataset():
        Returns the training dataset.
    get_valid_dataset():
        Returns the validation dataset.
    get_test_dataset():
        Returns the test dataset.
    _load_smiles_from_file(file_path):
        Loads SMILES strings from a specified file.
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
        products_valid_file: str,
        reactants_valid_file: str,
        num_samples: Optional[int] = None,
        max_encoder_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        max_decoder_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
        reverse_input_sequence: bool = DEFAULT_REVERSE_INPUT_SEQ_BOOL
    ) -> None:
        """
        Initializes the DataLoader with file paths and configuration parameters.

        Parameters
        ----------
        products_file : str
            Path to the file containing product SMILES strings.
        reactants_file : str
            Path to the file containing reactant SMILES strings.
        products_valid_file : str
            Path to the validation file for product SMILES strings.
        reactants_valid_file : str
            Path to the validation file for reactant SMILES strings.
        num_samples : Optional[int], default=None
            Number of samples to load. If None, all samples are loaded.
        max_encoder_seq_length : int, default=DEFAULT_MAX_SEQ_LENGTH
            Maximum sequence length for the encoder.
        max_decoder_seq_length : int, default=DEFAULT_MAX_SEQ_LENGTH
            Maximum sequence length for the decoder.
        batch_size : int, default=DEFAULT_BATCH_SIZE
            Size of batches for training.
        buffer_size : int, default=DEFAULT_BUFFER_SIZE
            Buffer size for shuffling datasets.
        test_size : float, default=DEFAULT_TEST_SIZE
            Proportion of the dataset to include in the test split.
        random_state : int, default=DEFAULT_RANDOM_STATE
            Seed for random number generators to ensure reproducibility.
        reverse_input_sequence : bool, default=DEFAULT_REVERSE_INPUT_SEQ_BOOL
            Whether to reverse input sequences before tokenization.
        """
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

        self._smiles_tokenizer = SmilesTokenizer(
            reverse_input_sequence=reverse_input_sequence
        )

        self.encoder_data_processor = None
        self.decoder_data_processor = None

        self.train_data = None
        self.valid_data = None
        self.test_data = None

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the tokenizer's vocabulary.

        Returns
        -------
        self.smiles_tokenizer.vocab_size : int
            Vocabulary size.
        """
        return self.smiles_tokenizer.vocab_size

    @property
    def smiles_tokenizer(self) -> 'SmilesTokenizer':
        """
        Returns the SMILES tokenizer instance.

        Returns
        -------
        self._smiles_tokenizer : SmilesTokenizer
            The SMILES tokenizer used for tokenizing SMILES strings.
        """
        return self._smiles_tokenizer

    def load_and_prepare_data(self) -> None:
        """
        Loads datasets from files, tokenizes them, splits into training and
        test sets, and preprocesses the data for model consumption.

        This method orchestrates the entire data preparation pipeline, including
        loading raw SMILES strings, tokenizing, splitting into training/testing
        sets, and preprocessing for model input.

        Returns
        -------
            None
        """
        self._load_datasets()
        self._tokenize_datasets()
        self._split_datasets()
        self._preprocess_datasets()

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
        self.products_x_dataset: List[str] = self._load_smiles_from_file(self.products_file)
        self.reactants_y_dataset: List[str] = self._load_smiles_from_file(self.reactants_file)
        self.products_x_valid_dataset: List[str] = self._load_smiles_from_file(self.products_valid_file)
        self.reactants_y_valid_dataset: List[str] = self._load_smiles_from_file(self.reactants_valid_file)

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

        # Canonicalize SMILES strings
        self.products_x_dataset = [self._canonicalize_smiles(smi) for smi in self.products_x_dataset]
        self.reactants_y_dataset = [self._canonicalize_smiles(smi) for smi in self.reactants_y_dataset]
        self.products_x_valid_dataset = [self._canonicalize_smiles(smi) for smi in self.products_x_valid_dataset]
        self.reactants_y_valid_dataset = [self._canonicalize_smiles(smi) for smi in self.reactants_y_valid_dataset]

    def _tokenize_datasets(self) -> None:
        """
        Tokenizes the datasets using the SMILES tokenizer for training
        validation and testing datasets.

        Returns
        -------
            None
        """
        self.tokenized_products_x_dataset: List[str] = self.smiles_tokenizer.tokenize_list(
            self.products_x_dataset,
            is_input_sequence=True
        )
        self.tokenized_reactants_y_dataset = self.smiles_tokenizer.tokenize_list(
            self.reactants_y_dataset,
            is_input_sequence=False
        )

        self.tokenized_products_x_valid_dataset = self.smiles_tokenizer.tokenize_list(
            self.products_x_valid_dataset,
            is_input_sequence=True
        )
        self.tokenized_reactants_y_valid_dataset = self.smiles_tokenizer.tokenize_list(
            self.reactants_y_valid_dataset,
            is_input_sequence=False
        )

    def _split_datasets(self) -> None:
        """
        Splits the datasets into training and test sets.

        Utilizes scikit-learn's `train_test_split()` method to partition the tokenized
        product and reactant datasets into training and testing subsets based on
        the specified test size and random state. Adapts the tokenizer only on
        the training data to prevent data leakage.

        Returns
        -------
            None

        Raises
        ------
        ValueError
            If the tokenized datasets are empty or invalid.
        """
        (self.tokenized_products_x_train_data, self.tokenized_products_x_test_data,
         self.tokenized_reactants_y_train_data, self.tokenized_reactants_y_test_data) = train_test_split(
            self.tokenized_products_x_dataset,
            self.tokenized_reactants_y_dataset,
            test_size=self.test_size,
            random_state=self.random_state
        )

        combined_tokenized_train_data = self.tokenized_products_x_train_data + self.tokenized_reactants_y_train_data
        self.smiles_tokenizer.adapt(combined_tokenized_train_data)

    def _preprocess_datasets(self) -> None:
        """
        Preprocesses the training, validation, and test datasets.

        Initializes data preprocessors for encoder and decoder, and applies
        preprocessing to the respective datasets to prepare them for model training
        and evaluation.

        Returns
        -------
            None
        """
        # Initialize DataPreprocessors
        self.encoder_data_processor = SmilesDataPreprocessor(
            self.smiles_tokenizer,
            self.max_encoder_seq_length
        )
        self.decoder_data_processor = SmilesDataPreprocessor(
            self.smiles_tokenizer,
            self.max_decoder_seq_length
        )

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

    def _preprocess_data_pair(
        self,encoder_data: List[str],
        decoder_data: List[str]
    ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Preprocesses a pair of encoder and decoder data.

        Parameters
        ----------
        encoder_data : List[str]
            Tokenized encoder data.
        decoder_data : List[str]
            Tokenized decoder data.

        Returns
        -------
        (encoder_input, decoder_input), decoder_target : Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]
            A tuple containing:
                - A tuple of encoder input and decoder input tensors.
                - Decoder target tensor.
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

        Converts the preprocessed data into a `tf.data.Dataset` object, applies
        shuffling, batching, and prefetching as per the configuration.

        Parameters
        ----------
        data : Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]
            The data to create the dataset from.
        training : bool, optional
            Whether the dataset is for training (default is True). If True, shuffling
            and dropping the remainder of batches is applied to ensure consistent
            batch sizes.

        Returns
        -------
        dataset : tf.data.Dataset
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

        Utilizes the `get_dataset` method with training-specific configurations.

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

        Utilizes the `get_dataset` method with validation-specific configurations.

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

        Utilizes the `get_dataset` method with test-specific configurations.

        Returns
        -------
        tf.data.Dataset
            The test dataset.
        """
        if self.test_data is None:
            raise ValueError("Test data has not been loaded and preprocessed.")

        return self.get_dataset(self.test_data, training=False)

    @staticmethod
    def _load_smiles_from_file(file_path: str) -> List[str]:
        """
        Loads SMILES strings from a specified file.

        Reads a file containing SMILES strings, ensuring that each line is
        properly stripped of whitespace and non-empty.

        Parameters
        ----------
        file_path : str
            The path to the file containing SMILES strings.

        Returns
        -------
        List[str]
            A list of SMILES strings.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, 'r') as file:
            smiles_list = [line.strip() for line in file if line.strip()]

        return smiles_list

    @staticmethod
    def _canonicalize_smiles(smiles: str) -> str:
        """
        Canonicalises SMILES strings using `rdkit.Chem.MolFromSmiles()`.

        Correctly handles reactant SMILES separated by `.` by canonicalising separately and re-concatenating the
        reassembling in the same order by concatenating with a `.` separator.

        Parameters
        ----------
        smiles : str
            The single SMILES string or multiple, `.` separated SMILES strings to canonicalise.

        Returns
        -------
        str
        str
             The single canonicalised SMILES string or multiple, `.` separated canonicalised SMILES strings.

        Raises
        ------
        ValueError
            If any of the SMILES components are invalid.
        """
        # Split the SMILES string on '.' to get individual components
        smiles_components = smiles.split('.')
        canonical_components = []
        for smi in smiles_components:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
            canonical_smi = Chem.MolToSmiles(mol, canonical=True)
            canonical_components.append(canonical_smi)

        # Reassemble the components in the same order, separated by '.'
        canonical_smiles = '.'.join(canonical_components)
        return canonical_smiles
