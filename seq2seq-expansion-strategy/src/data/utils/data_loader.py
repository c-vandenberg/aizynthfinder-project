import tensorflow as tf
from tensorflow.python.types.data import DatasetV2
from sklearn.model_selection import train_test_split
from data.utils.tokenization import SmilesTokenizer
from data.utils.preprocessing import DataPreprocessor

class DataLoader:
    def __init__(
        self,
        products_file,
        reactants_file,
        products_valid_file,
        reactants_valid_file,
        num_samples=None,
        max_encoder_seq_length=150,
        max_decoder_seq_length=150,
        batch_size=16,
        buffer_size=10000,
        test_size=0.3,
        random_state=42
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
        self.tokenizer = None
        self.vocab_size = None

        self.encoder_data_processor = None
        self.decoder_data_processor = None

        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def load_and_prepare_data(self):
        # 1. Load data
        products_x_dataset = self.load_smiles(self.products_file)
        reactants_y_dataset = self.load_smiles(self.reactants_file)
        products_x_valid_dataset = self.load_smiles(self.products_valid_file)
        reactants_y_valid_dataset = self.load_smiles(self.reactants_valid_file)

        # Ensure that the datasets have the same length
        assert len(products_x_dataset) == len(reactants_y_dataset), "Mismatch in dataset lengths."
        assert len(products_x_valid_dataset) == len(reactants_y_valid_dataset), \
            "Mismatch in validation dataset lengths."

        # Limit the number of samples if specified
        if self.num_samples is not None:
            products_x_dataset = products_x_dataset[:self.num_samples]
            reactants_y_dataset = reactants_y_dataset[:self.num_samples]
            products_x_valid_dataset = products_x_valid_dataset[:self.num_samples]
            reactants_y_valid_dataset = reactants_y_valid_dataset[:self.num_samples]

        # As we are using character-wise encoding, reverse the source (product) SMILES strings before tokenization
        products_x_dataset = [smiles[::-1] for smiles in products_x_dataset]
        products_x_valid_dataset = [smiles[::-1] for smiles in products_x_valid_dataset]

        # 2. Tokenize the datasets
        tokenized_products_x_dataset = self.smiles_tokenizer.tokenize_list(products_x_dataset)
        tokenized_reactants_y_dataset = self.smiles_tokenizer.tokenize_list(reactants_y_dataset)
        tokenized_products_x_valid_dataset = self.smiles_tokenizer.tokenize_list(products_x_valid_dataset)
        tokenized_reactants_y_valid_dataset = self.smiles_tokenizer.tokenize_list(reactants_y_valid_dataset)

        # 3. Split data into training and test datasets before preprocessing to prevent data leakage
        (tokenized_products_x_train_data, tokenized_products_x_test_data,
         tokenized_reactants_y_train_data, tokenized_reactants_y_test_data) = train_test_split(
            tokenized_products_x_dataset,
            tokenized_reactants_y_dataset,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # 4. Create the Tokenizer
        # Combine all tokenized SMILES strings to build a common tokenizer based on training data to
        # prevent data leakage
        train_tokenized_smiles = tokenized_products_x_train_data + tokenized_reactants_y_train_data

        # Create and fit the tokenizer
        self.tokenizer = self.smiles_tokenizer.create_tokenizer(train_tokenized_smiles)
        self.vocab_size = len(self.tokenizer.word_index) + 1  # +1 for padding token

        # 4. Initialize DataPreprocessors and preprocess data
        self.encoder_data_processor = DataPreprocessor(self.tokenizer, self.max_encoder_seq_length)
        self.decoder_data_processor = DataPreprocessor(self.tokenizer, self.max_decoder_seq_length)

        # 5. Preprocess the datasets
        # Preprocess the training data
        encoder_input_train = self.encoder_data_processor.preprocess_smiles(tokenized_products_x_train_data)
        decoder_full_train = self.decoder_data_processor.preprocess_smiles(tokenized_reactants_y_train_data)
        decoder_input_train = decoder_full_train[:, :-1]
        decoder_target_train = decoder_full_train[:, 1:]

        # Preprocess test data
        encoder_input_test = self.encoder_data_processor.preprocess_smiles(tokenized_products_x_test_data)
        decoder_full_test = self.decoder_data_processor.preprocess_smiles(tokenized_reactants_y_test_data)
        decoder_input_test = decoder_full_test[:, :-1]
        decoder_target_test = decoder_full_test[:, 1:]

        # Preprocess validation data
        encoder_input_valid = self.encoder_data_processor.preprocess_smiles(tokenized_products_x_valid_dataset)
        decoder_full_valid = self.decoder_data_processor.preprocess_smiles(tokenized_reactants_y_valid_dataset)
        decoder_input_valid = decoder_full_valid[:, :-1]
        decoder_target_valid = decoder_full_valid[:, 1:]

        # 6. Store datasets
        self.train_data = (
            (encoder_input_train, decoder_input_train),
            decoder_target_train
        )
        self.valid_data = (
            (encoder_input_valid, decoder_input_valid),
            decoder_target_valid
        )
        self.test_data = (
            (encoder_input_test, decoder_input_test),
            decoder_target_test
        )

    def get_dataset(self, data, training=True) -> DatasetV2:
        """
        Create a tf.data.Dataset from the given data.
        """
        (encoder_input, decoder_input), decoder_target = data

        dataset = tf.data.Dataset.from_tensor_slices((
            (encoder_input, decoder_input),
            decoder_target
        ))

        if training:
            dataset = dataset.shuffle(self.buffer_size)

        # Only use drop remainder for batching for training data
        drop_remainder = training
        dataset = dataset.batch(self.batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_train_dataset(self) -> DatasetV2:
        return self.get_dataset(self.train_data, training=True)

    def get_valid_dataset(self) -> DatasetV2:
        return self.get_dataset(self.valid_data, training=False)

    def get_test_dataset(self) -> DatasetV2:
        return self.get_dataset(self.test_data, training=False)

    @staticmethod
    def load_smiles(file_path):
        with open(file_path, 'r') as file:
            smiles_list = [line.strip() for line in file.readlines()]
        return smiles_list
