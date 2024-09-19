import os
import yaml
import tensorflow as tf
from models.seq2seq import RetrosynthesisSeq2SeqModel, CustomCheckpointCallback
from models.utils import Seq2SeqModelUtils
from data.utils.data_loader import DataLoader
from data.utils.tokenization import SmilesTokenizer
from data.utils.preprocessing import DataPreprocessor
from typing import Union


class Trainer:
    def __init__(self, config_path: str):
        """
        Initialize the Trainer with configurations.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config = self.load_config(config_path)
        self.setup_environment()

        self.tokenizer: Union[SmilesTokenizer, None] = None
        self.data_loader: Union[DataLoader, None] = None
        self.vocab_size: Union[int, None] = None
        self.encoder_preprocessor: Union[DataPreprocessor, None] = None
        self.decoder_preprocessor: Union[DataPreprocessor, None] = None
        self.model: Union[RetrosynthesisSeq2SeqModel, None] = None
        self.optimizer: Union[tf.keras.optimizers.Adam, None] = None
        self.loss_function = None
        self.metrics = None
        self.callbacks = None

        self.initialize_components()

    @staticmethod
    def load_config(config_path: str) -> dict:
        """
        Load configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            dict: Configuration dictionary.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def setup_environment(self):
        """
        Set up the environment, such as setting random seeds.
        """
        seed = self.config.get('misc', {}).get('seed', 42)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def initialize_components(self):
        """
        Initialize DataLoader, Tokenizer, Preprocessor, and hyperparameters.
        """
        # Initialize SmilesTokenizer
        self.tokenizer = SmilesTokenizer()

        # Initialize DataLoader with paths and parameters from 'data' config
        data_conf = self.config['data']
        model_conf = self.config['model']
        self.data_loader = DataLoader(
            products_file=data_conf['products_file'],
            reactants_file=data_conf['reactants_file'],
            products_valid_file=data_conf['products_valid_file'],
            reactants_valid_file=data_conf['reactants_valid_file'],
            max_encoder_seq_length=data_conf['max_encoder_seq_length'],
            max_decoder_seq_length=data_conf['max_decoder_seq_length'],
            batch_size=data_conf['batch_size'],
            test_size=data_conf['test_size'],
            random_state=data_conf['random_state']
        )

        # Load and prepare data
        self.data_loader.load_and_prepare_data()

        # Access tokenizer and vocab size
        self.tokenizer = self.data_loader.tokenizer
        self.vocab_size = self.data_loader.vocab_size

        # Save the tokenizer
        tokenizer_path = data_conf['tokenizer_save_path']
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        with open(tokenizer_path, 'w') as f:
            f.write(self.tokenizer.to_json())

        # Update model configuration with vocab sizes
        model_conf['input_vocab_size'] = self.vocab_size
        model_conf['output_vocab_size'] = self.vocab_size

        # Initialize Preprocessors
        self.encoder_preprocessor = DataPreprocessor(
            tokenizer=self.tokenizer,
            max_seq_length=data_conf['max_encoder_seq_length']
        )
        self.decoder_preprocessor = DataPreprocessor(
            tokenizer=self.tokenizer,
            max_seq_length=data_conf['max_decoder_seq_length']
        )

    def setup_model(self):
        """
        Initialize and compile the model.
        """
        model_conf = self.config['model']
        embedding_dim = model_conf.get('embedding_dim', 256)
        units = model_conf.get('units', 256)
        dropout_rate = model_conf.get('dropout_rate', 0.2)

        # Initialize the model
        self.model = RetrosynthesisSeq2SeqModel(
            input_vocab_size=self.vocab_size,
            output_vocab_size=self.vocab_size,
            embedding_dim=embedding_dim,
            units=units,
            dropout_rate=dropout_rate
        )

        # Set up the optimizer
        learning_rate = model_conf.get('learning_rate', 0.0001)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=5.0)

        # Set up the loss function and metrics
        self.loss_function = Seq2SeqModelUtils.masked_sparse_categorical_crossentropy
        self.metrics = model_conf.get('metrics', ['accuracy'])

        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)

    def build_model(self):
        """
        Build the model by running a sample input through it.
        """
        print("Building the model with sample data to initialize variables...")

        # Get a batch from the training dataset
        for batch in self.data_loader.get_train_dataset().take(1):
            (sample_encoder_input, sample_decoder_input), _ = batch
            self.model([sample_encoder_input, sample_decoder_input])
            break

        print("Model built successfully.\n")

    def setup_callbacks(self):
        """
        Set up training callbacks: EarlyStopping, TensorBoard, and CustomCheckpointCallback.
        """
        training_conf = self.config['training']

        # Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=training_conf.get('patience', 5),
            restore_best_weights=True
        )

        # TensorBoard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=training_conf['log_dir']
        )

        # Custom Checkpoint Callback
        checkpoint_dir = training_conf['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=checkpoint_dir,
            max_to_keep=5  # Keeps the latest 5 checkpoints
        )

        # Restore from latest checkpoint if exists
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f"Restored from {checkpoint_manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")

        # Initialize CustomCheckpointCallback
        custom_checkpoint_callback = CustomCheckpointCallback(checkpoint_manager)

        self.callbacks = [
            early_stopping,
            custom_checkpoint_callback,
            tensorboard_callback
        ]

    def train(self):
        """
        Train the Seq2Seq model using the training and validation datasets.
        """
        training_conf = self.config['training']

        train_dataset = self.data_loader.get_train_dataset()
        valid_dataset = self.data_loader.get_valid_dataset()

        self.model.fit(
            train_dataset,
            epochs=training_conf.get('epochs', 50),
            validation_data=valid_dataset,
            callbacks=self.callbacks
        )

    def evaluate(self):
        """
        Evaluate the trained model on the test dataset.
        """
        test_dataset = self.data_loader.get_test_dataset()

        test_loss, test_accuracy = self.model.evaluate(test_dataset)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

    def save_model(self):
        """
        Save the trained model to the specified path.
        """
        training_conf = self.config['training']
        model_save_path = training_conf['model_save_path']
        os.makedirs(model_save_path, exist_ok=True)

        tf.saved_model.save(self.model, model_save_path)
        print(f"Model saved to {model_save_path}")

    def run(self):
        """
        Execute the full training pipeline.
        """
        self.setup_model()
        self.build_model()
        self.setup_callbacks()
        self.train()
        self.save_model()
        self.evaluate()
