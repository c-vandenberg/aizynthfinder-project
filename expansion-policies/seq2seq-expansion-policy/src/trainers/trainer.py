import os
import yaml
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.train import Checkpoint, CheckpointManager
from tensorflow.python.types.data import DatasetV2
from models.seq2seq import RetrosynthesisSeq2SeqModel, BestValLossCheckpointCallback
from models.utils import Seq2SeqModelUtils
from data.utils.data_loader import DataLoader
from data.utils.tokenization import SmilesTokenizer
from data.utils.preprocessing import DataPreprocessor
from typing import Dict, Any, Optional, List


class Trainer:
    def __init__(self, config_path: str):
        """
        Initialize the Trainer with configurations.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config:Dict[str, Any] = self.load_config(config_path)
        self.setup_environment()

        self.tokenizer: Optional[SmilesTokenizer] = None
        self.data_loader: Optional[DataLoader] = None
        self.vocab_size: Optional[int] = None
        self.encoder_preprocessor: Optional[DataPreprocessor] = None
        self.decoder_preprocessor: Optional[DataPreprocessor] = None
        self.model: Optional[RetrosynthesisSeq2SeqModel] = None
        self.optimizer: Optional[Adam] = None
        self.loss_function: Any = None
        self.metrics: Optional[List[str]] = None
        self.callbacks: Optional[List[Callback]] = None

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

    def setup_environment(self) -> None:
        """
        Set up the environment for deterministic (reproducible) training.

        This method ensures that the training environment is configured to be deterministic, enabling reproducibility
        across runs. It sets seeds for Python, NumPy, and TensorFlow, and configures TensorFlow for deterministic
        operations.

        The process follows best practices for setting up deterministic environments, including setting the hash seed,
        configuring random number generators, and enabling deterministic operations in TensorFlow.

        More information on deterministic training and reproducibility can be found at:
        - NVIDIA Clara Train documentation: https://docs.nvidia.com/clara/clara-train-archive/3.1/nvmidl/additional_features/determinism.html
        - NVIDIA Reproducibility Framework GitHub: https://github.com/NVIDIA/framework-reproducibility/tree/master/doc/d9m

        Parameters
        ----------
        self

        Returns
        -------
        None

        Notes
        -----
        1. Sets the `PYTHONHASHSEED` environment variable to control the hash seed used by Python.
        2. Seeds Python's `random` module, NumPy, and TensorFlow's random number generators for consistency.
        3. Enables deterministic operations in TensorFlow by setting `TF_DETERMINISTIC_OPS=1`.
        4. Optionally disables GPU and limits TensorFlow to single-threaded execution.
            - This is because modern GPUs and CPUs are designed to execute computations in parallel across many cores.
            - This parallelism is typically managed asynchronously, meaning that the order of operations or the
            availability of computing resources can vary slightly from one run to another
            - It is this asynchronous parallelism that can introduce random noise, and hence, non-deterministic
            behaviour.
            - However, configuring TensorFlow to use the CPU (`os.environ['CUDA_VISIBLE_DEVICES'] = ''`) and configuring
            Tensorflow to use single-threaded execution severely impacts performance.
        """
        determinism_conf: dict[str, Any] = self.config['env']['determinism']

        # 1. Set Python's built-in hash seed
        os.environ['PYTHONHASHSEED'] = str(determinism_conf['python_seed'])

        # 2. Set Python's random module seed
        random.seed(determinism_conf['random_seed'])

        # 3. Set NumPy's random seed
        np.random.seed(determinism_conf['numpy_seed'])

        # 4. Set TensorFlow's random seed
        tf.random.set_seed(determinism_conf['tf_seed'])

        # 5. Configure TensorFlow for deterministic operations
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

        # Configure (optional, heavily impacts performance)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        # Configure TensorFlow session for single-threaded execution (optional, heavily impacts performance)
        # tf.config.threading.set_intra_op_parallelism_threads(1)
        # tf.config.threading.set_inter_op_parallelism_threads(1)

        print("Environment setup for deterministic (reproducible) training complete.")


    def initialize_components(self):
        """
        Initialize DataLoader, Tokenizer, Preprocessor, and hyperparameters.
        """
        # Initialize SmilesTokenizer
        self.tokenizer: SmilesTokenizer = SmilesTokenizer()

        # Initialize DataLoader with paths and parameters from 'data' config
        data_conf: dict[str, Any] = self.config['data']
        model_conf: dict[str, Any] = self.config['model']
        train_conf: dict[str, Any] = self.config['training']

        self.data_loader = DataLoader(
            products_file=data_conf['products_file'],
            reactants_file=data_conf['reactants_file'],
            products_valid_file=data_conf['products_valid_file'],
            reactants_valid_file=data_conf['reactants_valid_file'],
            num_samples=train_conf.get('num_samples', None),
            max_encoder_seq_length=data_conf['max_encoder_seq_length'],
            max_decoder_seq_length=data_conf['max_decoder_seq_length'],
            batch_size=data_conf['batch_size'],
            test_size=data_conf['test_size'],
            random_state=data_conf['random_state']
        )

        # Load and prepare data
        self.data_loader.load_and_prepare_data()

        # Access tokenizer and vocab size
        self.tokenizer: SmilesTokenizer = self.data_loader.tokenizer
        self.vocab_size: int = self.data_loader.vocab_size

        # Save the tokenizer
        tokenizer_path: str = data_conf['tokenizer_save_path']
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        with open(tokenizer_path, 'w') as f:
            f.write(self.tokenizer.to_json())

        # Update model configuration with vocab sizes
        model_conf['input_vocab_size'] = self.vocab_size
        model_conf['output_vocab_size'] = self.vocab_size

        # Initialize Preprocessors
        self.encoder_preprocessor: DataPreprocessor = DataPreprocessor(
            tokenizer=self.tokenizer,
            max_seq_length=data_conf['max_encoder_seq_length']
        )
        self.decoder_preprocessor: DataPreprocessor = DataPreprocessor(
            tokenizer=self.tokenizer,
            max_seq_length=data_conf['max_decoder_seq_length']
        )

    def setup_model(self):
        """
        Initialize and compile the model.
        """
        model_conf: dict[str, Any] = self.config['model']
        encoder_embedding_dim: int = model_conf.get('encoder_embedding_dim', 256)
        decoder_embedding_dim: int = model_conf.get('decoder_embedding_dim', 256)
        units: int = model_conf.get('units', 256)
        dropout_rate: float = model_conf.get('dropout_rate', 0.2)

        # Initialize the model
        self.model: RetrosynthesisSeq2SeqModel = RetrosynthesisSeq2SeqModel(
            input_vocab_size=self.vocab_size,
            output_vocab_size=self.vocab_size,
            encoder_embedding_dim=encoder_embedding_dim,
            decoder_embedding_dim=decoder_embedding_dim,
            units=units,
            dropout_rate=dropout_rate
        )

        # Set encoder and decoder preprocessors
        self.model.encoder_data_processor = self.encoder_preprocessor
        self.model.decoder_data_processor = self.decoder_preprocessor

        # Set up the optimizer
        learning_rate: float = model_conf.get('learning_rate', 0.0001)
        self.optimizer: Adam = Adam(learning_rate=learning_rate, clipnorm=5.0)

        # Set up the loss function and metrics
        self.loss_function = Seq2SeqModelUtils.masked_sparse_categorical_crossentropy
        self.metrics: List[Any] = model_conf.get('metrics', ['accuracy'])

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
        training_conf: dict[str, Any] = self.config['training']

        # Early Stopping
        early_stopping: EarlyStopping = EarlyStopping(
            monitor='val_loss',
            patience=training_conf.get('patience', 5),
            restore_best_weights=True
        )

        # TensorBoard
        tensorboard_callback: EarlyStopping = TensorBoard(
            log_dir=training_conf['log_dir']
        )

        # Custom Checkpoint Callback
        checkpoint_dir: str = training_conf['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint: Checkpoint = Checkpoint(model=self.model, optimizer=self.optimizer)
        checkpoint_manager: CheckpointManager = CheckpointManager(
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
        best_val_loss_checkpoint_callback: BestValLossCheckpointCallback = BestValLossCheckpointCallback(
            checkpoint_manager
        )

        # Initialize a learning rate scheduler
        lr_scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3
        )

        self.callbacks = [
            early_stopping,
            best_val_loss_checkpoint_callback,
            lr_scheduler,
            tensorboard_callback
        ]

    def train(self):
        """
        Train the Seq2Seq model using the training and validation datasets.
        """
        training_conf: dict[str, Any] = self.config['training']

        train_dataset: DatasetV2 = self.data_loader.get_train_dataset()
        valid_dataset: DatasetV2 = self.data_loader.get_valid_dataset()

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
        test_dataset: DatasetV2 = self.data_loader.get_test_dataset()

        test_loss, test_accuracy = self.model.evaluate(test_dataset)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

    def save_model(self):
        """
        Save the trained model in TensorFlow SavedModel format.
        """
        Seq2SeqModelUtils.inspect_model_layers(self.model)
        training_conf: dict[str, Any] = self.config['training']
        model_save_path: str = training_conf['model_save_path']
        os.makedirs(model_save_path, exist_ok=True)

        self.model.export(model_save_path)
        print(f"Model saved to {model_save_path}")

    def run(self):
        """
        Execute the full training pipeline.
        """
        self.setup_model()
        self.build_model()
        self.setup_callbacks()
        self.train()
        self.model.summary()
        self.save_model()
        self.evaluate()
