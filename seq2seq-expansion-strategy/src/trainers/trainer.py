import os
import tensorflow as tf
from models.seq2seq import RetrosynthesisSeq2SeqModel
from models.utils import Seq2SeqUtils
from data.utils.data_loader import DataLoader
from data.utils.tokenization import SmilesTokenizer
from data.utils.preprocessing import DataPreprocessor


class Trainer:
    def __init__(
        self,
        config,
        data_loader: DataLoader,
        tokenizer: SmilesTokenizer,
        preprocessor: DataPreprocessor,
        model_save_path: str,
        log_dir: str,
        checkpoint_dir: str,
    ):
        """
        Initialize the Trainer.

        Args:
            config (dict): Configuration dictionary containing hyperparameters and settings.
            data_loader (DataLoader): Instance of DataLoader for loading data.
            tokenizer (SmilesTokenizer): Instance of SmilesTokenizer for tokenization.
            preprocessor (DataPreprocessor): Instance of DataPreprocessor for data preprocessing.
            model_save_path (str): Path to save the trained model.
            log_dir (str): Directory for TensorBoard logs.
            checkpoint_dir (str): Directory to save model checkpoints.
        """
        self.config = config
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.model_save_path = model_save_path
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        # Initialize model components
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.metrics = None
        self.callbacks = []
        self.train_dataset = None
        self.valid_dataset = None

    def setup_model(self):
        """
                Initialize and compile the model.
                """
        vocab_size = self.tokenizer.get_vocab_size()
        embedding_dim = self.config['embedding_dim']
        units = self.config['units']
        dropout_rate = self.config.get('dropout_rate', 0.2)

        # Initialize the model
        self.model = RetrosynthesisSeq2SeqModel(
            input_vocab_size=vocab_size,
            output_vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            units=units,
            dropout_rate=dropout_rate
        )

        # Set up the optimizer
        learning_rate = self.config.get('learning_rate', 0.0001)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=5.0)

        # Set up the loss function and metrics
        self.loss_function = Seq2SeqUtils.masked_sparse_categorical_crossentropy
        self.metrics = ['accuracy']

        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)