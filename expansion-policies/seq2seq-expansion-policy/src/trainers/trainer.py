import os
from typing import Dict, Any, List, Union, Optional

import yaml
import numpy as np
import pydevd_pycharm
from keras.src.utils.module_utils import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (Callback, EarlyStopping,
                                        TensorBoard, ReduceLROnPlateau)
from tensorflow.train import Checkpoint, CheckpointManager
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from trainers.environment import TrainingEnvironment
from callbacks.checkpoints import BestValLossCallback
from callbacks.bleu_score import BLEUScoreCallback
from losses.losses import MaskedSparseCategoricalCrossentropy
from metrics.metrics import Perplexity
from data.utils.data_loader import DataLoader
from data.utils.tokenization import SmilesTokenizer
from data.utils.preprocessing import DataPreprocessor
from models.seq2seq import RetrosynthesisSeq2SeqModel
from models.utils import Seq2SeqModelUtils


class Trainer:
    """
    Trainer class for training and evaluating the Retrosynthesis Seq2Seq model.

    This class handles the setup of the environment, data loading, model
    initialization, training, evaluation, and saving of the model.
    """
    def __init__(self, config_path: str) -> None:
        """
        Initializes the Trainer with configurations.

        Parameters
        ----------
        config_path : str
            Path to the configuration YAML file.
        """
        self.config:Dict[str, Any] = self.load_config(config_path)

        self.tokenizer: Optional[SmilesTokenizer] = None
        self.data_loader: Optional[DataLoader] = None
        self.vocab_size: Optional[int] = None
        self.encoder_preprocessor: Optional[DataPreprocessor] = None
        self.decoder_preprocessor: Optional[DataPreprocessor] = None
        self.model: Optional[RetrosynthesisSeq2SeqModel] = None
        self.optimizer: Optional[Adam] = None
        self.loss_function: Optional[Any] = None
        self.metrics: Optional[List[str]] = None
        self.callbacks: Optional[List[Callback]] = None

        self.initialize_components()

    @staticmethod
    def load_config(config_path: str) -> dict:
        """
        Loads configuration from a YAML file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def initialize_components(self) -> None:
        """
        Initialize DataLoader, Tokenizer, Preprocessor, and hyperparameters.

        Returns
        -------
        None
        """
        # Retrieve configurations
        data_conf: Dict[str, Any] = self.config.get('data', {})
        train_conf: Dict[str, Any] = self.config.get('training', {})

        # Initialize DataLoader
        self.data_loader = DataLoader(
            products_file=data_conf.get('products_file', ''),
            reactants_file=data_conf.get('reactants_file', ''),
            products_valid_file=data_conf.get('products_valid_file', ''),
            reactants_valid_file=data_conf.get('reactants_valid_file', ''),
            num_samples=train_conf.get('num_samples'),
            max_encoder_seq_length=data_conf.get('max_encoder_seq_length', 140),
            max_decoder_seq_length=data_conf.get('max_decoder_seq_length', 140),
            batch_size=data_conf.get('batch_size', 16),
            test_size=data_conf.get('test_size', 0.3),
            random_state=data_conf.get('random_state', 42)
        )

        # Load and prepare data
        self.data_loader.load_and_prepare_data()

        # Access tokenizer and vocab size
        self.tokenizer = self.data_loader.tokenizer
        self.vocab_size = self.data_loader.vocab_size

        # Save the tokenizer
        self.save_tokenizer(data_conf.get('tokenizer_save_path', 'tokenizer.json'))

        # Initialize Preprocessors
        self.encoder_preprocessor = DataPreprocessor(
            smiles_tokenizer=self.data_loader.smiles_tokenizer,
            tokenizer=self.tokenizer,
            max_seq_length=data_conf.get('max_encoder_seq_length', 140)
        )
        self.decoder_preprocessor = DataPreprocessor(
            smiles_tokenizer=self.data_loader.smiles_tokenizer,
            tokenizer=self.tokenizer,
            max_seq_length=data_conf.get('max_decoder_seq_length', 140)
        )

    def save_tokenizer(self, tokenizer_path: str) -> None:
        """
        Saves the tokenizer to the specified path.

        Parameters
        ----------
        tokenizer_path : str
            Path where the tokenizer JSON will be saved.

        Returns
        -------
        None
        """
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        with open(tokenizer_path, 'w') as f:
            f.write(self.tokenizer.to_json())
        print(f"Tokenizer saved to {tokenizer_path}")

    def setup_model(self) -> None:
        """
        Initializes and compiles the Seq2Seq model.

        Returns
        -------
        None
        """
        model_conf: dict[str, Any] = self.config.get('model', {})

        # Retrieve model parameters with defaults
        encoder_embedding_dim: int = model_conf.get('encoder_embedding_dim', 256)
        decoder_embedding_dim: int = model_conf.get('decoder_embedding_dim', 256)
        units: int = model_conf.get('units', 256)
        attention_dim: int = model_conf.get('attention_dim', 256)
        encoder_num_layers: int = model_conf.get('encoder_num_layers', 2)
        decoder_num_layers: int = model_conf.get('decoder_num_layers', 4)
        dropout_rate: float = model_conf.get('dropout_rate', 0.2)
        weight_decay: Union[float, None] = model_conf.get('weight_decay', None)
        learning_rate: float = model_conf.get('learning_rate', 0.0001)

        # Initialize the model
        self.model: RetrosynthesisSeq2SeqModel = RetrosynthesisSeq2SeqModel(
            input_vocab_size=self.vocab_size,
            output_vocab_size=self.vocab_size,
            encoder_embedding_dim=encoder_embedding_dim,
            decoder_embedding_dim=decoder_embedding_dim,
            attention_dim=attention_dim,
            units=units,
            num_encoder_layers=encoder_num_layers,
            num_decoder_layers=decoder_num_layers,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )

        # Set encoder and decoder preprocessors
        self.model.encoder_data_processor = self.encoder_preprocessor
        self.model.decoder_data_processor = self.decoder_preprocessor

        # Set up the optimizer
        self.optimizer: Adam = Adam(learning_rate=learning_rate, clipnorm=5.0)

        # Set up the loss function and metrics
        self.loss_function = MaskedSparseCategoricalCrossentropy(padding_idx=0)
        self.metrics: List[Any] = model_conf.get('metrics', ['accuracy'])
        self.metrics.append(Perplexity(loss_function=self.loss_function))

        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)

    def build_model(self) -> None:
        """
        Builds the model by running a sample input through it to initialize weights.

        Returns
        -------
        None
        """
        print("Building the model with sample data to initialize variables...")

        # Get a batch from the training dataset
        train_dataset = self.data_loader.get_train_dataset()
        sample_batch = next(iter(train_dataset))
        (sample_encoder_input, sample_decoder_input), _ = sample_batch

        # Run the model on sample data
        self.model([sample_encoder_input, sample_decoder_input])

        print("Model built successfully.\n")


    def setup_callbacks(self) -> None:
        """
        Sets up training callbacks including EarlyStopping, TensorBoard, Checkpointing, and Learning Rate Scheduler.

        Returns
        -------
        None
        """
        training_conf: dict[str, Any] = self.config.get('training', {})

        # Early Stopping
        early_stopping: EarlyStopping = EarlyStopping(
            monitor='val_loss',
            patience=training_conf.get('patience', 5),
            restore_best_weights=True
        )

        # Checkpoint manager
        checkpoint_dir = training_conf.get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint: Checkpoint = Checkpoint(model=self.model, optimizer=self.optimizer)
        checkpoint_manager: CheckpointManager = CheckpointManager(
            checkpoint,
            directory=checkpoint_dir,
            max_to_keep=5
        )

        # Restore from latest checkpoint if exists
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f"Restored from {checkpoint_manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")

        # Checkpoint Callback
        best_val_loss_checkpoint_callback: BestValLossCallback = BestValLossCallback(
            checkpoint_manager
        )

        # Learning Rate Scheduler
        lr_scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3
        )

        # BLEU score callback
        bleu_callback: BLEUScoreCallback = BLEUScoreCallback(
            tokenizer=self.tokenizer,
            validation_data=self.data_loader.get_valid_dataset(),
            log_dir=os.path.join(training_conf.get('log_dir', './logs'), 'bleu_score')
        )

        # TensorBoard
        tensorboard_callback: EarlyStopping = TensorBoard(
            log_dir=training_conf.get('log_dir', './logs')
        )

        self.callbacks = [
            early_stopping,
            best_val_loss_checkpoint_callback,
            lr_scheduler,
            bleu_callback,
            tensorboard_callback
        ]

    def train(self) -> None:
        """
        Trains the Seq2Seq model using the training and validation datasets.

        Returns
        -------
        None
        """
        training_conf: Dict[str, Any] = self.config.get('training', {})

        train_dataset = self.data_loader.get_train_dataset()
        valid_dataset = self.data_loader.get_valid_dataset()

        self.model.fit(
            train_dataset,
            epochs=training_conf.get('epochs', 10),
            validation_data=valid_dataset,
            callbacks=self.callbacks
        )

    def evaluate(self) -> None:
        """
        Evaluates the trained model on the test dataset.

        Returns
        -------
        None
        """
        test_dataset = self.data_loader.get_test_dataset()
        training_conf: Dict[str, Any] = self.config.get('training', {})
        model_conf: Dict[str, Any] = self.config.get('model', {})
        test_metrics_dir: str = training_conf.get('test_metrics_dir', './evaluation')
        os.makedirs(test_metrics_dir, exist_ok=True)

        test_loss, test_accuracy, test_perplexity = self.model.evaluate(test_dataset)

        references = []
        hypotheses = []
        beam_width = training_conf.get('beam_width', 5)
        start_token = self.data_loader.smiles_tokenizer.start_token
        start_token_id = self.tokenizer.word_index[start_token]
        end_token = self.data_loader.smiles_tokenizer.end_token
        end_token_id = self.tokenizer.word_index[end_token]

        for (encoder_input, decoder_input), target_output in test_dataset:
            predicted_sequences = self.model.predict_sequence_beam_search(
                encoder_input,
                beam_width=beam_width,
                start_token_id=start_token_id,
                end_token_id=end_token_id
            )

            if isinstance(predicted_sequences, list):
                predicted_sequences = np.array(predicted_sequences)

            predicted_texts = self.tokenizer.sequences_to_texts(predicted_sequences)
            target_texts = self.tokenizer.sequences_to_texts(target_output.numpy())

            for ref, hyp in zip(target_texts, predicted_texts):
                ref_tokens = ref.split()
                hyp_tokens = hyp.split()
                references.append([ref_tokens])
                hypotheses.append(hyp_tokens)

        smoothing_function = SmoothingFunction().method1
        bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothing_function)

        with open(os.path.join(test_metrics_dir, 'test_metrics.txt'), "w") as f:
            f.write(f"Test Loss: {test_loss}\n")
            f.write(f"Test Accuracy: {test_accuracy}\n")
            f.write(f"Test Perplexity: {test_perplexity}\n")
            f.write(f"Test BLEU Score: {bleu_score}\n")

        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test Perplexity: {test_perplexity}")
        print(f"Test BLEU Score: {bleu_score}")

    def save_model(self) -> None:
        """
        Saves the trained model in TensorFlow SavedModel format and Onnx format.

        Returns
        -------
        None
        """
        Seq2SeqModelUtils.inspect_model_layers(model=self.model)
        training_conf: Dict[str, Any] = self.config.get('training', {})
        data_conf: Dict[str, Any] = self.config.get('data', {})
        model_save_dir: str = training_conf.get('model_save_dir', './model')
        keras_save_dir: str = os.path.join(model_save_dir, 'keras')
        hdf5_save_dir: str = os.path.join(model_save_dir, 'hdf5')
        onnx_save_dir: str = os.path.join(model_save_dir, 'onnx')
        saved_model_save_dir: str = os.path.join(model_save_dir, 'saved_model')

        Seq2SeqModelUtils.model_save_keras_format(
            keras_save_dir=keras_save_dir,
            model=self.model
        )

        Seq2SeqModelUtils.model_save_hdf5_format(
            hdf5_save_dir=hdf5_save_dir,
            model=self.model
        )

        Seq2SeqModelUtils.model_save_onnx_format(
            onnx_output_dir=onnx_save_dir,
            model=self.model,
            max_encoder_seq_length=data_conf.get('max_encoder_seq_length', 140),
            max_decoder_seq_length=data_conf.get('max_decoder_seq_length', 140)
        )

        Seq2SeqModelUtils.save_saved_model_format(
            model_save_path=saved_model_save_dir,
            model=self.model
        )

    def run(self):
        """
        Executes the full training pipeline.

        Returns
        -------
        None
        """
        TrainingEnvironment.setup_environment(self.config)
        self.load_model('data/training/liu-et-al/model-v17/model/keras/seq2seq_model.keras')
        self.evaluate()

    def load_model(self, model_path: str) -> None:
        self.model = tensorflow.keras.models.load_model(model_path)