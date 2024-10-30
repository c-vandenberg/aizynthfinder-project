import os
import json
from typing import Dict, Any, List, Tuple, Union, Callable, Optional

import yaml
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    TensorBoard,
    ReduceLROnPlateau
)
from tensorflow.train import Checkpoint, CheckpointManager

from trainers.environment import TrainingEnvironment
from callbacks.checkpoints import BestValLossCallback
from callbacks.validation_metrics import ValidationMetricsCallback
from callbacks.gradient_monitoring import GradientMonitoringCallback
from metrics.perplexity import Perplexity
from data.utils.data_loader import DataLoader
from data.utils.tokenization import SmilesTokenizer
from data.utils.preprocessing import SmilesDataPreprocessor
from data.utils.logging import (compute_metrics, log_metrics, print_metrics,
                                log_sample_predictions, print_sample_predictions)
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

        Returns
        -------
        None
        """
        self.config:Dict[str, Any] = self.load_config(config_path)

        self.tokenizer: Optional[SmilesTokenizer] = None
        self.data_loader: Optional[DataLoader] = None
        self.vocab_size: Optional[int] = None
        self.encoder_preprocessor: Optional[SmilesDataPreprocessor] = None
        self.decoder_preprocessor: Optional[SmilesDataPreprocessor] = None
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

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        yaml.YAMLError
            If there is an error parsing the YAML file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path, 'r') as file:
            try:
                config: Dict[str, Any] = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file: {e}")
        return config

    def initialize_components(self) -> None:
        """
        Initialize DataLoader, Tokenizer, Preprocessor, and hyperparameters.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If required configuration keys are missing.
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
            random_state=data_conf.get('random_state', 42),
            reverse_input_sequence=train_conf.get('reverse_tokenized_input_sequence', True)
        )

        # Load and prepare data
        self.data_loader.load_and_prepare_data()

        # Access tokenizer and vocab size
        self.tokenizer = self.data_loader.smiles_tokenizer
        self.vocab_size = self.data_loader.vocab_size

        # Save the tokenizer
        self.save_tokenizer(data_conf.get('tokenizer_save_path', 'tokenizer.json'))

        # Initialize Preprocessors
        self.encoder_preprocessor = SmilesDataPreprocessor(
            smiles_tokenizer=self.data_loader.smiles_tokenizer,
            max_seq_length=data_conf.get('max_encoder_seq_length', 140)
        )
        self.decoder_preprocessor = SmilesDataPreprocessor(
            smiles_tokenizer=self.data_loader.smiles_tokenizer,
            max_seq_length=data_conf.get('max_decoder_seq_length', 140)
        )

    def save_tokenizer(self, tokenizer_path: str) -> None:
        """
        Saves the tokenizer's vocabulary to a JSON file.

        Parameters
        ----------
        tokenizer_path : str
            Path where the tokenizer vocabulary JSON will be saved.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If the tokenizer cannot be saved to the specified path.
        """
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        try:
            with open(tokenizer_path, 'w') as f:
                json.dump(self.tokenizer.word_index, f, indent=4)
            print(f"Tokenizer vocabulary saved to {tokenizer_path}")
        except IOError as e:
            print(f"Failed to save tokenizer to {tokenizer_path}: {e}")
            raise

    def setup_model(self) -> None:
        """
        Initializes and compiles the Seq2Seq model.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If required model configuration keys are missing.
        """
        model_conf: Dict[str, Any] = self.config.get('model', {})

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
        self.model.smiles_tokenizer = self.tokenizer

        # Set up the optimizer
        self.optimizer: Adam = Adam(learning_rate=learning_rate, clipnorm=5.0)

        # Set up the loss function and metrics
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.metrics: List[Any] = model_conf.get('metrics', ['accuracy'])
        self.metrics.append(Perplexity(loss_function=self.loss_function))

        # Compile the model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=self.metrics
        )

    def build_model(self) -> None:
        """
        Builds the model by running a sample input through it to initialize weights.

        Returns
        -------
        None

        Raises
        ------
        StopIteration
            If the training dataset is empty.
        """
        print("Building the model with sample data to initialize variables...")

        # Get a batch from the training dataset
        train_dataset = self.data_loader.get_train_dataset()
        try:
            sample_batch = next(iter(train_dataset))
        except StopIteration:
            raise StopIteration("Training dataset is empty. Cannot build the model.")

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

        Raises
        ------
        KeyError
            If required training configuration keys are missing.
        """
        training_conf: Dict[str, Any] = self.config.get('training', {})

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

        # Validation metrics callback
        valid_metrics_dir: str = training_conf.get('valid_metrics_dir', './validation-metrics')
        tensorboard_dir: str = training_conf.get('tensorboard_dir', './tensorboard')
        validation_metrics_callback: ValidationMetricsCallback = ValidationMetricsCallback(
            tokenizer=self.tokenizer,
            validation_data=self.data_loader.get_valid_dataset(),
            validation_metrics_dir=valid_metrics_dir,
            tensorboard_dir=os.path.join(tensorboard_dir, 'validation_metrics'),
            max_length=self.data_loader.max_decoder_seq_length
        )

        # TensorBoard
        tensorboard_callback: TensorBoard = TensorBoard(
            log_dir=tensorboard_dir
        )

        # Gradient monitoring
        gradient_callback = GradientMonitoringCallback(
            log_dir=os.path.join(tensorboard_dir, 'gradients')
        )

        self.callbacks = [
            early_stopping,
            best_val_loss_checkpoint_callback,
            lr_scheduler,
            validation_metrics_callback,
            tensorboard_callback
        ]

    def train(self) -> None:
        """
        Trains the Seq2Seq model using the training and validation datasets.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If dataloader or training/validation datasets are not initialised.
        """
        training_conf: Dict[str, Any] = self.config.get('training', {})

        if self.data_loader is None:
            raise ValueError("DataLoader is not initialised.")

        train_dataset = self.data_loader.get_train_dataset()
        valid_dataset = self.data_loader.get_valid_dataset()

        if train_dataset is None or valid_dataset is None:
            raise ValueError("Training or validation datasets are not available.")

        self.model.fit(
            train_dataset,
            epochs=training_conf.get('epochs', 10),
            validation_data=valid_dataset,
            callbacks=self.callbacks
        )

    def evaluate(self) -> None:
        """
        Evaluates the trained model on the test dataset.

        Computes various metrics including loss, accuracy, perplexity, BLEU score, chemical validity,
        Tanimoto similarity, and Levenshtein distance. Results are logged and printed.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If tokenizer is not set or if required metrics functions are not available.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")

        test_dataset = self.data_loader.get_test_dataset()
        training_conf: Dict[str, Any] = self.config.get('training', {})
        model_conf: Dict[str, Any] = self.config.get('model', {})
        test_metrics_dir: str = training_conf.get('test_metrics_dir', './evaluation')

        # Evaluate the model on the test dataset
        evaluation_results: List[float] = self.model.evaluate(test_dataset)
        if len(evaluation_results) < 3:
            raise ValueError("Expected at least three evaluation metrics (loss, accuracy, perplexity).")

        test_loss, test_accuracy, test_perplexity = evaluation_results[:3]

        references: List[List[List[str]]] = []
        hypotheses: List[List[str]] = []
        target_smiles: List[str] = []
        predicted_smiles: List[str] = []
        start_token: str = self.tokenizer.start_token
        end_token: str = self.tokenizer.end_token

        for (encoder_input, decoder_input), target_sequences in test_dataset:
            # Generate sequences
            predicted_sequences_list, predicted_scores_list = self.model.predict_sequence_beam_search(
                encoder_input=encoder_input,
                beam_width=model_conf.get('beam_width', 5),
                max_length=self.data_loader.max_decoder_seq_length,
                start_token_id=self.tokenizer.word_index.get(start_token),
                end_token_id=self.tokenizer.word_index.get(end_token)
            )

            top_predicted_sequences = [seq_list[0] for seq_list in predicted_sequences_list]

            # Convert sequences to text
            predicted_texts: List[str]  = self.tokenizer.sequences_to_texts(
                top_predicted_sequences,
                is_input_sequence=False
            )
            target_texts: List[str]  = self.tokenizer.sequences_to_texts(
                target_sequences,
                is_input_sequence=False
            )

            for ref, hyp in zip(target_texts, predicted_texts):
                ref_tokens: List[str]  = ref.split()
                hyp_tokens: List[str]  = hyp.split()
                references.append([ref_tokens])
                hypotheses.append(hyp_tokens)
                target_smiles.append(ref)
                predicted_smiles.append(hyp)

        metrics: Dict[str, float] = {
            'Test Loss': test_loss,
            'Test Accuracy': test_accuracy,
            'Test Perplexity': test_perplexity,
        }

        additional_metrics: Dict[str, float] = compute_metrics(
            references=references,
            hypotheses=hypotheses,
            target_smiles=target_smiles,
            predicted_smiles=predicted_smiles,
            evaluation_stage='Test'
        )

        metrics.update(additional_metrics)

        log_metrics(
            metrics=metrics,
            directory=test_metrics_dir,
            filename='test_metrics.txt',
            separator='-' * 40
        )

        print_metrics(metrics=metrics)

        log_sample_predictions(
            target_smiles=target_smiles,
            predicted_smiles=predicted_smiles,
            directory=test_metrics_dir,
            filename='test_sample_predictions.txt',
            num_samples=5,
            separator_length=153
        )

        print_sample_predictions(
            target_smiles=target_smiles,
            predicted_smiles=predicted_smiles,
            num_samples=5,
            separator_length=153
        )

    def save_model(self) -> None:
        """
        Saves the trained model in TensorFlow SavedModel format and ONNX format.

        The method inspects the model layers, saves the model in Keras, HDF5, ONNX, and SavedModel formats.
        It ensures that the model is saved in multiple formats for compatibility and deployment purposes.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If any of the model saving processes fail.
        """
        Seq2SeqModelUtils.inspect_model_layers(model=self.model)
        training_conf: Dict[str, Any] = self.config.get('training', {})
        data_conf: Dict[str, Any] = self.config.get('data', {})
        model_save_dir: str = training_conf.get('model_save_dir', './model')
        keras_save_dir: str = os.path.join(model_save_dir, 'keras')
        hdf5_save_dir: str = os.path.join(model_save_dir, 'hdf5')
        onnx_save_dir: str = os.path.join(model_save_dir, 'onnx')
        saved_model_save_dir: str = os.path.join(model_save_dir, 'saved_model')

        try:
            Seq2SeqModelUtils.model_save_keras_format(
                keras_save_dir=keras_save_dir,
                model=self.model
            )
        except Exception as e:
            print(f"Error saving model in Keras format: {e}")

        try:
            Seq2SeqModelUtils.model_save_hdf5_format(
                hdf5_save_dir=hdf5_save_dir,
                model=self.model
            )
        except Exception as e:
            print(f"Error saving model in HDF5 format: {e}")

        try:
            Seq2SeqModelUtils.model_save_onnx_format(
                onnx_output_dir=onnx_save_dir,
                model=self.model,
                max_encoder_seq_length=data_conf.get('max_encoder_seq_length', 140),
                max_decoder_seq_length=data_conf.get('max_decoder_seq_length', 140)
            )
        except Exception as e:
            print(f"Error saving model in ONNX format: {e}")

        try:
            Seq2SeqModelUtils.save_saved_model_format(
                model_save_path=saved_model_save_dir,
                model=self.model
            )
        except Exception as e:
            print(f"Error saving model in SavedModel format: {e}")

    def run(self):
        """
        Executes the full training pipeline.

        The pipeline includes:
        1. Setting up the training environment.
        2. Setting up and compiling the model.
        3. Building the model by initializing weights.
        4. Setting up training callbacks.
        5. Training the model.
        6. Saving the trained model.
        7. Evaluating the model on the test dataset.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If any step in the training pipeline fails.
        """
        try:
            TrainingEnvironment.setup_environment(self.config)
            self.setup_model()
            self.build_model()
            self.setup_callbacks()
            self.train()
            self.model.summary()
            self.save_model()
            self.evaluate()
        except Exception as e:
            print(f"An error occurred during the training pipeline: {e}")
            raise

def custom_train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    gradient_callback: Any
) -> Callable[[Tuple[tf.Tensor, tf.Tensor]], tf.Tensor]:
    """
    Creates a custom training step function.

    This function defines a custom training step that computes the loss, calculates gradients,
    applies them using the optimizer, and invokes a gradient callback for monitoring or logging.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to be trained.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer used to apply gradients.
    loss_fn : Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
        A function that computes the loss given true and predicted values.
    gradient_callback : Any
        An object that has an `on_gradients_computed` method to handle gradient-related operations
        (e.g., logging, monitoring).

    Returns
    -------
    Callable[[Tuple[tf.Tensor, tf.Tensor]], tf.Tensor]
        A TensorFlow function that performs a single training step.

    Raises
    ------
    AttributeError
        If `gradient_callback` does not have an `on_gradients_computed` method.
    """
    if not hasattr(gradient_callback, 'on_gradients_computed'):
        raise AttributeError("gradient_callback must have an 'on_gradients_computed' method.")

    @tf.function
    def train_step(inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Performs a single training step.

        Parameters
        ----------
        inputs : Tuple[tf.Tensor, tf.Tensor]
            A tuple containing input features and target labels:
            - inputs[0]: tf.Tensor representing input features.
            - inputs[1]: tf.Tensor representing target labels.

        Returns
        -------
        tf.Tensor
            The computed loss for the training step.
        """
        x, y = inputs
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Call gradient callback
        gradient_callback.on_gradients_computed(gradients, model.trainable_variables)

        return loss

    return train_step