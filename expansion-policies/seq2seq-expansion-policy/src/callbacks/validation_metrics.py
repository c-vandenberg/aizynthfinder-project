import os
import logging
import time
from typing import Iterable, Optional, Union, Tuple, List, Dict, Any

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.summary import SummaryWriter

from metrics.smiles_string_metrics import SmilesStringMetrics
from data.utils.logging_utils import (extract_core_log_metrics, compute_metrics,
                                log_metrics, print_metrics, log_sample_predictions,
                                print_sample_predictions, log_to_tensorboard)

class ValidationMetricsCallback(Callback):
    """
    Callback to compute and log BLEU score, exact SMILES match accuracy, chemical validity, and Levenshtein Distance
    on validation data at the end of each epoch.

    Parameters
    ----------
    tokenizer : Any
        The tokenizer used to convert sequences to text.
    validation_data : Iterable[Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]]
        The validation dataset.
    tensorboard_dir : str, optional
        Directory to save the TensorBoard logs, by default None.
    max_length : int, optional
        The maximum length of the generated sequences, by default 140.

    Attributes
    ----------
    tokenizer : Any
        The tokenizer used to convert sequences to text.
    validation_data : Iterable[Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]]
        The validation dataset.
    tensorboard_dir : Optional[str]
        Directory to save the TensorBoard logs.
    max_length : int
        The maximum length of the generated sequences.
    file_writer : Optional[tf.summary.SummaryWriter]
        TensorBoard summary writer for logging metrics.
    """
    def __init__(
        self,
        tokenizer: Any,
        validation_data: Iterable[Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]],
        validation_metrics_dir: str,
        tensorboard_dir: str,
        logger: logging.Logger,
        max_length: int = 140
    ) -> None:
        super(ValidationMetricsCallback, self).__init__()
        self.tokenizer = tokenizer
        self.validation_data = validation_data
        self.validation_metrics_dir = validation_metrics_dir
        self.tensorboard_dir = tensorboard_dir
        self.logger = logger
        self.max_length = max_length

        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.file_writer = tf.summary.create_file_writer(self.tensorboard_dir)

        self.smiles_string_metrics = SmilesStringMetrics()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """
        Computes BLEU score, Exact SMILES Match Accuracy, Chemical Validity,
        and Levenshtein Distance on validation data and logs them.

        Parameters
        ----------
        epoch : int
            The index of the epoch.
        logs : dict, optional
            Dictionary of logs from the training process.
        """
        references = []
        hypotheses = []
        target_smiles = []
        predicted_smiles = []
        start_token = self.tokenizer.start_token
        end_token = self.tokenizer.end_token

        validation_start_time: float = time.time()

        for (encoder_input, decoder_input), target_sequences in self.validation_data:
            # Generate sequences
            predicted_sequences = self.model.predict_sequence_tf(
                encoder_input,
                max_length=self.max_length,
                start_token_id=self.tokenizer.word_index.get(start_token),
                end_token_id=self.tokenizer.word_index.get(end_token)
            )

            # Convert sequences to text
            predicted_texts = self.tokenizer.sequences_to_texts(
                predicted_sequences,
                is_input_sequence=False
            )
            target_texts = self.tokenizer.sequences_to_texts(
                target_sequences,
                is_input_sequence=False
            )

            for ref, hyp in zip(target_texts, predicted_texts):
                ref_tokens = ref.split()
                hyp_tokens = hyp.split()
                references.append([ref_tokens])
                hypotheses.append(hyp_tokens)
                target_smiles.append(ref)
                predicted_smiles.append(hyp)

        self.validation(
            logs=logs,
            epoch=epoch,
            references=references,
            hypotheses=hypotheses,
            target_smiles=target_smiles,
            predicted_smiles=predicted_smiles,
            validation_metrics_dir=self.validation_metrics_dir,
            tensorboard_dir=self.tensorboard_dir,
            file_writer=self.file_writer
        )

        validation_end_time: float = time.time()

        validation_time = validation_end_time - validation_start_time

        self.logger.info(f'Epoch {epoch} Validation Time: {round(validation_time)} seconds')

    def validation(
        self,
        logs: Union[Dict, None],
        epoch: int,
        references,
        hypotheses,
        target_smiles: List[str],
        predicted_smiles: List[str],
        validation_metrics_dir: str,
        tensorboard_dir: Optional[str] = None,
        file_writer: Optional[SummaryWriter] = None
    ) -> None:
        """
        Performs validation by computing metrics, logging them, and optionally
        recording to TensorBoard.

        This method extracts core log metrics, computes custom metrics based on
        references and hypotheses, logs the metrics to specified directories,
        prints them, and logs sample predictions. If TensorBoard directories
        and writers are provided, it logs metrics to TensorBoard as well.

        Parameters
        ----------
        logs : Optional[Dict]
            A dictionary containing log information from the training process.
        epoch : int
            The current epoch number.
        references : List[str]
            A list of reference SMILES strings.
        hypotheses : List[str]
            A list of hypothesis SMILES strings.
        target_smiles : List[str]
            A list of target SMILES strings used for validation.
        predicted_smiles : List[str]
            A list of predicted SMILES strings generated by the model.
        validation_metrics_dir : str
            Directory path where validation metrics will be saved.
        tensorboard_dir : Optional[str], default=None
            Directory path for TensorBoard logs. If None, TensorBoard logging is skipped.
        file_writer : Optional[SummaryWriter], default=None
            TensorBoard SummaryWriter instance. Required if `tensorboard_dir` is provided.

        Returns
        -------
        None
            This method does not return any value.
        """
        metrics = extract_core_log_metrics(logs)
        custom_metrics = compute_metrics(
            references=references,
            hypotheses=hypotheses,
            target_smiles=target_smiles,
            predicted_smiles=predicted_smiles,
            smiles_string_metrics=self.smiles_string_metrics,
            evaluation_stage='Validation'
        )
        metrics.update(custom_metrics)

        log_metrics(metrics=metrics, directory=validation_metrics_dir, epoch=epoch)
        print_metrics(logger=self.logger, metrics=metrics, epoch=epoch)
        log_sample_predictions(
            target_smiles=target_smiles,
            predicted_smiles=predicted_smiles,
            directory=validation_metrics_dir,
            epoch=epoch,
        )
        print_sample_predictions(logger=self.logger, target_smiles=target_smiles, predicted_smiles=predicted_smiles)

        if tensorboard_dir and file_writer:
            log_to_tensorboard(file_writer, metrics, epoch)
