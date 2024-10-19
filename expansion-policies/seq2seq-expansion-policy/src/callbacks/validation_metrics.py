import os
from typing import Iterable, Optional, Union, Tuple, List, Any

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.summary import SummaryWriter

from data.utils.logging import (compute_metrics, log_metrics, print_metrics,
                                log_sample_predictions, print_sample_predictions,
                                log_to_tensorboard)

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
        max_length: int = 140
    ) -> None:
        super(ValidationMetricsCallback, self).__init__()
        self.tokenizer = tokenizer
        self.validation_data = validation_data
        self.validation_metrics_dir = validation_metrics_dir
        self.tensorboard_dir = tensorboard_dir
        self.max_length = max_length

        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.file_writer = tf.summary.create_file_writer(self.tensorboard_dir)

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

        for (encoder_input, decoder_input), target_output in self.validation_data:
            # Generate sequences
            predicted_sequences = self.model.predict_sequence(
                encoder_input,
                max_length=self.max_length,
                start_token_id=self.tokenizer.word_index.get(start_token),
                end_token_id=self.tokenizer.word_index.get(end_token)
            )

            # Convert sequences to text
            predicted_texts = self.tokenizer.sequences_to_texts(
                predicted_sequences.numpy(),
                is_input_sequence=False
            )
            target_texts = self.tokenizer.sequences_to_texts(
                target_output.numpy(),
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
            epoch=epoch,
            references=references,
            hypotheses=hypotheses,
            target_smiles=target_smiles,
            predicted_smiles=predicted_smiles,
            validation_metrics_dir=self.validation_metrics_dir,
            tensorboard_dir=self.tensorboard_dir,
            file_writer=self.file_writer,
        )

    @staticmethod
    def validation(
        epoch: int,
        references,
        hypotheses,
        target_smiles: List[str],
        predicted_smiles: List[str],
        validation_metrics_dir: str,
        tensorboard_dir: Optional[Union[str, None]] = None,
        file_writer: Optional[Union[SummaryWriter, None]] = None
    ) -> None:
        metrics = compute_metrics(references, hypotheses, target_smiles, predicted_smiles)
        log_metrics(epoch, metrics, validation_metrics_dir)
        print_metrics(epoch, metrics)
        log_sample_predictions(epoch, target_smiles, predicted_smiles, validation_metrics_dir)
        print_sample_predictions(target_smiles, predicted_smiles)

        if tensorboard_dir and file_writer:
            log_to_tensorboard(file_writer, metrics, epoch)
