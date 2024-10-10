from typing import Iterable, Optional, Tuple, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

class BLEUScoreCallback(Callback):
    """
    Custom callback to compute and log the BLEU score on validation data at the end of each epoch.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model being trained.
    tokenizer : Any
        The tokenizer used to convert sequences to text.
    validation_data : Iterable[Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]]
        The validation dataset.
    log_dir : str, optional
        Directory where to save the TensorBoard logs, by default None.
    max_length : int, optional
        The maximum length of the generated sequences, by default 100.

    Attributes
    ----------
    tokenizer : Any
        The tokenizer used to convert sequences to text.
    validation_data : Iterable[Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]]
        The validation dataset.
    log_dir : Optional[str]
        Directory where to save the TensorBoard logs.
    max_length : int
        The maximum length of the generated sequences.
    file_writer : Optional[tf.summary.SummaryWriter]
        TensorBoard summary writer for logging BLEU scores.
    """
    def __init__(
        self,
        tokenizer: Any,
        validation_data: Iterable[Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]],
        log_dir: Optional[str] = None,
        max_length: int = 100,
        beam_width: int = 5
    ) -> None:
        super(BLEUScoreCallback, self).__init__()
        self.tokenizer = tokenizer
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.max_length = max_length
        self.beam_width = beam_width
        self.file_writer: Optional[tf.summary.SummaryWriter] = (
            tf.summary.create_file_writer(log_dir) if log_dir else None
        )

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """
        Computes BLEU score on validation data and logs it.

        Parameters
        ----------
        epoch : int
            The index of the epoch.
        logs : dict, optional
            Dictionary of logs from the training process.
        """
        references = []
        hypotheses = []
        for (encoder_input, decoder_input), target_output in self.validation_data:
            predicted_sequences = self.model.predict_sequence_beam_search(
                encoder_input, beam_width=self.beam_width
            )

            # Ensure predicted_sequences is a list of lists
            if isinstance(predicted_sequences, list):
                predicted_sequences = np.array(predicted_sequences)
            else:
                predicted_sequences = predicted_sequences.numpy()

            predicted_texts = self.tokenizer.sequences_to_texts(predicted_sequences)
            target_texts = self.tokenizer.sequences_to_texts(target_output.numpy())

            for ref, hyp in zip(target_texts, predicted_texts):
                ref_tokens = ref.split()
                hyp_tokens = hyp.split()
                references.append([ref_tokens])
                hypotheses.append(hyp_tokens)

        # Compute BLEU score
        bleu_score = corpus_bleu(references, hypotheses)

        print(f'\nEpoch {epoch +1}: Validation BLEU score: {bleu_score:.4f}')
        if self.log_dir:
            with self.file_writer.as_default():
                tf.summary.scalar('bleu_score', bleu_score, step=epoch)