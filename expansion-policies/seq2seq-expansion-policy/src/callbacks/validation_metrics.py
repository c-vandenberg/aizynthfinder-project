from typing import Iterable, Optional, Tuple, Any

import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from metrics.bleu_score import BleuScore
from metrics.smiles_string_metrics import SmilesStringMetrics

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
    log_dir : str, optional
        Directory where to save the TensorBoard logs, by default None.
    max_length : int, optional
        The maximum length of the generated sequences, by default 140.

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
        TensorBoard summary writer for logging metrics.
    """
    def __init__(
        self,
        tokenizer: Any,
        validation_data: Iterable[Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]],
        validation_metrics_dir: str,
        log_dir: str,
        max_length: int = 140
    ) -> None:
        super(ValidationMetricsCallback, self).__init__()
        self.tokenizer = tokenizer
        self.validation_data = validation_data
        self.validation_metrics_dir = validation_metrics_dir
        self.log_dir = log_dir
        self.max_length = max_length

        os.makedirs(self.log_dir, exist_ok=True)
        self.file_writer = tf.summary.create_file_writer(self.log_dir)

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

        for (encoder_input, decoder_input), target_output in self.validation_data:
            # Generate sequences
            predicted_sequences = self.model.predict_sequence(encoder_input, max_length=self.max_length)

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

        bleu_score = BleuScore.smoothed_corpus_bleu(references, hypotheses)
        exact_accuracy = SmilesStringMetrics.smiles_exact_match(target_smiles, predicted_smiles)
        validity_score = SmilesStringMetrics.chemical_validity(predicted_smiles)
        ave_levenshtein_distance = SmilesStringMetrics.levenshtein_distance(target_smiles, predicted_smiles)

        os.makedirs(self.validation_metrics_dir, exist_ok=True)
        with open(os.path.join(self.validation_metrics_dir, 'valid_metrics.txt'), "a") as f:
            f.write(f"Epoch {epoch + 1} Validation Metrics\n")
            f.write(f"BLEU score: {bleu_score:.4f}\n")
            f.write(f"Exact Match Accuracy: {exact_accuracy:.4f}\n")
            f.write(f"Chemical Validity Score: {validity_score:.4f}\n")
            f.write(f"Average Levenshtein Distance: {ave_levenshtein_distance:.4f}\n")
            f.write(f"{'-' * 40}\n")
            f.write(f"\n \n")

        print(f'Epoch {epoch + 1}: BLEU score: {bleu_score:.4f}')
        print(f'Epoch {epoch + 1}: Exact Match Accuracy: {exact_accuracy:.4f}')
        print(f'Epoch {epoch + 1}: Chemical Validity Score: {validity_score:.4f}')
        print(f'Epoch {epoch + 1}: Average Levenshtein Distance: {ave_levenshtein_distance:.4f}')

        num_samples = min(5, len(target_smiles))
        with open(os.path.join(self.validation_metrics_dir, 'sample_predictions.txt'), "a") as f:
            f.write(f"Epoch {epoch + 1} Sample Predictions\n")
            for i in range(num_samples):
                f.write(f"Sample {i + 1}:\n")
                f.write(f"  Target:    {target_smiles[i]}\n")
                f.write(f"  Predicted: {predicted_smiles[i]}\n")
                f.write(f"{'-' * 153}\n")
            f.write(f"\n \n")

        print("\nSample Predictions:")
        for i in range(num_samples):
            print(f"Sample {i + 1}:")
            print(f"  Target:    {target_smiles[i]}")
            print(f"  Predicted: {predicted_smiles[i]}")
            print("-" * 153)

        # Log Metrics to TensorBoard
        if self.log_dir and self.file_writer:
            with self.file_writer.as_default():
                tf.summary.scalar('bleu_score', bleu_score, step=epoch)
                tf.summary.scalar('exact_match', exact_accuracy, step=epoch)
                tf.summary.scalar('chem_validity', validity_score, step=epoch)
                tf.summary.scalar('ave_levenshtein_dist', ave_levenshtein_distance, step=epoch)
