import os
from typing import Dict, List, Optional

import tensorflow as tf
from tensorflow.summary import SummaryWriter

from metrics.bleu_score import BleuScore
from metrics.smiles_string_metrics import SmilesStringMetrics

def compute_metrics(
    references,
    hypotheses,
    target_smiles: List[str],
    predicted_smiles: List[str]
) -> Dict[str, float]:
    """
       Compute all required metrics and return them as a dictionary.
       """
    return {
        'BLEU score': BleuScore.smoothed_corpus_bleu(references, hypotheses),
        'Exact Match Accuracy': SmilesStringMetrics.smiles_exact_match(target_smiles, predicted_smiles),
        'Chemical Validity Score': SmilesStringMetrics.chemical_validity(predicted_smiles),
        'Average Levenshtein Distance': SmilesStringMetrics.levenshtein_distance(target_smiles, predicted_smiles),
    }

def log_metrics(
    epoch: int,
    metrics: Dict[str, float],
    directory: str,
    filename: Optional[str] = 'valid_metrics.txt',
    separator: Optional[str] = '-'*40
) -> None:
    """
    Append model metrics to a specified file.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    with open(filepath, "a") as f:
        f.write(f"Epoch {epoch + 1} Validation Metrics\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
        f.write(f"{separator}\n\n\n")

def print_metrics(epoch: int, metrics: Dict[str, float]) -> None:
    """
    Print metrics to the console.
    """
    for name, value in metrics.items():
        print(f'Epoch {epoch + 1}: {name}: {value:.4f}')

def log_sample_predictions(
    epoch: int,
    target_smiles: List[str],
    predicted_smiles: List[str],
    directory: str,
    filename='sample_predictions.txt',
    num_samples: Optional[int] = 5,
    separator_length: Optional[int] = 153
) -> None:
    """
    Log sample predictions to a specified file.
    """
    num_samples = min(num_samples, len(target_smiles))
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    separator = '-' * separator_length

    with open(filepath, "a") as f:
        f.write(f"Epoch {epoch + 1} Sample Predictions\n")
        for i in range(num_samples):
            f.write(f"Sample {i + 1}:\n")
            f.write(f"  Target:    {target_smiles[i]}\n")
            f.write(f"  Predicted: {predicted_smiles[i]}\n")
            f.write(f"{separator}\n")
        f.write("\n\n")

def print_sample_predictions(
    target_smiles: List[str],
    predicted_smiles: List[str],
    num_samples: int = 5,
    separator_length: int = 153
) -> None:
    """
    Print sample predictions to the console.
    """
    num_samples = min(num_samples, len(target_smiles))
    separator = '-' * separator_length

    print("\nSample Predictions:")
    for i in range(num_samples):
        print(f"Sample {i + 1}:")
        print(f"  Target:    {target_smiles[i]}")
        print(f"  Predicted: {predicted_smiles[i]}")
        print(separator)

def log_to_tensorboard(
    writer: SummaryWriter,
    metrics: Dict[str, float],
    epoch: int
):
    """
    Log metrics to TensorBoard.
    """
    if writer is not None:
        with writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(
                    name=name.replace(" ", "_").lower(),
                    data=value,
                    step=epoch
                )
        writer.flush()
