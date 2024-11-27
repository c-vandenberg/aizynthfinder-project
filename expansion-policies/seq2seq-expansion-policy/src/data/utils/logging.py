import os
import logging
from typing import Dict, List, Optional, Union, Tuple

import tensorflow as tf
from tensorflow.summary import SummaryWriter

from metrics.bleu_score import BleuScore
from metrics.smiles_string_metrics import SmilesStringMetrics

logger = logging.getLogger(__name__)

CORE_LOG_METRICS_KEY_MAPPING: Dict[str, str] = {
    'loss': 'Training Loss',
    'accuracy': 'Training Accuracy',
    'perplexity': 'Training Perplexity',
    'val_loss': 'Validation Loss',
    'val_accuracy': 'Validation Accuracy',
    'val_perplexity': 'Validation Perplexity',
}


def extract_core_log_metrics(logs: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Extracts core metrics from the provided logs based on a predefined key mapping.

    This function filters and renames the metrics from the training logs
    using the `CORE_LOG_METRICS_KEY_MAPPING`.

    Parameters
    ----------
    logs : Optional[Dict[str, float]]
        A dictionary containing log metrics from the training process.
        If `None`, an empty dictionary is returned.

    Returns
    -------
    Dict[str, float]
        A dictionary of extracted and renamed core metrics.
    """
    metrics: Dict[str, float] = {}
    if logs:
        metrics = {
            formatted_key: logs[log_key]
            for log_key, formatted_key in CORE_LOG_METRICS_KEY_MAPPING.items() if log_key in logs
        }
    return metrics

def compute_metrics(
    references: List[List[List[str]]],
    hypotheses: List[List[str]],
    target_smiles: List[str],
    predicted_smiles: List[str],
    evaluation_stage: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute all required metrics and return them as a dictionary.

    This function calculates various evaluation metrics such as BLEU score,
    Levenshtein distance, exact match accuracy, chemical validity score, and
    Tanimoto similarity. Optionally, it prefixes metric names with an evaluation stage.

    Parameters:
    ----------
    references : List[List[List[str]]]
        List of reference SMILES token strings per sample.
    hypotheses : List[List[str]]
        List of hypothesis SMILES strings per sample.
    target_smiles : List[str]
        List of target SMILES strings per sample.
    predicted_smiles : List[str]
        List of predicted SMILES strings per sample.
    evaluation_stage : Optional[str]
        Optional prefix for metric names (e.g., 'Validation'), by default None.

    Returns:
    -------
    Dict[str, float]
        Dictionary of computed metrics.

    Raises
    ------
    ValueError
        If the lengths of `references`, `hypotheses`, `target_smiles`, and
        `predicted_smiles` do not match.
    """
    if not (len(references) == len(hypotheses) == len(target_smiles) == len(predicted_smiles)):
        raise ValueError("All input lists must have the same length.")

    metrics: Dict[str, float] = {
        'BLEU score': BleuScore.smoothed_corpus_bleu(references, hypotheses),
        'Average Levenshtein Distance': SmilesStringMetrics.levenshtein_distance(target_smiles, predicted_smiles),
        'Exact Match Accuracy': SmilesStringMetrics.smiles_exact_match(target_smiles, predicted_smiles),
        'Chemical Validity Score': SmilesStringMetrics.chemical_validity(predicted_smiles),
        'Average Tanimoto Similarity': SmilesStringMetrics.average_tanimoto_similarity(target_smiles, predicted_smiles)
    }

    if evaluation_stage:
        metrics = {f"{evaluation_stage} {name}": value for name, value in metrics.items()}

    return metrics

def log_metrics(
    metrics: Dict[str, float],
    directory: str,
    epoch: Optional[Union[int, None]] = None,
    filename: Optional[str] = 'valid_metrics.txt',
    separator: Optional[str] = '-' * 40
) -> None:
    """
    Append model metrics to a specified file.

    This function logs the provided metrics by appending them to a file
    within the specified directory. Specifies epoch number if provided.

    Parameters
    ----------
    metrics : Dict[str, float]
        A dictionary of metric names and their corresponding values.
    directory : str
        The directory path where the metrics file will be saved.
    epoch : Optional[int], default=None
        The current epoch number. If provided, it will be included in the log.
    filename : Optional[str], default='valid_metrics.txt'
        The name of the metrics file.
    separator : Optional[str], default='----------------------------------------
        A string separator to delineate different logging entries.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the directory cannot be created.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    with open(filepath, "a") as f:
        if epoch is not None:
            f.write(f"Epoch {epoch + 1} Metrics\n")
        else:
            f.write(f"Testing Metrics\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
        f.write(f"{separator}\n\n\n")

def print_metrics(
    metrics: Dict[str, float],
    epoch: Optional[int] = None
) -> None:
    """
    Prints metrics to the console.

    Parameters
    ----------
    metrics : dict of str to float
        A dictionary containing metric names and their corresponding values.
    epoch : int, optional
        The current epoch number. If provided, it will be included in the printed output.

    Returns
    -------
    None
    """
    for name, value in metrics.items():
        if epoch is not None:
            logger.info(f'Epoch {epoch + 1}: {name}: {value:.4f}')
        else:
            logger.info(f'{name}: {value:.4f}')

def log_sample_predictions(
    target_smiles: List[str],
    predicted_smiles: List[str],
    directory: str,
    filename='sample_predictions.txt',
    epoch: Optional[int] = None,
    num_samples: Optional[int] = 5,
    separator_length: Optional[int] = 153
) -> None:
    """
    Logs sample predictions to a specified file.

    Parameters
    ----------
    target_smiles : list of str
        The list of target SMILES strings.
    predicted_smiles : list of str
        The list of predicted SMILES strings.
    directory : str
        The directory where the log file will be saved.
    filename : str, optional
        The name of the log file. Default is 'sample_predictions.txt'.
    epoch : int, optional
        The current epoch number. If provided, it will be included in the log.
    num_samples : int, optional
        The number of sample predictions to log. Default is 5.
    separator_length : int, optional
        The length of the separator line in the log. Default is 153.

    Returns
    -------
    None
    """
    num_samples = min(num_samples, len(target_smiles))
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    separator = '-' * separator_length

    with open(filepath, "a") as f:
        if epoch is not None:
            f.write(f"Epoch {epoch + 1} Validation Sample Predictions\n")
        else:
            f.write(f"Testing Sample Predictions\n")

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
    Prints sample predictions to the console.

    Parameters
    ----------
    target_smiles : list of str
        The list of target SMILES strings.
    predicted_smiles : list of str
        The list of predicted SMILES strings.
    num_samples : int, optional
        The number of sample predictions to print. Default is 5.
    separator_length : int, optional
        The length of the separator line in the console output. Default is 153.

    Returns
    -------
    None
    """
    num_samples = min(num_samples, len(target_smiles))
    separator = '-' * separator_length

    logger.info("\nSample Predictions:")
    for i in range(num_samples):
        logger.info(f"Sample {i + 1}:")
        logger.info(f"  Target:    {target_smiles[i]}")
        logger.info(f"  Predicted: {predicted_smiles[i]}")
        logger.info(separator)

def log_to_tensorboard(
    writer: SummaryWriter,
    metrics: Dict[str, float],
    epoch: int
):
    """
    Logs metrics to TensorBoard.

    Parameters
    ----------
    writer : tf.summary.SummaryWriter or None
        The TensorBoard summary writer. If None, no logging is performed.
    metrics : dict of str to float
        A dictionary containing metric names and their corresponding values.
    epoch : int
        The current epoch number.

    Returns
    -------
    None
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
