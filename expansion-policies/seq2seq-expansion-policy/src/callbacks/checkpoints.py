import logging
from typing import Optional

from tensorflow.keras.callbacks import Callback
from tensorflow.train import CheckpointManager


class BestValLossCallback(Callback):
    """
    Checkpoint callback save the model when there is an improvement in validation loss.

    Parameters
    ----------
    checkpoint_manager : tf.train.CheckpointManager
        The checkpoint manager to handle saving checkpoints.

    Attributes
    ----------
    checkpoint_manager : tf.train.CheckpointManager
        Manages the saving of checkpoints.
    best_val_loss : float
        Tracks the best validation loss observed so far.

    Methods
    -------
    on_epoch_end(epoch, logs=None)
        Called at the end of each epoch to check for improvement in validation loss.
    """
    def __init__(self, checkpoint_manager: CheckpointManager) -> None:
        super(BestValLossCallback, self).__init__()
        self.checkpoint_manager: CheckpointManager = checkpoint_manager
        self.best_val_loss: float = float('inf')  # Initialize as infinity

        self._logger = logging.getLogger(__name__)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """
        Checks validation loss at the end of an epoch and saves the model if it improves.

        Parameters
        ----------
        epoch : int
            The index of the epoch.
        logs : dict, optional
            Dictionary of logs from the training process.
        """
        logs = logs or {}
        current_val_loss: float = logs.get('val_loss')
        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                save_path: str = self.checkpoint_manager.save()
                self._logger.info(
                    f"\nEpoch {epoch+1}: Validation loss improved to {current_val_loss:.4f}. "
                    f"Saving checkpoint to {save_path}"
                )
