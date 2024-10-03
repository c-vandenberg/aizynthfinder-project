from tensorflow.keras.callbacks import Callback
from tensorflow.train import CheckpointManager

class BestValLossCheckpointCallback(Callback):
    def __init__(self, checkpoint_manager: CheckpointManager) -> None:
        super(BestValLossCheckpointCallback, self).__init__()
        self.checkpoint_manager: CheckpointManager = checkpoint_manager
        self.best_val_loss: float = float('inf')  # Initialize with infinity

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss: float = logs.get('val_loss')
        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                save_path: str = self.checkpoint_manager.save()
                print(
                    f"\nEpoch {epoch+1}: Validation loss improved to {current_val_loss:.4f}. "
                    f"Saving checkpoint to {save_path}"
                )
