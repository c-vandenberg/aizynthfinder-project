from typing import Optional, List

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class GradientMonitoringCallback(Callback):
    """
    Monitors and logs gradient norms during training.

    This callback computes the average gradient norm and optionally logs individual
    gradient histograms to TensorBoard.

    Parameters
    ----------
    log_dir : Optional[str], default=None
        Directory where TensorBoard logs will be saved. If None, logging is disabled.
    """
    def __init__(self, log_dir: Optional[str] = None) -> None:
        super().__init__()
        self.log_dir: Optional[str] = log_dir
        if log_dir:
            self.writer: Optional[tf.summary.SummaryWriter] = tf.summary.create_file_writer(log_dir)
        else:
            self.writer = None
        self.step: int = 0

    def on_gradients_computed(
        self,
        gradients: List[Optional[tf.Tensor]],
        variables: List[tf.Variable]
    ) -> None:
        """
        Called when gradients are computed during training.

        Processes the gradients by computing their norms and logging the average
        gradient norm. Optionally logs histograms of individual gradients.

        Parameters
        ----------
        gradients : List[Optional[tf.Tensor]]
            List of gradient tensors computed for each trainable variable.
        variables : List[tf.Variable]
            List of variables corresponding to the gradients.

        Returns
        -------
        None
            This method does not return any value.
        """
        # Process gradients here
        grad_norms = [tf.norm(grad) for grad in gradients if grad is not None]
        avg_grad_norm = tf.reduce_mean(grad_norms)

        if self.writer:
            with self.writer.as_default():
                tf.summary.scalar('avg_grad_norm', avg_grad_norm, step=self.step)
                # Optionally log individual gradients
                for grad, var in zip(gradients, variables):
                    if grad is not None:
                        tf.summary.histogram(f'gradients/{var.name}', grad, step=self.step)
            self.writer.flush()
        self.step += 1