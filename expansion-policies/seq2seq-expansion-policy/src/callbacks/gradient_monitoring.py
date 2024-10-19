import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class GradientMonitoringCallback(Callback):
    def __init__(self, log_dir=None):
        super().__init__()
        self.log_dir = log_dir
        if log_dir:
            self.writer = tf.summary.create_file_writer(log_dir)
        else:
            self.writer = None
        self.step = 0

    def on_gradients_computed(self, gradients, variables):
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