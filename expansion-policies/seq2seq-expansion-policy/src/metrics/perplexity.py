from typing import Callable, Optional

import tensorflow as tf
from tensorflow.keras.metrics import Mean

@tf.keras.utils.register_keras_serializable()
class Perplexity(Mean):
    def __init__(
        self,
        loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        name:str ='perplexity',
        **kwargs
    ) -> None:
        """
        Initialize the Perplexity metric.

        Parameters
        ----------
        loss_function : Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
            A function that computes the loss given true and predicted values.
        name : str, optional
            Name of the metric, by default 'perplexity'.
        **kwargs
            Additional keyword arguments passed to the base `Mean` class.

        """
        super(Perplexity, self).__init__(name=name, **kwargs)
        self.loss_function = loss_function

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """
        Accumulate the mean loss.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth values.
        y_pred : tf.Tensor
            The predicted values.
        sample_weight : tf.Tensor, optional
            Weights for the samples, by default None.
        """
        # Compute the loss per sample and per timestep
        loss = self.loss_function(y_true, y_pred)  # Shape: (batch_size, sequence_length)

        # Reduce the loss to a scalar value
        loss = tf.reduce_mean(loss)  # Scalar mean loss over batch and sequence

        # Update the state using the base class method
        super().update_state(loss, sample_weight=sample_weight)

    def result(self) -> tf.Tensor:
        """
        Compute perplexity as the exponential of the mean loss, and return the final metric value.

        Returns
        -------
        tf.Tensor
            The computed perplexity as the exponential of the mean loss.

        """
        mean_loss = super().result()
        return tf.exp(mean_loss)

    def get_config(self):
        config = super(Perplexity, self).get_config()
        config.update({
            'loss_function': tf.keras.losses.serialize(self.loss_function),
        })
        return config

    @classmethod
    def from_config(cls, config):
        loss_function = tf.keras.losses.deserialize(config.pop('loss_function'))
        return cls(loss_function=loss_function, **config)
