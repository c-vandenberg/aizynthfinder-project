from typing import Any, Callable, Dict, Optional

import tensorflow as tf
from tensorflow.keras.metrics import Mean

@tf.keras.utils.register_keras_serializable()
class Perplexity(Mean):
    """
    Perplexity

    Computes the perplexity metric, which is a common evaluation metric for language models.

    Perplexity measures how well a probability model predicts a sample and is defined as the exponential
    of the cross-entropy loss. Lower perplexity indicates better model performance.

    Parameters
    ----------
    loss_function : Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
        A function that computes the loss given true and predicted values.
    name : str, optional
        Name of the metric (default is 'perplexity').
    **kwargs : Any
        Additional keyword arguments passed to the base `Mean` class.

    Attributes
    ----------
    _loss_function : Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
        The loss function used to compute the loss.
    """
    def __init__(
        self,
        loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        name:str ='perplexity',
        **kwargs
    ) -> None:
        super(Perplexity, self).__init__(name=name, **kwargs)
        self._loss_function = loss_function

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """
        Accumulate the mean loss.

        Computes the loss using the provided loss function and updates the internal state
        to keep track of the mean loss.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth values with shape `(batch_size, sequence_length)`.
        y_pred : tf.Tensor
            The predicted values with shape `(batch_size, sequence_length, vocab_size)`.
        sample_weight : Optional[tf.Tensor], default=None
            Weights for the samples, by default None.

        Raises
        ------
        ValueError
            If `y_true` or `y_pred` have incompatible shapes.
        """
        # Compute the loss per sample and per timestep
        loss = self._loss_function(y_true, y_pred)  # Shape: (batch_size, sequence_length)

        # Reduce the loss to a scalar value
        loss = tf.reduce_mean(loss)  # Scalar mean loss

        # Update the state using the base class method
        super().update_state(loss, sample_weight=sample_weight)

    def result(self) -> tf.Tensor:
        """
        Compute perplexity as the exponential of the mean loss.

        Returns
        -------
        tf.Tensor
            The computed perplexity as the exponential of the mean loss.

        Notes
        -----
        Perplexity is defined as `exp(loss)`, where `loss` is the cross-entropy loss.
        """
        mean_loss = super().result()
        return tf.exp(mean_loss)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the metric for serialization.

        This configuration can be used to re-instantiate the metric with the same parameters.

        Returns
        -------
        config : Dict[str, Any]
            Configuration dictionary containing all necessary parameters to recreate the metric.
        """
        config = super(Perplexity, self).get_config()
        config.update({
            'loss_function': tf.keras.losses.serialize(self._loss_function),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Perplexity':
        """
        Creates an instance of the metric from its configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        Perplexity
            An instance of the Perplexity metric configured as per the provided dictionary.
        """
        loss_function = tf.keras.losses.deserialize(config.pop('loss_function'))
        return cls(loss_function=loss_function, **config)
