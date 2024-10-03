from abc import ABC, abstractmethod
from typing import Optional, Any

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding


class EncoderInterface(Layer, ABC):
    """
    Abstract base class for encoder layers.

    This class defines the interface that all encoder layers should implement.

    Methods
    -------
    call(encoder_inputs, training=None)
        Forward pass of the encoder.
    """
    def __init__(self, **kwargs):
        super(EncoderInterface, self).__init__(**kwargs)

    @abstractmethod
    def call(
        self,
        encoder_inputs: tf.Tensor,
        training: Optional[bool] = None
    ) -> Any:
        """
        Forward pass of the encoder.

        Parameters
        ----------
        encoder_inputs : tf.Tensor
            Input tensor for the encoder.
        training : bool, optional
            Training flag, by default None.

        Raises
        -------
        NotImplementedError: If the method is not implemented in the subclass.

        Returns
        -------
        Any
            Output of the encoder.
        """
        raise NotImplementedError('Encoder layer subclasses must implement `call` method')
