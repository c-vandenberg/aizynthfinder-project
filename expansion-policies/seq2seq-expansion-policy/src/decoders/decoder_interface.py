from abc import ABC, abstractmethod
from typing import Optional, Any

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding


class DecoderInterface(Layer, ABC):
    """
    Abstract base class for decoder layers.

    This class defines the interface that all decoder layers should implement.

    Methods
    -------
    call(inputs, training=None, mask=None)
        Forward pass of the decoder.
    """
    def __init__(self, **kwargs) -> None:
        super(DecoderInterface, self).__init__(**kwargs)

    @abstractmethod
    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None
    ) -> Any:
        """
        Forward pass of the decoder.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        training : bool, optional
            Training flag, by default None.
        mask : tf.Tensor, optional
            Mask tensor, by default None.

        Raises
        -------
        NotImplementedError: If the method is not implemented in the subclass.

        Returns
        -------
        Any
            Output of the decoder.
        """
        raise NotImplementedError('Decoder layer subclasses must implement `call` method')
