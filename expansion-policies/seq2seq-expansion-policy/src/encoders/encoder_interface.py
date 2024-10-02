import tensorflow as tf
from abc import abstractmethod, ABCMeta
from tensorflow.keras.layers import Layer, Embedding
from typing import Optional, Any


class EncoderInterface(Layer, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super(EncoderInterface, self).__init__(**kwargs)

    @abstractmethod
    def call(self, encoder_inputs: tf.Tensor, training: Optional[bool] = None) -> Any:
        """
        Abstract method for the encoder's forward pass.

        Args:
            encoder_inputs (tf.Tensor): Input tensor for the encoder.
            training (Optional[bool], optional): Training flag. Defaults to None.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError('Encoder layer subclasses must implement `call` method')
