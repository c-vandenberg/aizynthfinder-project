import tensorflow as tf
from abc import abstractmethod, ABCMeta
from tensorflow.keras.layers import Layer, Embedding
from typing import Optional, Any


class DecoderInterface(Layer, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super(DecoderInterface, self).__init__(**kwargs)

    @abstractmethod
    def call(self,inputs: tf.Tensor,training: Optional[bool] = None,mask: Optional[tf.Tensor] = None) -> Any:
        """
        Abstract method for the decoder's forward pass.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (Optional[bool], optional): Training flag. Defaults to None.
            mask (Optional[tf.Tensor], optional): Mask tensor. Defaults to None.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError('Decoder layer subclasses must implement `call` method')
