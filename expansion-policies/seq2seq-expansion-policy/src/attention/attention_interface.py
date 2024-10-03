import tensorflow as tf
from abc import abstractmethod, ABCMeta
from tensorflow.keras.layers import Layer, Embedding
from typing import List, Tuple

class AttentionInterface(Layer, metaclass=ABCMeta):
    def __init__(self, **kwargs) -> None:
        super(AttentionInterface, self).__init__(**kwargs)

    @abstractmethod
    def call(self, outputs: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Abstract method to compute attention.

        Args:
            outputs (List[tf.Tensor]): A list of tensors to apply attention on.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the context vector and attention weights.
        """
        raise NotImplementedError('Attention layer subclasses must implement `call` method')
