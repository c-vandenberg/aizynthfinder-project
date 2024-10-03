from abc import ABC, abstractmethod
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer


class AttentionInterface(Layer, ABC):
    """
    Abstract base class for attention mechanisms.

    This class defines the interface that all attention layers should implement.

    Methods
    -------
    call(inputs)
        Computes the attention context vector and attention weights.
    """
    def __init__(self, **kwargs) -> None:
        super(AttentionInterface, self).__init__(**kwargs)

    @abstractmethod
    def call(self, inputs: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes the attention context vector and attention weights.

        Parameters
        ----------
        inputs : list of tf.Tensor
            A list of tensors to apply attention on.

        Raises
        -------
        NotImplementedError: If the method is not implemented in the subclass.

        Returns
        -------
        context_vector : tf.Tensor
            The context vector resulting from the attention mechanism.

        attention_weights : tf.Tensor
            The attention weights computed over the inputs.
        """
        raise NotImplementedError('Attention layer subclasses must implement `call` method')
