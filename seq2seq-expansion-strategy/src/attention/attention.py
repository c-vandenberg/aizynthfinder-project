import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from attention.attention_interface import AttentionInterface
from typing import List, Optional, Tuple, Union


class BahdanauAttention(AttentionInterface):
    def __init__(self, units: int, **kwargs):
        super(BahdanauAttention, self).__init__(units, **kwargs)
        self.units: int = units
        self.attention_dense1: Dense = Dense(units, name='attention_dense1')
        self.attention_dense2: Dense = Dense(units, name='attention_dense2')
        self.attention_v: Dense = Dense(1, name='attention_v')
        self.supports_masking: bool = True

    def call(self, inputs: List[tf.Tensor], mask: Optional[tf.Tensor] = None,
             training: Union[None, bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the context vector and attention weights.

        Args:
            inputs (List[tf.Tensor]): A list containing encoder and decoder outputs.
            mask (Optional[tf.Tensor], optional): Mask tensor. Defaults to None.
            training (Union[None, bool], optional): Training flag. Defaults to None.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Context vector and attention weights.
        """
        # Unpack inputs
        encoder_output, decoder_output = inputs

        # Attention Mechanism
        # Calculate attention scores
        # Expand dimensions to match the shapes for broadcasting
        encoder_output_expanded: tf.Tensor = tf.expand_dims(encoder_output,
                                                 1)  # Shape: (batch_size, 1, seq_len_encoder, units*2)
        decoder_output_expanded: tf.Tensor = tf.expand_dims(decoder_output,
                                                 2)  # Shape: (batch_size, seq_len_decoder, 1, units)

        # Compute the attention scores
        score: tf.Tensor = tf.nn.tanh(
            self.attention_dense1(encoder_output_expanded) + self.attention_dense2(decoder_output_expanded)
        )  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, units)

        # Apply mask if available
        if mask is not None:
            # If mask is a list or tuple, both encoder and decoder mask have been passed.
            # Extract the encoder mask
            if isinstance(mask, (list, tuple)):
                encoder_mask: tf.Tensor = mask[0]
            else:
                encoder_mask = mask
            if encoder_mask is not None:
                # mask shape: (batch_size, seq_len_encoder)
                # Expand mask to match score dimensions
                encoder_mask = tf.cast(tf.expand_dims(encoder_mask, 1), dtype=score.dtype)  # (batch_size, 1, seq_len_encoder)
                encoder_mask = tf.expand_dims(encoder_mask, -1)  # (batch_size, 1, seq_len_encoder, 1)
                # Add a large negative value to masked positions to nullify their effect after softmax
                score += (1.0 - encoder_mask) * -1e9

        attention_weights: tf.Tensor = tf.nn.softmax(self.attention_v(score),
                                          axis=2)  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, 1)

        # Compute the context vector
        context_vector: tf.Tensor = attention_weights * encoder_output_expanded  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, units*2)
        context_vector: tf.Tensor = tf.reduce_sum(context_vector, axis=2)  # Shape: (batch_size, seq_len_decoder, units*2)

        return context_vector, attention_weights

    @staticmethod
    def compute_mask(inputs: List[tf.Tensor], mask: Optional[tf.Tensor] = None) -> None:
        # This layer does not propagate the mask further
        return None

    def get_config(self) -> dict:
        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict: A Python dictionary containing the layer's configuration.
        """
        config = super(BahdanauAttention, self).get_config()
        config.update({
            'units': self.units,
            'attention_dense1': tf.keras.layers.serialize(self.attention_dense1),
            'attention_dense2': tf.keras.layers.serialize(self.attention_dense2),
            'attention_v': tf.keras.layers.serialize(self.attention_v),
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'BahdanauAttention':
        """
        Creates a layer from its config.

        Args:
            config (dict): A Python dictionary containing the layer's configuration.

        Returns:
            BahdanauAttention: A new instance of BahdanauAttention configured using the provided config.
        """
        # Deserialize layers
        config['attention_dense1'] = tf.keras.layers.deserialize(config['attention_dense1'])
        config['attention_dense2'] = tf.keras.layers.deserialize(config['attention_dense2'])
        config['attention_v'] = tf.keras.layers.deserialize(config['attention_v'])
        return cls(**config)
