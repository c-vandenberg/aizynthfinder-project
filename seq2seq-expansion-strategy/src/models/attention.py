import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from models.interfaces import AttentionInterface
from typing import List, Optional, Tuple


class BahdanauAttention(Layer, AttentionInterface):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.units = units
        self.attention_dense1 = Dense(units, name='attention_dense1')
        self.attention_dense2 = Dense(units, name='attention_dense2')
        self.attention_v = Dense(1, name='attention_v')
        self.supports_masking = True

    def call(self, inputs: List[tf.Tensor], mask: Optional[tf.Tensor] = None,
             training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        # Unpack inputs
        encoder_output, decoder_output = inputs

        # Attention Mechanism
        # Calculate attention scores
        # Expand dimensions to match the shapes for broadcasting
        encoder_output_expanded = tf.expand_dims(encoder_output,
                                                 1)  # Shape: (batch_size, 1, seq_len_encoder, units*2)
        decoder_output_expanded = tf.expand_dims(decoder_output,
                                                 2)  # Shape: (batch_size, seq_len_decoder, 1, units)

        # Compute the attention scores
        score = tf.nn.tanh(
            self.attention_dense1(encoder_output_expanded) + self.attention_dense2(decoder_output_expanded)
        )  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, units)

        # Apply mask if available
        if mask is not None:
            # mask shape: (batch_size, seq_len_encoder)
            # Expand mask to match score dimensions
            mask = tf.cast(tf.expand_dims(mask, 1), dtype=score.dtype)  # (batch_size, 1, seq_len_encoder)
            mask = tf.expand_dims(mask, -1)  # (batch_size, 1, seq_len_encoder, 1)
            # Add a large negative value to masked positions to nullify their effect after softmax
            score += (1.0 - mask) * -1e9

        attention_weights = tf.nn.softmax(self.attention_v(score),
                                          axis=2)  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, 1)

        # Compute the context vector
        context_vector = attention_weights * encoder_output_expanded  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, units*2)
        context_vector = tf.reduce_sum(context_vector, axis=2)  # Shape: (batch_size, seq_len_decoder, units*2)

        return context_vector, attention_weights

    @staticmethod
    def compute_mask(inputs, mask=None):
        # This layer does not propagate the mask further
        return None
