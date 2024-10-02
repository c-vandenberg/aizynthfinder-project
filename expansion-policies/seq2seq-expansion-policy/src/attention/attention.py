import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from attention.attention_interface import AttentionInterface
from typing import List, Optional, Tuple, Union

@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(AttentionInterface):
    """
    BahdanauAttention

    Implements the Bahdanau attention mechanism for Seq2Seq models, allowing the decoder to dynamically focus on
    different parts of the encoder processed input sequence at each decoding time step.

    Parameters
    ----------
    units : int
        Number of units in the attention mechanism.

    Methods
    -------
    call(inputs, mask=None, training=None)
        Computes the context vector and attention weights.
    """
    def __init__(self, units: int, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.attention_dense1 = Dense(units, name='attention_dense1')
        self.attention_dense2 = Dense(units, name='attention_dense2')
        self.attention_v = Dense(1, name='attention_v')
        self.supports_masking = True

    def call(self, inputs: List[tf.Tensor], mask: Optional[tf.Tensor] = None,
             training: Union[None, bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes the context vector and attention weights using the Bahdanau attention mechanism.

        Parameters
        ----------
        inputs : list of tf.Tensor
            A list containing:
                - encoder_output : tf.Tensor
                    The outputs from the encoder.
                    Shape: (batch_size, seq_len_enc, enc_units)
                - decoder_output : tf.Tensor
                    The outputs from the decoder before attention is applied.
                    Shape: (batch_size, seq_len_dec, dec_units)
        mask : tf.Tensor or list of tf.Tensor, optional
            Mask tensor or list of masks to prevent attention over padded positions.
            If a list, the first element should be the encoder mask.
            Shape: (batch_size, seq_len_enc)
        training : bool, optional
            Indicates whether the layer should behave in training mode or inference mode.

        Returns
        -------
        context_vector : tf.Tensor
            The context vector computed as a weighted sum of encoder outputs.
            Shape: (batch_size, seq_len_dec, enc_units)
        attention_weights : tf.Tensor
            The attention weights for each encoder output.
            Shape: (batch_size, seq_len_dec, seq_len_enc)
        """
        # Unpack inputs
        encoder_output: tf.Tensor  # Shape: (batch_size, seq_len_enc, enc_units)
        decoder_output: tf.Tensor  # Shape: (batch_size, seq_len_dec, dec_units)
        encoder_output, decoder_output = inputs

        # Compute attention scores for encoder outputs
        score_enc: tf.Tensor = self.attention_dense1(encoder_output)  # Shape: (batch_size, seq_len_enc, units)
        score_dec: tf.Tensor = self.attention_dense2(decoder_output)  # Shape: (batch_size, seq_len_dec, units)

        # Compute attention scores for decoder outputs
        # Expand dimensions to enable broadcasting for addition
        score_enc_expanded: tf.Tensor = tf.expand_dims(score_enc, axis=1) # Shape: (batch_size, 1, seq_len_enc, units)
        score_dec_expanded: tf.Tensor = tf.expand_dims(score_dec, axis=2) # Shape: (batch_size, seq_len_dec, 1, units)

        # Calculate the combined score using tanh activation
        score_combined: tf.Tensor = tf.nn.tanh(
            score_enc_expanded + score_dec_expanded
        )  # Shape: (batch_size, seq_len_dec, seq_len_enc, units)

        # Compute final attention scores
        score: tf.Tensor = self.attention_v(score_combined)  # Shape: (batch_size, seq_len_dec, seq_len_enc, 1)

        # Remove the last dimension of size 1
        score = tf.squeeze(score, axis=-1)  # Shape: (batch_size, seq_len_dec, seq_len_enc)

        # Apply encoder mask if available
        if mask is not None:
            if isinstance(mask, (list, tuple)):
                encoder_mask: Optional[tf.Tensor] = mask[0] # Shape: (batch_size, seq_len_enc)
            else:
                encoder_mask = mask # Shape: (batch_size, seq_len_enc)
            if encoder_mask is not None:
                # Expand mask dimensions to align with score tensor
                encoder_mask_expanded = tf.expand_dims(encoder_mask, axis=1)  # Shape: (batch_size, 1, seq_len_enc)

                # Cast mask to float and adjust score
                score += (1.0 - tf.cast(encoder_mask_expanded, score.dtype)) * -1e9

        # Compute attention weights using softmax over the encoder sequence length
        attention_weights = tf.nn.softmax(score, axis=-1)  # Shape: (batch_size, seq_len_dec, seq_len_enc)

        # Compute context vector as weighted sum of encoder outputs
        context_vector = tf.matmul(attention_weights, encoder_output)  # Shape: (batch_size, seq_len_dec, enc_units)

        return context_vector, attention_weights

    @staticmethod
    def compute_mask(inputs: List[tf.Tensor], mask: Optional[tf.Tensor] = None) -> None:
        # This layer does not propagate the mask further
        return None

    def get_config(self) -> dict:
        config = super(BahdanauAttention, self).get_config()
        config.update({
            'units': self.units,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'BahdanauAttention':
        return cls(**config)
