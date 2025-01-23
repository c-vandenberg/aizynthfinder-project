from typing import List, Optional, Tuple, Union, Any, Dict

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

from attention.attention_interface import AttentionInterface


@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(AttentionInterface):
    """
    BahdanauAttention

    Implements the Bahdanau attention mechanism for Seq2Seq models, enabling the decoder to dynamically
    focus on different parts of the encoder's processed input sequence at each decoding time step.

    This enhances the model's ability to capture relevant contextual information, improving translation
    quality and performance.

    Architecture:
        - Attention Dense Layers: Two Dense layers (`attention_dense1` and `attention_dense2`) transform
                                    the encoder and decoder outputs into a common space to compute attention
                                    scores.
        - Score Calculation: Combines transformed encoder and decoder outputs using `tanh` activation to
                                    compute raw attention scores.
        - Attention Weights: A Dense layer (`attention_v`) projects the combined scores to a scalar,
                                    followed by a `softmax` activation to obtain normalized attention weights.
        - Context Vector: Computes a weighted sum of encoder outputs based on the attention weights,
                                    producing the context vector that encapsulates relevant information from
                                    the encoder.
        - Mask Handling: Applies masking to prevent the model from attending to padded positions in the
                                    encoder outputs, ensuring that attention focuses only on meaningful tokens.

    Parameters
    ----------
    units : int
        Number of units in the attention mechanism. Determines the dimensionality of the transformation
        applied to the encoder and decoder outputs.

    Methods
    -------
    call(inputs, mask=None, training=None)
        Computes the context vector and attention weights based on the encoder and decoder outputs.

    compute_mask(inputs, mask=None)
        Computes the mask for the given inputs. This layer does not propagate the mask further.

    get_config()
        Returns the configuration of the BahdanauAttention layer for serialization.

    from_config(config)
        Creates a BahdanauAttention layer from its configuration.

    Returns
    -------
    context_vector : tf.Tensor
        The context vector computed as a weighted sum of encoder outputs.
        Shape: (batch_size, seq_len_dec, enc_units)

    attention_weights : tf.Tensor
        The attention weights for each encoder output.
        Shape: (batch_size, seq_len_dec, seq_len_enc)
    """
    def __init__(self, units: int, **kwargs) -> None:
        super(BahdanauAttention, self).__init__(**kwargs)
        self._units = units
        self._attention_dense1 = Dense(units, name='attention_dense1')
        self._attention_dense2 = Dense(units, name='attention_dense2')
        self._attention_v = Dense(1, name='attention_v')
        self._supports_masking = True

    def call(
        self,
        inputs: List[tf.Tensor],
        mask: Optional[tf.Tensor] = None,
        training: Union[None, bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes the context vector and attention weights using the Bahdanau attention mechanism.

        Parameters
        ----------
        inputs : List[tf.Tensor]
            A list containing:
                - encoder_output : tf.Tensor
                    The outputs from the encoder.
                    Shape: (batch_size, seq_len_enc, enc_units)
                - decoder_output : tf.Tensor
                    The outputs from the decoder before attention is applied.
                    Shape: (batch_size, seq_len_dec, dec_units)
        mask : Optional[Union[tf.Tensor, List[tf.Tensor]]], default=None
            Mask tensor or list of masks to prevent attention over padded positions.
            If a list, the first element should be the encoder mask.
            Shape: (batch_size, seq_len_enc)
        training : Optional[bool], default=None
            Indicates whether the layer should behave in training mode or inference mode.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            - context_vector : tf.Tensor
                The context vector computed as a weighted sum of encoder outputs.
                Shape: (batch_size, seq_len_dec, enc_units)
            - attention_weights : tf.Tensor
                The attention weights for each encoder output.
                Shape: (batch_size, seq_len_dec, seq_len_enc)

        Raises
        ------
        ValueError
            If the `inputs` list does not contain exactly two tensors (encoder_output and decoder_output).
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(
                "BahdanauAttention expects a list of two tensors as inputs: "
                "encoder_output and decoder_output."
            )

        # Unpack inputs
        encoder_output: tf.Tensor  # Shape: (batch_size, seq_len_enc, enc_units)
        decoder_output: tf.Tensor  # Shape: (batch_size, seq_len_dec, dec_units)
        encoder_output, decoder_output = inputs

        # Transform encoder and decoder outputs
        score_enc: tf.Tensor = self._attention_dense1(encoder_output)  # Shape: (batch_size, seq_len_enc, units)
        score_dec: tf.Tensor = self._attention_dense2(decoder_output)  # Shape: (batch_size, seq_len_dec, units)

        # Expand dimensions to enable broadcasting for addition
        score_enc_expanded: tf.Tensor = tf.expand_dims(score_enc, axis=1) # Shape: (batch_size, 1, seq_len_enc, units)
        score_dec_expanded: tf.Tensor = tf.expand_dims(score_dec, axis=2) # Shape: (batch_size, seq_len_dec, 1, units)

        # Calculate the combined score using tanh activation
        score_combined: tf.Tensor = tf.nn.tanh(
            score_enc_expanded + score_dec_expanded
        )  # Shape: (batch_size, seq_len_dec, seq_len_enc, units)

        # Compute final attention scores
        score: tf.Tensor = self._attention_v(score_combined)  # Shape: (batch_size, seq_len_dec, seq_len_enc, 1)

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
                # Adding a large negative value to masked positions to nullify their effect in softmax
                score += (1.0 - tf.cast(encoder_mask_expanded, score.dtype)) * -1e9

        # Compute attention weights using softmax over the encoder sequence length
        attention_weights = tf.nn.softmax(score, axis=-1)  # Shape: (batch_size, seq_len_dec, seq_len_enc)

        # Compute context vector as weighted sum of encoder outputs
        context_vector = tf.matmul(attention_weights, encoder_output)  # Shape: (batch_size, seq_len_dec, enc_units)

        return context_vector, attention_weights

    @staticmethod
    def compute_mask(inputs: List[tf.Tensor], mask: Optional[tf.Tensor] = None) -> None:
        """
        Computes the mask for the given inputs.

        This layer does not propagate the mask further.

        Parameters
        ----------
        inputs : List[tf.Tensor]
            A list of input tensors.
        mask : Optional[Union[tf.Tensor, List[tf.Tensor]]], default=None
            An optional mask tensor or list of masks. If a list, the first element should be the encoder mask.

        Returns
        -------
        None
            This method does not return any value.
        """
        return None

    def get_config(self) -> dict:
        """
        Returns the configuration of the BahdanauAttention layer.

        This configuration can be used to re-instantiate the layer, preserving its settings.

        Returns
        -------
        config : Dict[str, Any]
            A dictionary containing the configuration of the layer, including the number of units.
        """
        config = super(BahdanauAttention, self).get_config()
        config.update({
            'units': self._units,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'BahdanauAttention':
        """
        Creates a BahdanauAttention layer from its configuration.

        This class method allows the creation of a new `BahdanauAttention` instance
        from a configuration dictionary, enabling model reconstruction from saved configurations.

        Parameters
        ----------
        config : Dict[str, Any]
            A dictionary containing the configuration of the layer.

        Returns
        -------
        BahdanauAttention
            An instance of BahdanauAttention configured as per the provided config.
        """
        return cls(**config)
