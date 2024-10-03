from typing import Optional, Union, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (Embedding, Bidirectional, LSTM,
                                     Dropout, Dense, Layer, LayerNormalization)

from encoders.encoder_interface import EncoderInterface


@tf.keras.utils.register_keras_serializable()
class StackedBidirectionalLSTMEncoder(EncoderInterface):
    """
    StackedBidirectionalLSTMEncoder

    A custom TensorFlow Keras layer that encodes input sequences into context-rich representations for a Seq2Seq model.

    Architecture:
        - Embedding Layer: Converts input token indices into dense embedding vectors.
        - Stacked Bidirectional LSTM Layers: Processes embeddings in both forward and backward directions to capture
                                                 context from past and future tokens.
        - Dropout Layers: Applies dropout after each LSTM layer to prevent overfitting.

    Encoder output is the processed sequence representations, along with the concatenated hidden and cell states from
    the last Bidirectional LSTM layer. These states serve as the initial states for the decoder.

    Parameters
    ----------
    vocab_size : int
        Size of the input vocabulary.
    embedding_dim : int
        Dimensionality of the embedding vectors.
    units : int
        Number of units in each LSTM layer.
    num_layers : int, optional
        Number of stacked Bidirectional LSTM layers (default is 2).
    dropout_rate : float, optional
        Dropout rate applied after each LSTM layer (default is 0.2).

    Methods
    -------
    call(encoder_input, training=False)
        Encodes the input sequence and returns the encoder outputs and final states.

    Returns
    -------
    encoder_output : tf.Tensor
        Encoded sequence representations.
    final_state_h : tf.Tensor
        Concatenated hidden state from the last Bidirectional LSTM layer.
    final_state_c : tf.Tensor
        Concatenated cell state from the last Bidirectional LSTM layer.
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_embedding_dim: int,
        units: int,
        num_layers: int,
        dropout_rate: float = 0.2,
        **kwargs
    ) -> None:
        super(StackedBidirectionalLSTMEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, encoder_embedding_dim, mask_zero=True)
        self.units= units
        self.num_layers = num_layers
        self.dropout_rate= dropout_rate

        self.supports_masking = True

        # Build Bidirectional LSTM, Dropout, and LayerNormalization layers
        self.bidirectional_lstm_layers = []
        self.dropout_layers = []
        self.layer_norm_layers = []
        for i in range(num_layers):
            lstm_layer = Bidirectional(
                LSTM(units, return_sequences=True, return_state=True),
                name=f'bidirectional_lstm_encoder_{i + 1}'
            )
            self.bidirectional_lstm_layers.append(lstm_layer)

            dropout_layer = Dropout(dropout_rate, name=f'encoder_dropout_{i + 1}')
            self.dropout_layers.append(dropout_layer)

            layer_norm_layer = LayerNormalization(name=f'encoder_layer_norm_{i + 1}')
            self.layer_norm_layers.append(layer_norm_layer)

    def call(
        self,
        encoder_input: tf.Tensor,
        training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Encodes the input sequence and returns the encoder outputs and final states.

        Parameters
        ----------
        encoder_input: tf.Tensor Input tensor for the encoder.
        training: bool, optional
            Training flag. Defaults to None.

        Returns
        ----------
        encoder output, final hidden state, final cell state: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            Encoder processed sequence and final encoder context vectors.
        """
        # Embed the input and obtain mask
        encoder_output: tf.Tensor = self.embedding(encoder_input) # Shape: (batch_size, seq_len, embedding_dim)

        final_state_h: Union[None, tf.Tensor] = None
        final_state_c: Union[None, tf.Tensor] = None

        for i, (lstm_layer, dropout_layer, layer_norm_layer) in enumerate(
                zip(self.bidirectional_lstm_layers, self.dropout_layers, self.layer_norm_layers)
        ):
            # Pass through the Bidirectional LSTM layer
            # encoder_output shape: (batch_size, seq_len, units * 2)
            # forward_h shape: (batch_size, units)
            # backward_h shape: (batch_size, units)
            # forward_c shape: (batch_size, units)
            # backward_c shape: (batch_size, units)
            encoder_output, forward_h, forward_c, backward_h, backward_c = lstm_layer(
                encoder_output, training=training
            )

            # Apply Layer Normalization
            encoder_output = layer_norm_layer(encoder_output)

            # Concatenate the final forward and backward hidden states
            final_state_h = tf.concat([forward_h, backward_h], axis=-1) # Shape: (batch_size, units * 2)

            # Concatenate the final forward and backward cell states
            final_state_c = tf.concat([forward_c, backward_c], axis=-1) # Shape: (batch_size, units * 2)

            # Apply dropout to the encoder output
            encoder_output: tf.Tensor = dropout_layer(
                encoder_output,
                training=training
            ) # Shape: (batch_size, seq_len, units * 2)

        if final_state_h is None or final_state_c is None:
            raise ValueError("No Encoder LSTM layers detected; Encoder 'num_layers' must be at least 1.")

        return encoder_output, final_state_h, final_state_c

    def compute_mask(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None
    ) -> Optional[tf.Tensor]:
        """
        Propagates the mask forward by computing an output mask tensor for the layer.

        Parameters
        ----------
        inputs: tf.Tensor
            Input tensor.
        mask: tf.Tensor, optional
            Input encoder mask. Defaults to None.

        Returns
        ----------
        encoder_mask: tf.Tensor
            Mask tensor based on the encoder's mask.
        """
        return self.embedding.compute_mask(inputs, mask)

    def get_config(self) -> dict:
        """
        Returns the configuration of the layer for serialization.

        Returns
        ----------
        config: dict
            A Python dictionary containing the layer's configuration.
        """
        config = super(StackedBidirectionalLSTMEncoder, self).get_config()
        config.update({
            'vocab_size': self.embedding.input_dim,
            'encoder_embedding_dim': self.embedding.output_dim,
            'units': self.units,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'StackedBidirectionalLSTMEncoder':
        """
        Creates a layer from its config.

        Parameters
        ----------
        config: dict
            A Python dictionary containing the layer's configuration.

        Returns
        ----------
        decoder: StackedBidirectionalLSTMEncoder
            A new instance of StackedBidirectionalLSTMEncoder configured using the provided config.
        """
        return cls(**config)
