import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, Layer
from encoders.encoder_interface import EncoderInterface
from typing import Tuple, Optional, Union

@tf.keras.utils.register_keras_serializable()
class StackedBidirectionalLSTMEncoder(EncoderInterface):
    def __init__(self, vocab_size: int, encoder_embedding_dim: int, units: int, num_layers: int,
                 dropout_rate: float = 0.2, **kwargs):
        super(StackedBidirectionalLSTMEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, encoder_embedding_dim, mask_zero=True)
        self.units= units
        self.num_layers = num_layers
        self.dropout_rate= dropout_rate

        self.supports_masking = True

        # Build first Bidirectional LSTM layer
        self.bidirectional_lstm_layers = []
        self.dropout_layers = []
        for i in range(num_layers):
            lstm_layer = Bidirectional(
                LSTM(units, return_sequences=True, return_state=True),
                name=f'bidirectional_lstm_encoder_{i + 1}'
            )
            dropout_layer = Dropout(dropout_rate, name=f'encoder_dropout_{i + 1}')
            self.bidirectional_lstm_layers.append(lstm_layer)
            self.dropout_layers.append(dropout_layer)


    def call(self, encoder_input: tf.Tensor, training: Optional[bool] = None):
        """
        Forward pass of the encoder.

        Args:
            encoder_input (tf.Tensor): Input tensor for the encoder.
            training (Optional[bool], optional): Training flag. Defaults to None.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Encoder output, final hidden state, and final cell state.
        """
        # Embed the input and obtain mask
        encoder_output: tf.Tensor = self.embedding(encoder_input)
        final_state_h: Union[None, tf.Tensor] = None
        final_state_c: Union[None, tf.Tensor] = None

        for lstm_layer, dropout_layer in zip(self.bidirectional_lstm_layers, self.dropout_layers):
            encoder_output, forward_h, forward_c, backward_h, backward_c = lstm_layer(
                encoder_output, training=training
            )
            final_state_h = tf.concat([forward_h, backward_h], axis=-1)
            final_state_c = tf.concat([forward_c, backward_c], axis=-1)
            encoder_output: tf.Tensor = dropout_layer(encoder_output, training=training)

        if final_state_h is None or final_state_c is None:
            raise ValueError("No Encoder LSTM layers detected; Encoder 'num_layers' must be at least 1.")

        return encoder_output, final_state_h, final_state_c

    def compute_mask(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None) -> Optional[tf.Tensor]:
        """
        Propagates the mask forward.

        Args:
            inputs (tf.Tensor): Input tensors.
            mask (Optional[tf.Tensor], optional): Input mask. Defaults to None.

        Returns:
            Optional[tf.Tensor]: Propagated mask.
        """
        return self.embedding.compute_mask(inputs, mask)

    def get_config(self) -> dict:
        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict: A Python dictionary containing the layer's configuration.
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

        Args:
            config (dict): A Python dictionary containing the layer's configuration.

        Returns:
            StackedBidirectionalLSTMEncoder: A new instance of StackedBidirectionalLSTMEncoder configured using the
            provided config.
        """
        return cls(**config)
