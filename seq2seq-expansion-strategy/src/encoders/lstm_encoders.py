import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout
from encoders.encoder_interface import EncoderInterface
from typing import Tuple, Optional

class StackedBidirectionalLSTMEncoder(EncoderInterface):
    """
    Encoder: StackedBidirectionalLSTMEncoder

    Description:
    The StackedBidirectionalLSTMEncoder is a custom TensorFlow Keras layer that encodes the preprocessed, tokenized
    SMILES input sequence into context-rich representations. It consists of:

    1. **Embedding Layer**: Transforms input SMILES tokens into dense vectors of fixed size.
    2. **Stacked Bidirectional LSTM Layers**:
      - **Layer 1**: Bidirectional LSTM with a specified number of units, processing the embedded input.
      - **Dropout 1**: Applies dropout with a rate of 0.2 to the outputs of the first LSTM layer.
      - **Layer 2**: Bidirectional LSTM with the same number of units, further processing the outputs.
      - **Dropout 2**: Applies dropout with a rate of 0.2 to the outputs of the second LSTM layer.

    Functionality:
    1. **Embedding**:
       Converts input SMILES token indices into dense vectors, enabling the model to learn meaningful molecular
       representations.
    2. **Bidirectional LSTM Layers**:
       Processes the embeddings in both forward and backward directions, capturing context from both past and future
       SMILES tokens.
    3. **Dropout Layers**:
       Introduces regularization by randomly setting a fraction of input units to 0 during training, mitigating
       overfitting.

    The encoder outputs the final sequence representations along with the concatenated hidden and cell states
    from the last Bidirectional LSTM layer, which serve as the initial states for the decoder.

    Parameters
    ----------
    vocab_size : int
        The size of the vocabulary (number of unique tokens).
    embedding_dim : int
        Dimensionality of the token embedding vectors.
    units : int
        The number of units (neurons) in each LSTM layer.
    dropout_rate : float, optional
        The dropout rate applied to the LSTM layers (default is 0.2).

    Methods
    -------
    call(encoder_input, training=False)
        Forward pass of the encoder. Processes the input sequence through the embedding and LSTM layers,
        returning the final sequence output and the concatenated hidden and cell states from the last Bidirectional
        LSTM layer.

    Returns
    -------
    encoder_output : Tensor
        The final sequence representations output by the encoder.
    final_state_h : Tensor
        The concatenated hidden state from the last Bidirectional LSTM layer.
    final_state_c : Tensor
        The concatenated cell state from the last Bidirectional LSTM layer.

    Notes
    -----
    The model uses two Bidirectional LSTM layers, each followed by a dropout layer to avoid overfitting.
    The hidden and cell states from both forward and backward passes of the LSTM are concatenated and returned as
    final states.
    """
    def __init__(self, vocab_size: int, encoder_embedding_dim: int, units: int, dropout_rate: float = 0.2, **kwargs):
        super(StackedBidirectionalLSTMEncoder, self).__init__(vocab_size, encoder_embedding_dim, units, **kwargs)
        self.units: int = units
        self.embedding: Embedding = Embedding(vocab_size, encoder_embedding_dim, mask_zero=True)
        self.dropout_rate: float = dropout_rate

        self.bidirectional_lstm_1: Bidirectional = Bidirectional(
            LSTM(units, return_sequences=True, return_state=True),
            name='bidirectional_lstm_1'
        )

        self.dropout_1: Dropout = Dropout(dropout_rate, name='encoder_dropout_1')

        self.bidirectional_lstm_2: Bidirectional = Bidirectional(
            LSTM(units, return_sequences=True, return_state=True),
            name='bidirectional_lstm_2'
        )

        self.dropout_2: Dropout = Dropout(dropout_rate, name='encoder_dropout_2')

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
        mask = self.embedding.compute_mask(encoder_input)

        # Process through encoder layers
        # First LSTM layer
        encoder_output, forward_h, forward_c, backward_h, backward_c = self.bidirectional_lstm_1(
            encoder_output, mask=mask, training=training
        )
        # Concatenate forward and backward states
        state_h_1: tf.Tensor = tf.concat([forward_h, backward_h], axis=-1)
        state_c_1: tf.Tensor = tf.concat([forward_c, backward_c], axis=-1)

        # Apply dropout
        encoder_output: Optional[tf.Tensor] = self.dropout_1(encoder_output, training=training)

        # Second LSTM layer
        encoder_output, forward_h, forward_c, backward_h, backward_c = self.bidirectional_lstm_2(
            encoder_output, mask=mask, training=training
        )

        # Concatenate forward and backward states
        state_h_2: tf.Tensor = tf.concat([forward_h, backward_h], axis=-1)
        state_c_2: tf.Tensor = tf.concat([forward_c, backward_c], axis=-1)

        # Apply dropout
        encoder_output: tf.Tensor = self.dropout_2(encoder_output, training=training)

        # Final states
        final_state_h: tf.Tensor = state_h_2
        final_state_c: tf.Tensor = state_c_2

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
            'dropout_rate': self.dropout_rate,
            'embedding': tf.keras.layers.serialize(self.embedding),
            'bidirectional_lstm_1': tf.keras.layers.serialize(self.bidirectional_lstm_1),
            'dropout_1': tf.keras.layers.serialize(self.dropout_1),
            'bidirectional_lstm_2': tf.keras.layers.serialize(self.bidirectional_lstm_2),
            'dropout_2': tf.keras.layers.serialize(self.dropout_2),
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
        # Deserialize layers
        config['embedding'] = tf.keras.layers.deserialize(config['embedding'])
        config['bidirectional_lstm_1'] = tf.keras.layers.deserialize(config['bidirectional_lstm_1'])
        config['dropout_1'] = tf.keras.layers.deserialize(config['dropout_1'])
        config['bidirectional_lstm_2'] = tf.keras.layers.deserialize(config['bidirectional_lstm_2'])
        config['dropout_2'] = tf.keras.layers.deserialize(config['dropout_2'])
        return cls(**config)
