import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, Layer
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

    def build(self, input_shape):
        """
        Build the StackedBidirectionalLSTMEncoder layer by initializing its sublayers.

        Args:
            input_shape (tf.TensorShape): Shape of the encoder input.
        """
        # Build the embedding layer
        self.embedding.build(input_shape)
        embedded_shape = self.embedding.compute_output_shape(input_shape)

        # Build first Bidirectional LSTM layer
        # For LSTM layers with `return_type=True`, method `compute_output_shape` returns a tuple containing:
        # 1. Output Shape: Shape of the output tensor.
        # 2. State Shapes: Shapes of the hidden state (state_h) and cell state (state_c).
        # We therefore must only extract the first element (i.e. the output shape)
        self.bidirectional_lstm_1.build(embedded_shape)
        lstm1_output_shape = self.bidirectional_lstm_1.compute_output_shape(embedded_shape)[0]  # Extract only output shape
        # Ensure lstm1_output_shape is a TensorShape
        if isinstance(lstm1_output_shape, tuple):
            lstm1_output_shape = tf.TensorShape(lstm1_output_shape)
        self.dropout_1.build(lstm1_output_shape)

        # Build second Bidirectional LSTM layer
        self.bidirectional_lstm_2.build(lstm1_output_shape)
        lstm2_output_shape = self.bidirectional_lstm_2.compute_output_shape(lstm1_output_shape)[0]  # Extract only output shape
        if isinstance(lstm2_output_shape, tuple):
            lstm2_output_shape = tf.TensorShape(lstm2_output_shape)
        self.dropout_2.build(lstm2_output_shape)

        # Mark the layer as built
        super(StackedBidirectionalLSTMEncoder, self).build(input_shape)

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


class SimpleEncoder(Layer):
    def __init__(self, vocab_size: int, embedding_dim: int, units: int, dropout_rate: float = 0.2, **kwargs):
        super(SimpleEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.dropout_rate = dropout_rate

        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name='simple_embedding')
        self.dense = Dense(units, activation='relu', name='simple_dense')
        self.dropout = Dropout(dropout_rate, name='simple_dropout')

    def build(self, input_shape):
        """
        Build method to initialize sublayers with the correct input shapes.
        """
        # Build the embedding layer
        self.embedding.build(input_shape)

        # Compute the output shape after embedding
        embedding_output_shape = self.embedding.compute_output_shape(input_shape)
        self.dense.build(embedding_output_shape)

        # Compute the output shape after dense
        dense_output_shape = self.dense.compute_output_shape(embedding_output_shape)
        self.dropout.build(dense_output_shape)

        # Finally, call the superclass build method
        super(SimpleEncoder, self).build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Embed input
        x = self.embedding(inputs)  # Shape: (batch_size, sequence_length, embedding_dim)

        # Apply Dense layer
        encoder_output = self.dense(x)  # Shape: (batch_size, sequence_length, units)

        # Apply Dropout
        encoder_output = self.dropout(encoder_output, training=training)

        # For compatibility, return dummy states
        state_h = tf.zeros_like(encoder_output[:, 0, :])  # Shape: (batch_size, units)
        state_c = tf.zeros_like(encoder_output[:, 0, :])  # Shape: (batch_size, units)

        return encoder_output, state_h, state_c

    def compute_mask(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None) -> Optional[tf.Tensor]:
        return self.embedding.compute_mask(inputs, mask)

    def get_config(self) -> dict:
        config = super(SimpleEncoder, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'embedding': tf.keras.layers.serialize(self.embedding),
            'dense': tf.keras.layers.serialize(self.dense),
            'dropout': tf.keras.layers.serialize(self.dropout),
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'SimpleEncoder':
        # Deserialize layers
        config['embedding'] = tf.keras.layers.deserialize(config['embedding'])
        config['dense'] = tf.keras.layers.deserialize(config['dense'])
        config['dropout'] = tf.keras.layers.deserialize(config['dropout'])
        return cls(**config)


