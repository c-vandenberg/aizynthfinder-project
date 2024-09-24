import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout
from models.interfaces import EncoderInterface


class StackedBidirectionalLSTMEncoder(tf.keras.layers.Layer, EncoderInterface):
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
    def __init__(self, vocab_size: int, embedding_dim: int, units: int, dropout_rate=0.2):
        super(StackedBidirectionalLSTMEncoder, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)

        self.bidirectional_lstm_1 = Bidirectional(
            LSTM(units, return_sequences=True, return_state=True),
            name='bidirectional_lstm_1'
        )
        self.dropout_1 = Dropout(dropout_rate, name='encoder_dropout_1')
        self.bidirectional_lstm_2 = Bidirectional(
            LSTM(units, return_sequences=True, return_state=True),
            name='bidirectional_lstm_2'
        )
        self.dropout_2 = Dropout(dropout_rate, name='encoder_dropout_2')

    def call(self, encoder_input, training=None):
        # Embed the input and obtain mask
        encoder_output = self.embedding(encoder_input)
        mask = self.embedding.compute_mask(encoder_input)

        # Process through encoder layers
        # First LSTM layer
        encoder_output, forward_h, forward_c, backward_h, backward_c = self.bidirectional_lstm_1(
            encoder_output, mask=mask, training=training
        )
        # Concatenate forward and backward states
        state_h_1 = tf.concat([forward_h, backward_h], axis=-1)
        state_c_1 = tf.concat([forward_c, backward_c], axis=-1)

        # Apply dropout
        encoder_output = self.dropout_1(encoder_output, training=training)

        # Second LSTM layer
        encoder_output, forward_h, forward_c, backward_h, backward_c = self.bidirectional_lstm_2(
            encoder_output, mask=mask, training=training
        )
        # Concatenate forward and backward states
        state_h_2 = tf.concat([forward_h, backward_h], axis=-1)
        state_c_2 = tf.concat([forward_c, backward_c], axis=-1)
        # Apply dropout
        encoder_output = self.dropout_2(encoder_output, training=training)

        # Final states
        final_state_h = state_h_2
        final_state_c = state_c_2

        return encoder_output, final_state_h, final_state_c

    def compute_mask(self, inputs, mask=None):
        # Propagate the mask forward
        return self.embedding.compute_mask(inputs, mask)


class SingleBidirectionalLSTMEncoder(tf.keras.layers.Layer, EncoderInterface):
    """
    Encoder: SingleBidirectionalLSTMEncoder

    Description:
    The SingleBidirectionalLSTMEncoder is a custom TensorFlow Keras layer that encodes the preprocessed, tokenized
    SMILES input sequence into context-rich representations. It consists of:

    1. **Embedding Layer**: Transforms input SMILES tokens into dense vectors of fixed size.
    2. **Single Bidirectional LSTM Layer**:
      - **LSTM Layer**: Bidirectional LSTM with a specified number of units, processing the embedded input.
      - **Dropout Layer**: Applies dropout with a rate of 0.2 to the outputs of the first LSTM layer.

    Functionality:
    1. **Embedding**:
       Converts input SMILES token indices into dense vectors, enabling the model to learn meaningful molecular
       representations.
    2. **Bidirectional LSTM Layer**:
       Processes the embeddings in both forward and backward directions, capturing context from both past and future
       SMILES tokens.
    3. **Dropout Layer**:
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
        returning the LSTM layer final sequence output and the concatenated hidden and cell states.

    Returns
    -------
    encoder_output : Tensor
        The final sequence representations output by the encoder.
    final_state_h : Tensor
        The concatenated hidden state from the Bidirectional LSTM layer.
    final_state_c : Tensor
        The concatenated cell state from the Bidirectional LSTM layer.

    Notes
    -----
    The model uses a single Bidirectional LSTM layers, each followed by a dropout layer to avoid overfitting.
    The hidden and cell states from both forward and backward passes of the LSTM are concatenated and returned as
    final states.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, units: int, dropout_rate=0.2):
        super(SingleBidirectionalLSTMEncoder, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)

        self.bidirectional_lstm = Bidirectional(
            LSTM(units, return_sequences=True, return_state=True),
            name='bidirectional_lstm'
        )
        self.dropout = Dropout(dropout_rate, name='encoder_dropout')

    def call(self, encoder_input, training=None):
        # Embed the input and obtain mask
        encoder_output = self.embedding(encoder_input)
        mask = self.embedding.compute_mask(encoder_input)

        # Process through encoder layers
        # First LSTM layer
        encoder_output, forward_h, forward_c, backward_h, backward_c = self.bidirectional_lstm_1(
            encoder_output, mask=mask, training=training
        )

        # Concatenate forward and backward states
        final_state_h = tf.concat([forward_h, backward_h], axis=-1)
        final_state_c = tf.concat([forward_c, backward_c], axis=-1)

        # Apply dropout
        encoder_output = self.dropout_2(encoder_output, training=training)

        return encoder_output, final_state_h, final_state_c

    def compute_mask(self, inputs, mask=None):
        # Propagate the mask forward
        return self.embedding.compute_mask(inputs, mask)