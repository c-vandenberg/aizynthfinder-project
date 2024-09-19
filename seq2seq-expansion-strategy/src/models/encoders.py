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

        # Encoder: 2-layer Bidirectional LSTM without internal Dropout
        self.encoder_layers = tf.keras.Sequential([
            Bidirectional(LSTM(units, return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0),
                          name='bidirectional_lstm_1'),
            Dropout(dropout_rate, name='encoder_dropout_1'),
            Bidirectional(LSTM(units, return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0),
                          name='bidirectional_lstm_2'),
            Dropout(dropout_rate, name='encoder_dropout_2')
        ])

    def call(self, encoder_input, training=False):
        # Embed the input
        encoder_output = self.embedding(encoder_input)
        states_h = []
        states_c = []

        # Process through encoder layers
        # Since encoder_layers is a Sequential model, it will process both Bidirectional LSTMs sequentially
        # However, to extract states, we need to manually iterate through them
        # Therefore, it's better to separate the layers or adjust accordingly
        # Alternatively, you can define the encoder_layers as a list of layers inside a custom Layer
        # For clarity, we'll process layers one by one
        for layer in self.encoder_layers.layers:
            if isinstance(layer, Bidirectional):
                encoder_output, forward_h, forward_c, backward_h, backward_c = layer(encoder_output, training=training)
                # Concatenate forward and backward states
                state_h = tf.concat([forward_h, backward_h], axis=-1)  # Shape: (batch_size, units * 2)
                state_c = tf.concat([forward_c, backward_c], axis=-1)
            elif isinstance(layer, Dropout):
                # Apply Dropout
                encoder_output = layer(encoder_output, training=training)
                continue
            else:
                encoder_output, state_h, state_c = layer(encoder_output, training=training)
            states_h.append(state_h)
            states_c.append(state_c)

        # Use the last layer's states for final states
        final_state_h = states_h[-1]
        final_state_c = states_c[-1]

        return encoder_output, final_state_h, final_state_c
