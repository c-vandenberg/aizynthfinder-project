import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from models.interfaces import DecoderInterface
from models.attention import BahdanauAttention


class StackedLSTMDecoder(tf.keras.layers.Layer, DecoderInterface):
    """
    Decoder: StackedLSTMDecoder

    Description:
    The StackedLSTMDecoder is a custom TensorFlow Keras layer responsible for generating the target SMILES sequence based
    on the encoder's context and the previously generated tokens. It consists of:

    - **Embedding Layer**: Transforms target tokens into dense vectors.
    - **Stacked LSTM Layers**:
      - **Layers 1-4**: Four consecutive LSTM layers with a specified number of units, each processing the embedded inputs.
      - **Dropout Layers**: Applies dropout with a rate of 0.2 after each LSTM layer to enhance generalization.
    - **Attention Mechanism**: BahdanauAttention to focus on relevant encoder outputs during decoding.
    - **Output Dense Layer**: Generates probability distributions over the target vocabulary using softmax activation.

    Functionality:
    1. **Embedding**:
       Converts target token indices into dense vectors, facilitating the learning of target representations.
    2. **LSTM Layers**:
       Processes the embedded inputs through multiple LSTM layers, capturing complex sequential patterns.
    3. **Dropout Layers**:
       Introduces regularization to prevent overfitting by randomly deactivating a subset of neurons during training.
    4. **Attention Mechanism**:
       Computes attention weights to dynamically focus on different parts of the encoder's output, improving the relevance
       and coherence of the generated sequence.
    5. **Output Layer**:
       Transforms the combined decoder and context vectors into probability distributions over the target vocabulary,
       enabling token prediction.

    The decoder leverages the stacked LSTM architecture and attention mechanism to generate accurate and contextually
    relevant target sequences.

    Parameters
    ----------
    vocab_size : int
        The size of the target vocabulary (number of unique tokens).
    embedding_dim : int
        Dimensionality of the token embedding vectors.
    units : int
        The number of units (neurons) in each LSTM layer.
    dropout_rate : float, optional
        The dropout rate applied to the LSTM layers (default is 0.2).

    Methods
    -------
    call(inputs, training=False, **kwargs)
        Forward pass of the decoder. Processes the input sequence through the embedding and LSTM layers, applies attention,
        and generates the final output probability distributions over the target vocabulary.

    Returns
    -------
    decoder_output : Tensor
        The predicted token probabilities for each timestep in the target sequence.

    Notes
    -----
    The decoder consists of four LSTM layers, each followed by a dropout layer to enhance generalization.
    The attention mechanism helps the decoder focus on relevant encoder outputs during each timestep of decoding.
    """
    def __init__(self, vocab_size, embedding_dim, units, dropout_rate=0.2):
        super(StackedLSTMDecoder, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)

        # Decoder: 4-layer LSTM without internal Dropout
        # Define LSTM and Dropout layers individually
        self.lstm_decoder_1 = LSTM(
            units,
            return_sequences=True,
            return_state=True,
            name='lstm_decoder_1'
        )
        self.dropout_1 = Dropout(dropout_rate, name='decoder_dropout_1')

        self.lstm_decoder_2 = LSTM(
            units,
            return_sequences=True,
            return_state=True,
            name='lstm_decoder_2'
        )
        self.dropout_2 = Dropout(dropout_rate, name='decoder_dropout_2')

        self.lstm_decoder_3 = LSTM(
            units,
            return_sequences=True,
            return_state=True,
            name='lstm_decoder_3'
        )
        self.dropout_3 = Dropout(dropout_rate, name='decoder_dropout_3')

        self.lstm_decoder_4 = LSTM(
            units,
            return_sequences=True,
            return_state=True,
            name='lstm_decoder_4'
        )
        self.dropout_4 = Dropout(dropout_rate, name='decoder_dropout_4')

        # Attention Mechanism
        self.attention = BahdanauAttention(units=units)

        # Output layer
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, inputs, training=None, mask=None, **kwargs):
        # Extract initial state and encoder output from inputs
        decoder_input, initial_state, encoder_output = inputs

        if decoder_input is None or initial_state is None or encoder_output is None:
            raise ValueError('decoder_input, initial_state and encoder_output must be provided to the Decoder.')

        # Embed the input and extract decoder mask
        decoder_output = self.embedding(decoder_input)
        decoder_mask = self.embedding.compute_mask(decoder_input)

        # Process through decoder layers
        # First LSTM layer with initial state
        decoder_output, _, _ = self.lstm_decoder_1(
            decoder_output,
            mask=decoder_mask,
            initial_state=initial_state,
            training=training
        )
        decoder_output = self.dropout_1(decoder_output, training=training)

        # Second LSTM layer
        decoder_output, _, _ = self.lstm_decoder_2(
            decoder_output,
            mask=decoder_mask,
            training=training
        )
        decoder_output = self.dropout_2(decoder_output, training=training)

        # Third LSTM layer
        decoder_output, _, _ = self.lstm_decoder_3(
            decoder_output,
            mask=decoder_mask,
            training=training
        )
        decoder_output = self.dropout_3(decoder_output, training=training)

        # Fourth LSTM layer
        decoder_output, final_state_h, final_state_c = self.lstm_decoder_4(
            decoder_output,
            mask=decoder_mask,
            training=training
        )
        decoder_output = self.dropout_4(decoder_output, training=training)

        # Extract only the encoder_mask from the mask list
        if mask is not None and isinstance(mask, (list, tuple)):
            encoder_mask = mask[1]
        else:
            encoder_mask = mask

        # Apply attention
        context_vector, attention_weights = self.attention(inputs=[encoder_output, decoder_output], mask=encoder_mask)

        # Concatenate decoder outputs and context vector
        concat_output = tf.concat([decoder_output, context_vector], axis=-1)  # (batch_size, seq_len_dec, units + units_enc)

        # Generate outputs
        decoder_output = self.dense(concat_output)  # (batch_size, seq_len_dec, vocab_size)

        return decoder_output


class SingleLSTMDecoder(tf.keras.layers.Layer, DecoderInterface):
    """
    Decoder: SingleLSTMDecoder

    Description:
    The SingleLSTMDecoder is a custom TensorFlow Keras layer responsible for generating the target SMILES sequence based
    on the encoder's context and the previously generated tokens. It consists of:

    - **Embedding Layer**: Transforms target tokens into dense vectors.
    - **LSTM and Droput Layers**:
      - **LSTM Layer**: A single LSTM layer with a specified number of units, to process the embedded inputs.
      - **Dropout Layer**: Applies dropout with a rate of 0.2 after the LSTM layer to enhance generalization.
    - **Attention Mechanism**: BahdanauAttention to focus on relevant encoder outputs during decoding.
    - **Output Dense Layer**: Generates probability distributions over the target vocabulary using softmax activation.

    Functionality:
    1. **Embedding**:
       Converts target token indices into dense vectors, facilitating the learning of target representations.
    2. **LSTM Layers**:
       Processes the embedded inputs through a single LSTM layers, capturing complex sequential patterns.
    3. **Dropout Layers**:
       Introduces regularization to prevent overfitting by randomly deactivating a subset of neurons during training.
    4. **Attention Mechanism**:
       Computes attention weights to dynamically focus on different parts of the encoder's output, improving the relevance
       and coherence of the generated sequence.
    5. **Output Layer**:
       Transforms the combined decoder and context vectors into probability distributions over the target vocabulary,
       enabling token prediction.

    The decoder leverages LSTM architecture and attention mechanism to generate accurate and contextually relevant
    target sequences.

    Parameters
    ----------
    vocab_size : int
        The size of the target vocabulary (number of unique tokens).
    embedding_dim : int
        Dimensionality of the token embedding vectors.
    units : int
        The number of units (neurons) in the LSTM layer.
    dropout_rate : float, optional
        The dropout rate applied to the LSTM layer (default is 0.2).

    Methods
    -------
    call(inputs, training=False, **kwargs)
        Forward pass of the decoder. Processes the input sequence through the embedding and LSTM layer, applies attention,
        and generates the final output probability distributions over the target vocabulary.

    Returns
    -------
    decoder_output : Tensor
        The predicted token probabilities for each timestep in the target sequence.

    Notes
    -----
    The decoder consists of a single LSTM layer, followed by a dropout layer to enhance generalization.
    The attention mechanism helps the decoder focus on relevant encoder outputs during each timestep of decoding.
    """
    def __init__(self, vocab_size, embedding_dim, units, dropout_rate=0.2):
        super(SingleLSTMDecoder, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)

        # Decoder: 4-layer LSTM without internal Dropout
        self.decoder_layers = tf.keras.Sequential([
            LSTM(units, return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0,
                 name='lstm_decoder_1'),
            Dropout(dropout_rate, name='decoder_dropout_1'),
        ])

        # Attention Mechanism
        self.attention = BahdanauAttention(units=units)

        # Output layer
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, inputs, training=None, mask=None, **kwargs):
        # Extract initial state and encoder output from inputs
        decoder_input, initial_state, encoder_output = inputs

        if decoder_input is None or initial_state is None or encoder_output is None:
            raise ValueError('decoder_input, initial_state and encoder_output must be provided to the Decoder.')

        # Embed the input and extract decoder mask
        decoder_output = self.embedding(decoder_input)
        decoder_mask = self.embedding.compute_mask(decoder_input)

        # Process through decoder layers
        # LSTM layer
        decoder_output, state_h, state_c = self.lstm_decoder(
            decoder_output,
            mask=decoder_mask,
            initial_state=initial_state,
            training=training
        )

        # Apply Dropout
        decoder_output = self.dropout(decoder_output, training=training)

        # Extract only the encoder_mask from the mask list
        if mask is not None and isinstance(mask, (list, tuple)):
            encoder_mask = mask[1]
        else:
            encoder_mask = mask

        # Apply attention
        context_vector, attention_weights = self.attention(
            inputs=[encoder_output, decoder_output],
            mask=encoder_mask
        )

        # Concatenate decoder outputs and context vector
        concat_output = tf.concat([decoder_output, context_vector], axis=-1)  # (batch_size, seq_len_dec, units + units_enc)

        # Generate outputs
        decoder_output = self.dense(concat_output)  # (batch_size, seq_len_dec, vocab_size)

        return decoder_output