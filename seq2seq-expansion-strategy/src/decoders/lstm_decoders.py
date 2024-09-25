import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from decoders.decoder_interface import DecoderInterface
from attention.attention import BahdanauAttention


class StackedLSTMDecoder(DecoderInterface):
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
    def __init__(self, vocab_size: int, decoder_embedding_dim: int, units: int, dropout_rate=0.2):
        super(StackedLSTMDecoder, self).__init__(vocab_size, decoder_embedding_dim, units)
        self.units = units
        self.embedding = Embedding(vocab_size, decoder_embedding_dim, mask_zero=True)

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

    def call(self, inputs, training=None, mask=None):
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

    def single_step(self, decoder_input, states, encoder_output):
        """
        Performs a single decoding step.

        Args:
            decoder_input: The input token at the current time step (shape: [batch_size, 1]).
            states: The initial state of the decoder LSTM layers (list of tensors).
            encoder_output: The output from the encoder (shape: [batch_size, seq_len_enc, units]).

        Returns:
            Tuple of (decoder_output, state_h, state_c)
        """
        # Unpack states
        if len(states) == 2:
            # Initial state provided only for the first LSTM layer
            state_h1, state_c1 = states
            state_h2 = tf.zeros_like(state_h1)
            state_c2 = tf.zeros_like(state_c1)
            state_h3 = tf.zeros_like(state_h1)
            state_c3 = tf.zeros_like(state_c1)
            state_h4 = tf.zeros_like(state_h1)
            state_c4 = tf.zeros_like(state_c1)
        else:
            # States for all layers provided
            state_h1, state_c1, state_h2, state_c2, state_h3, state_c3, state_h4, state_c4 = states

        # Embed the input
        decoder_output = self.embedding(decoder_input)

        # First LSTM layer with initial state
        decoder_output, state_h1, state_c1 = self.lstm_decoder_1(
            decoder_output,
            initial_state=[state_h1, state_c1],
            training=False
        )
        # No dropout during inference
        # Subsequent LSTM layers
        decoder_output, state_h2, state_c2 = self.lstm_decoder_2(
            decoder_output,
            initial_state=[state_h2, state_c2],
            training=False
        )
        decoder_output, state_h3, state_c3 = self.lstm_decoder_3(
            decoder_output,
            initial_state=[state_h3, state_c3],
            training=False
        )
        decoder_output, state_h4, state_c4 = self.lstm_decoder_4(
            decoder_output,
            initial_state=[state_h4, state_c4],
            training=False
        )

        # Attention mechanism
        context_vector, attention_weights = self.attention(
            inputs=[encoder_output, decoder_output],
            mask=None  # No mask during inference
        )

        # Concatenate decoder outputs and context vector
        concat_output = tf.concat([decoder_output, context_vector], axis=-1)

        # Generate outputs
        decoder_output = self.dense(concat_output)  # Shape: (batch_size, 1, vocab_size)

        # Collect all states
        decoder_states = [state_h1, state_c1, state_h2, state_c2, state_h3, state_c3, state_h4, state_c4]

        return decoder_output, decoder_states
