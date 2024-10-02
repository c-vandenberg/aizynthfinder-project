import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Layer
from decoders.decoder_interface import DecoderInterface
from attention.attention import BahdanauAttention
from typing import List, Optional, Tuple, Union, Any

@tf.keras.utils.register_keras_serializable()
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
    def __init__(self, vocab_size: int, decoder_embedding_dim: int, units: int, num_layers: int,
                 dropout_rate: float = 0.2, **kwargs) -> None:
        super(StackedLSTMDecoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, decoder_embedding_dim, mask_zero=True)
        self.units = units
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        self.supports_masking = True

        # Decoder: 4-layer LSTM without internal Dropout
        # Define LSTM and Dropout layers separately
        self.lstm_layers = []
        self.dropout_layers = []
        for i in range(num_layers):
            lstm_layer = LSTM(
                units=units,
                return_sequences=True,
                return_state=True,
                name=f'lstm_decoder_{i + 1}'
            )
            dropout_layer = Dropout(dropout_rate, name=f'decoder_dropout_{i + 1}')
            self.lstm_layers.append(lstm_layer)
            self.dropout_layers.append(dropout_layer)

        # Attention Mechanism
        self.attention: BahdanauAttention = BahdanauAttention(units=units)

        # Output layer
        self.dense: Dense = Dense(vocab_size, activation='softmax')

    def call(self, inputs: Tuple[tf.Tensor, List[tf.Tensor], tf.Tensor], training: Optional[bool] = None,
             mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Forward pass of the decoder.

        Args:
            inputs (Tuple[tf.Tensor, List[tf.Tensor], tf.Tensor]): Tuple containing decoder input, initial state,
            and encoder output.
            training (Optional[bool], optional): Training flag. Defaults to None.
            mask (Optional[tf.Tensor], optional): Encoder mask. Defaults to None.

        Returns:
            tf.Tensor: The predicted token probabilities for each timestep in the target sequence.
        """
        # Extract initial state and encoder output from inputs
        decoder_input, initial_state, encoder_output = inputs

        if decoder_input is None or initial_state is None or encoder_output is None:
            raise ValueError('decoder_input, initial_state and encoder_output must be passed to the Decoder.')

        # Embed the input and extract decoder mask
        decoder_output: tf.Tensor = self.embedding(decoder_input)
        decoder_mask: Optional[tf.Tensor] = self.embedding.compute_mask(decoder_input)

        # Process through decoder layers
        for i, (lstm_layer, dropout_layer) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            if i == 0:
                # Use initial_state (encoder final state) for the first LSTM layer
                decoder_output, state_h, state_c = lstm_layer(
                    decoder_output,
                    mask=decoder_mask,
                    initial_state=initial_state,
                    training=training
                )
            else:
                decoder_output, state_h, state_c = lstm_layer(
                    decoder_output,
                    mask=decoder_mask,
                    training=training
                )
            decoder_output = dropout_layer(decoder_output, training=training)

        # Extract only the encoder_mask if passed mask list of tuple
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
        concat_output: tf.Tensor = tf.concat([decoder_output, context_vector], axis=-1)  # (batch_size, seq_len_dec, units + units_enc)

        # Generate outputs
        decoder_output: tf.Tensor = self.dense(concat_output)  # (batch_size, seq_len_dec, vocab_size)

        return decoder_output

    def single_step(self, decoder_input: tf.Tensor, states: List[tf.Tensor], encoder_output: tf.Tensor):
        """
        Performs a single decoding step.

        Args:
            decoder_input: The input token at the current time step (shape: [batch_size, 1]).
            states: The initial state of the decoder LSTM layers (list of tensors).
            encoder_output: The output from the encoder (shape: [batch_size, seq_len_enc, units]).

        Returns:
            Tuple of (decoder_output, state_h, state_c)
        """
        # Embed the input
        decoder_output = self.embedding(decoder_input)

        # Prepare the initial states
        num_states = len(states)
        expected_states = self.num_layers * 2  # h and c for each layer

        if num_states == 2:
            # Only initial state for the first layer is provided
            state_h, state_c = states
            states = [(state_h, state_c)] + [(None, None)] * (self.num_layers - 1)
        elif num_states == expected_states:
            # States for all layers are provided
            states = [(states[i], states[i + 1]) for i in range(0, num_states, 2)]
        else:
            raise ValueError(f"Expected states length to be 2 or {expected_states}, got {num_states}")

        new_states = []
        for i, lstm_layer in enumerate(self.lstm_layers):
            state_h, state_c = states[i]
            if state_h is None or state_c is None:
                batch_size = tf.shape(decoder_output)[0]
                state_h = tf.zeros((batch_size, self.units))
                state_c = tf.zeros((batch_size, self.units))
            decoder_output, state_h, state_c = lstm_layer(
                decoder_output,
                initial_state=[state_h, state_c],
                training=False
            )
            new_states.extend([state_h, state_c])

        # Attention mechanism
        context_vector, attention_weights = self.attention(
            inputs=[encoder_output, decoder_output],
            mask=None
        )

        # Concatenate decoder outputs and context vector
        concat_output = tf.concat([decoder_output, context_vector], axis=-1)

        # Generate outputs
        decoder_output = self.dense(concat_output)

        return decoder_output, new_states

    def compute_mask(self, inputs: Any, mask: Optional[Any] = None) -> None:
        """
        Computes an output mask tensor for the layer.

        Args:
            inputs: A tensor or list of tensors.
            mask: A mask or list of masks corresponding to the inputs.

        Returns:
            An output mask tensor.
        """
        decoder_input, initial_state, encoder_output = inputs

        # Get the mask from the embedding layer
        decoder_mask = self.embedding.compute_mask(decoder_input)

        # The output mask is based on the decoder's mask
        return decoder_mask

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'decoder_embedding_dim': self.embedding.output_dim,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'num_layers': self.num_layers,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'StackedLSTMDecoder':
        """
        Creates a layer from its config.

        Args:
            config (dict): A Python dictionary containing the layer's configuration.

        Returns:
            StackedLSTMDecoder: A new instance of StackedLSTMDecoder configured using the provided config.
        """
        return cls(**config)
