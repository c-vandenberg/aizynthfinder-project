import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Layer
from decoders.decoder_interface import DecoderInterface
from attention.attention import BahdanauAttention
from typing import List, Optional, Tuple, Union, Any


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
    def __init__(self, vocab_size: int, decoder_embedding_dim: int, units: int, dropout_rate: float = 0.2,
                 **kwargs) -> None:
        super(StackedLSTMDecoder, self).__init__(vocab_size, decoder_embedding_dim, units, **kwargs)
        self.units: int = units
        self.embedding: Embedding = Embedding(vocab_size, decoder_embedding_dim, mask_zero=True)
        self.vocab_size: int = vocab_size
        self.dropout_rate: float = dropout_rate

        # Decoder: 4-layer LSTM without internal Dropout
        # Define LSTM and Dropout layers individually
        self.lstm_decoder_1: LSTM = LSTM(
            units,
            return_sequences=True,
            return_state=True,
            name='lstm_decoder_1'
        )
        self.dropout_1: Dropout = Dropout(dropout_rate, name='decoder_dropout_1')

        self.lstm_decoder_2: LSTM = LSTM(
            units,
            return_sequences=True,
            return_state=True,
            name='lstm_decoder_2'
        )
        self.dropout_2: Dropout = Dropout(dropout_rate, name='decoder_dropout_2')

        self.lstm_decoder_3: LSTM = LSTM(
            units,
            return_sequences=True,
            return_state=True,
            name='lstm_decoder_3'
        )
        self.dropout_3: Dropout = Dropout(dropout_rate, name='decoder_dropout_3')

        self.lstm_decoder_4: LSTM = LSTM(
            units,
            return_sequences=True,
            return_state=True,
            name='lstm_decoder_4'
        )
        self.dropout_4: Dropout = Dropout(dropout_rate, name='decoder_dropout_4')

        # Attention Mechanism
        self.attention: BahdanauAttention = BahdanauAttention(units=units)

        # Output layer
        self.dense: Dense = Dense(vocab_size, activation='softmax')

    def build(self, input_shape):
        """
        Build the StackedLSTMDecoder layer by initializing its sublayers.

        Args:
            input_shape (Tuple[tf.TensorShape, List[tf.TensorShape], tf.TensorShape]):
                A tuple containing:
                    - decoder_input_shape
                    - initial_state_shape (list of TensorShapes)
                    - encoder_output_shape
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 3:
            raise ValueError("Input shape must be a tuple or list of three TensorShape objects.")

        decoder_input_shape, initial_state_shape, encoder_output_shape = input_shape

        # Ensure decoder_input_shape and encoder_output_shape are of type TensorShape
        if isinstance(decoder_input_shape, tuple):
            decoder_input_shape = tf.TensorShape(decoder_input_shape)
        if isinstance(encoder_output_shape, tuple):
            encoder_output_shape = tf.TensorShape(encoder_output_shape)
        # Ensure initial_state_shape is of type TensorShape
        if isinstance(initial_state_shape, list) or isinstance(initial_state_shape, tuple):
            initial_state_shape = [
                tf.TensorShape(s) if isinstance(s, tuple) else s for s in initial_state_shape
            ]
        else:
            raise ValueError(f"initial_state_shape must be a list of TensorShape objects. Type {type(initial_state_shape)} found")

        # Build the embedding layer
        self.embedding.build(decoder_input_shape)
        embedded_shape = self.embedding.compute_output_shape(decoder_input_shape)

        # Build first LSTM layer with initial state
        self.lstm_decoder_1.build(embedded_shape)
        lstm1_output_shape = self.lstm_decoder_1.compute_output_shape(embedded_shape)[0]  # Extract only output shape
        lstm1_output_shape = tf.TensorShape(lstm1_output_shape) if isinstance(lstm1_output_shape,
                                                                              tuple) else lstm1_output_shape
        self.dropout_1.build(lstm1_output_shape)

        # Build second LSTM layer
        self.lstm_decoder_2.build(lstm1_output_shape)
        lstm2_output_shape = self.lstm_decoder_2.compute_output_shape(lstm1_output_shape)[0]  # Extract only output shape
        lstm2_output_shape = tf.TensorShape(lstm2_output_shape) if isinstance(lstm2_output_shape,
                                                                              tuple) else lstm2_output_shape
        self.dropout_2.build(lstm2_output_shape)

        # Build third LSTM layer
        self.lstm_decoder_3.build(lstm2_output_shape)
        lstm3_output_shape = self.lstm_decoder_3.compute_output_shape(lstm2_output_shape)[0]  # Extract only output shape
        lstm3_output_shape = tf.TensorShape(lstm3_output_shape) if isinstance(lstm3_output_shape,
                                                                              tuple) else lstm3_output_shape
        self.dropout_3.build(lstm3_output_shape)

        # Build fourth LSTM layer
        self.lstm_decoder_4.build(lstm3_output_shape)
        lstm4_output_shape = self.lstm_decoder_4.compute_output_shape(lstm3_output_shape)[0]  # Extract only output shape
        lstm4_output_shape = tf.TensorShape(lstm4_output_shape) if isinstance(lstm4_output_shape,
                                                                              tuple) else lstm4_output_shape
        self.dropout_4.build(lstm4_output_shape)

        # Build the attention layer
        # BahdanauAttention expects encoder_output and decoder_output as inputs
        self.attention.build([encoder_output_shape, lstm4_output_shape])

        # Determine the Correct Input Shape for the Dense Layer
        # - The Dense layer receives the concatenated output of decoder and context vectors.
        # - decoder_output_shape: (batch_size, seq_len_dec, units_decoder)
        # - context_vector_shape: (batch_size, seq_len_dec, units_encoder)
        # - Thus, concat_output_shape: (batch_size, seq_len_dec, units_decoder + units_encoder)
        units_decoder = lstm4_output_shape[-1]  # 256
        units_encoder = encoder_output_shape[-1]  # 512
        concat_last_dim = units_decoder + units_encoder  # 768

        # Build the Dense Layer with the Correct Input Shape
        self.dense.build(tf.TensorShape((concat_last_dim,)))  # (768,)

        # Mark the layer as built
        super(StackedLSTMDecoder, self).build(input_shape)

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
            raise ValueError('decoder_input, initial_state and encoder_output must be provided to the Decoder.')

        # Embed the input and extract decoder mask
        decoder_output: tf.Tensor = self.embedding(decoder_input)
        decoder_mask: Optional[tf.Tensor] = self.embedding.compute_mask(decoder_input)

        # Process through decoder layers
        # First LSTM layer with initial state
        decoder_output, _, _ = self.lstm_decoder_1(
            decoder_output,
            mask=decoder_mask,
            initial_state=initial_state,
            training=training
        )
        decoder_output: tf.Tensor = self.dropout_1(decoder_output, training=training)

        # Second LSTM layer
        decoder_output, _, _ = self.lstm_decoder_2(
            decoder_output,
            mask=decoder_mask,
            training=training
        )
        decoder_output: tf.Tensor = self.dropout_2(decoder_output, training=training)

        # Third LSTM layer
        decoder_output, _, _ = self.lstm_decoder_3(
            decoder_output,
            mask=decoder_mask,
            training=training
        )
        decoder_output: tf.Tensor = self.dropout_3(decoder_output, training=training)

        # Fourth LSTM layer
        decoder_output, final_state_h, final_state_c = self.lstm_decoder_4(
            decoder_output,
            mask=decoder_mask,
            training=training
        )
        decoder_output: tf.Tensor = self.dropout_4(decoder_output, training=training)

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
        decoder_output: tf.Tensor = self.embedding(decoder_input)

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
        concat_output: tf.Tensor = tf.concat([decoder_output, context_vector], axis=-1)

        # Generate outputs
        decoder_output: tf.Tensor = self.dense(concat_output)  # Shape: (batch_size, 1, vocab_size)

        # Collect all states
        decoder_states: List[tf.Tensor] = [state_h1, state_c1, state_h2, state_c2, state_h3, state_c3,
                                           state_h4, state_c4]

        return decoder_output, decoder_states

    @staticmethod
    def compute_mask(inputs: Any, mask: Optional[Any] = None) -> None:
        """
        This layer does not propagate the mask further.

        Args:
            inputs (Any): Input tensors.
            mask (Optional[Any], optional): Input mask. Defaults to None.

        Returns:
            None
        """
        return None

    def get_config(self) -> dict:
        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict: A Python dictionary containing the layer's configuration.
        """
        config = super(StackedLSTMDecoder, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'decoder_embedding_dim': self.embedding.output_dim,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'embedding': tf.keras.layers.serialize(self.embedding),
            'lstm_decoder_1': tf.keras.layers.serialize(self.lstm_decoder_1),
            'dropout_1': tf.keras.layers.serialize(self.dropout_1),
            'lstm_decoder_2': tf.keras.layers.serialize(self.lstm_decoder_2),
            'dropout_2': tf.keras.layers.serialize(self.dropout_2),
            'lstm_decoder_3': tf.keras.layers.serialize(self.lstm_decoder_3),
            'dropout_3': tf.keras.layers.serialize(self.dropout_3),
            'lstm_decoder_4': tf.keras.layers.serialize(self.lstm_decoder_4),
            'dropout_4': tf.keras.layers.serialize(self.dropout_4),
            'attention': tf.keras.layers.serialize(self.attention),
            'dense': tf.keras.layers.serialize(self.dense),
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
        # Deserialize layers
        config['embedding'] = tf.keras.layers.deserialize(config['embedding'])
        config['lstm_decoder_1'] = tf.keras.layers.deserialize(config['lstm_decoder_1'])
        config['dropout_1'] = tf.keras.layers.deserialize(config['dropout_1'])
        config['lstm_decoder_2'] = tf.keras.layers.deserialize(config['lstm_decoder_2'])
        config['dropout_2'] = tf.keras.layers.deserialize(config['dropout_2'])
        config['lstm_decoder_3'] = tf.keras.layers.deserialize(config['lstm_decoder_3'])
        config['dropout_3'] = tf.keras.layers.deserialize(config['dropout_3'])
        config['lstm_decoder_4'] = tf.keras.layers.deserialize(config['lstm_decoder_4'])
        config['dropout_4'] = tf.keras.layers.deserialize(config['dropout_4'])
        config['attention'] = tf.keras.layers.deserialize(config['attention'])
        config['dense'] = tf.keras.layers.deserialize(config['dense'])
        return cls(**config)


class SimpleDecoder(Layer):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        units: int,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        super(SimpleDecoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.dropout_rate = dropout_rate

        # Define layers
        self.embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name='decoder_embedding'
        )
        self.lstm = LSTM(
            units,
            return_sequences=True,
            return_state=True,
            name='decoder_lstm'
        )
        self.dropout = Dropout(dropout_rate, name='decoder_dropout')
        self.dense = Dense(vocab_size, activation='softmax', name='decoder_dense')

    def build(self, input_shape):
        decoder_input_shape, initial_states_shape = input_shape

        self.embedding.build(decoder_input_shape)

        embedding_output_shape = self.embedding.compute_output_shape(decoder_input_shape)
        self.lstm.build(embedding_output_shape)

        lstm_output_shape = self.lstm.compute_output_shape(embedding_output_shape)
        self.dropout.build(lstm_output_shape)

        dropout_output_shape = self.dropout.compute_output_shape(lstm_output_shape)
        self.dense.build(dropout_output_shape)

        super(SimpleDecoder, self).build(input_shape)

    def call(
        self,
        inputs: Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        decoder_input, initial_state = inputs

        if decoder_input is None or initial_state is None:
            raise ValueError('decoder_input and initial_state must be provided to the Decoder.')

        # Embed input
        x = self.embedding(decoder_input)  # (batch_size, seq_len_decoder, embedding_dim)

        # Pass through LSTM
        lstm_output, state_h, state_c = self.lstm(
            x,
            initial_state=initial_state,
            training=training,
            mask=None  # LSTM uses mask from embedding
        )  # (batch_size, seq_len_decoder, units)

        # Apply Dropout
        lstm_output = self.dropout(lstm_output, training=training)  # (batch_size, seq_len_decoder, units)

        # Output layer
        output = self.dense(lstm_output)  # (batch_size, seq_len_decoder, vocab_size)

        return output

    def compute_mask(self, inputs: Tuple, mask: Optional[tf.Tensor] = None) -> None:
        # This decoder does not propagate the mask further
        return None

    def get_config(self) -> dict:
        config = super(SimpleDecoder, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'embedding': tf.keras.layers.serialize(self.embedding),
            'lstm': tf.keras.layers.serialize(self.lstm),
            'dropout': tf.keras.layers.serialize(self.dropout),
            'dense': tf.keras.layers.serialize(self.dense),
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'SimpleDecoder':
        # Deserialize layers
        config['embedding'] = tf.keras.layers.deserialize(config['embedding'])
        config['lstm'] = tf.keras.layers.deserialize(config['lstm'])
        config['dropout'] = tf.keras.layers.deserialize(config['dropout'])
        config['dense'] = tf.keras.layers.deserialize(config['dense'])
        return cls(**config)


