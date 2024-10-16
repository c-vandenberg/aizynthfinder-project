from typing import List, Optional, Tuple, Union, Any

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (Embedding, LSTM, Dropout,
                                     Dense, Layer, LayerNormalization)

from decoders.decoder_interface import DecoderInterface
from attention.attention import BahdanauAttention


@tf.keras.utils.register_keras_serializable()
class StackedLSTMDecoder(DecoderInterface):
    """
    StackedLSTMDecoder

    A custom TensorFlow Keras layer that generates the target sequence in a Seq2Seq model using the encoder's final
    context/state vectors and its own previously generated tokens. It includes:

    Architecture:
        - Embedding Layer: Transforms target token indices into dense embedding vectors.
        - Stacked LSTM Layers: Processes embeddings through multiple LSTM layers to capture sequential dependencies
                                   and patterns.
        - Dropout Layers: Applies dropout after each LSTM layer for regularization.
        - Attention Mechanism: Utilizes a Bahdanau attention mechanism to focus on relevant parts of the encoder's
                                   outputs during decoding.
        - Output Dense Layer: Produces probability distributions over the target vocabulary via softmax activation.

    Parameters
    ----------
    vocab_size : int
        Size of the target vocabulary.
    embedding_dim : int
        Dimensionality of the embedding vectors.
    units : int
        Number of units in each LSTM layer.
    num_layers : int, optional
        Number of stacked LSTM layers (default is 4).
    dropout_rate : float, optional
        Dropout rate applied after each LSTM layer (default is 0.2).

    Methods
    -------
    call(inputs, training=False)
        Processes the input sequence, applies attention, and generates output probabilities over the target vocabulary.

    Returns
    -------
    decoder_output : tf.Tensor
        Predicted token probabilities for each timestep in the target sequence.
    """
    def __init__(
        self,
        vocab_size: int,
        decoder_embedding_dim: int,
        units: int,
        num_layers: int,
        attention_dim: int,
        dropout_rate: float = 0.2,
        weight_decay: float = 1e-4,
        **kwargs
    ) -> None:
        super(StackedLSTMDecoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, decoder_embedding_dim, mask_zero=True)
        self.units = units
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        self.supports_masking = True

        # Build LSTM, Dropout, and LayerNormalization layers
        self.lstm_layers = []
        self.dropout_layers = []
        self.layer_norm_layers = []
        for i in range(num_layers):
            lstm_layer = LSTM(
                units=units,
                return_sequences=True,
                return_state=True,
                kernel_regularizer=l2(weight_decay) if weight_decay is not None else None,
                name=f'lstm_decoder_{i + 1}'
            )
            self.lstm_layers.append(lstm_layer)

            dropout_layer = Dropout(dropout_rate, name=f'decoder_dropout_{i + 1}')
            self.dropout_layers.append(dropout_layer)

            layer_norm_layer = LayerNormalization(name=f'decoder_layer_norm_{i + 1}')
            self.layer_norm_layers.append(layer_norm_layer)

        # Attention Mechanism
        self.attention: BahdanauAttention = BahdanauAttention(
            units=attention_dim
        )

        # Output layer
        self.dense: Dense = Dense(
            vocab_size,
            activation='softmax',
            kernel_regularizer=l2(weight_decay) if weight_decay is not None else None
        )

    def call(
        self,
        inputs: Tuple[tf.Tensor, List[tf.Tensor], tf.Tensor],
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Processes the input sequence, applies attention, and generates output probabilities over the target vocabulary.

        Parameters
        ----------
        inputs : (Tuple[tf.Tensor, List[tf.Tensor], tf.Tensor])
            Tuple containing decoder input, initial state, and encoder output.
        training : (Optional[bool], optional)
            Training flag. Defaults to None.
        mask : (Optional[tf.Tensor], optional)
            Encoder mask. Defaults to None.

        Returns
        -------
        decoder_output : tf.Tensor
            The predicted token probabilities for each timestep in the target sequence.
        """
        # Unpack inputs
        decoder_input: tf.Tensor  # Shape: (batch_size, seq_len_dec)
        initial_state: List[tf.Tensor]  # List of tensors for initial hidden and cell states
        encoder_output: tf.Tensor  # Shape: (batch_size, seq_len_enc, enc_units)
        decoder_input, initial_state, encoder_output = inputs

        if decoder_input is None or initial_state is None or encoder_output is None:
            raise ValueError('decoder_input, initial_state and encoder_output must be passed to the Decoder.')

        # Unpack masks
        decoder_mask, encoder_mask = None, None
        if mask is not None:
            decoder_mask, encoder_mask = mask

        # Embed the input and extract decoder mask if not provided
        decoder_output = self.embedding(decoder_input) # Shape: (batch_size, seq_len_dec, decoder_embedding_dim)
        if decoder_mask is None:
            decoder_mask = self.embedding.compute_mask(decoder_input) # Shape: (batch_size, seq_len_dec)

        # Initialize previous_output with the embeddings
        previous_output = decoder_output

        # Prepare initial states for all layers
        num_states = len(initial_state)
        expected_states = self.num_layers * 2
        if num_states == 2:
            # Use initial_state (encoder final state) for the first LSTM layer
            states_list = [initial_state] + [(None, None)] * (self.num_layers - 1)
        elif num_states == expected_states:
            # Prepare initial states for all layers, initialising to zeros if not provided
            states_list = [
                (initial_state[i], initial_state[i + 1]) for i in range(0, num_states, 2)
            ]
        else:
            raise ValueError(f"Expected initial_state length to be 2 or {expected_states}, got {num_states}")

        # Process through decoder layers
        new_states = []
        for i, (lstm_layer, dropout_layer, layer_norm_layer) in enumerate(
                zip(self.lstm_layers, self.dropout_layers, self.layer_norm_layers)
        ):
            state_h, state_c = states_list[i]
            if state_h is None or state_c is None:
                batch_size = tf.shape(decoder_output)[0]
                state_h = tf.zeros((batch_size, self.units))
                state_c = tf.zeros((batch_size, self.units))

            decoder_output, state_h, state_c = lstm_layer(
                decoder_output,
                mask=decoder_mask,
                initial_state=[state_h, state_c],
                training=training
            )
            new_states.extend([state_h, state_c])

            # Apply Layer Normalization
            decoder_output = layer_norm_layer(decoder_output)

            # Apply residual connection from the second layer onwards
            if i > 0:
                decoder_output += previous_output

            # Update previous_output
            previous_output = decoder_output

            # Apply dropout
            decoder_output = dropout_layer(decoder_output, training=training) # Shape: (batch_size, seq_len_dec, units)

        # Extract only the encoder_mask if passed mask list of tuple
        if mask is not None and isinstance(mask, (list, tuple)):
            encoder_mask = mask[1]
        else:
            encoder_mask = mask

        # Apply attention mechanism
        context_vector: tf.Tensor  # Shape: (batch_size, seq_len_dec, enc_units)
        attention_weights: tf.Tensor  # Shape: (batch_size, seq_len_dec, seq_len_enc)
        context_vector, attention_weights = self.attention(
            inputs=[encoder_output, decoder_output],
            mask=encoder_mask
        )

        # Concatenate decoder outputs and context vector
        concat_output: tf.Tensor = tf.concat(
            [decoder_output, context_vector],
            axis=-1
        )  # Shape: (batch_size, seq_len_dec, units + units_enc)

        # Generate output probabilities
        decoder_output: tf.Tensor = self.dense(concat_output)  # Shape: (batch_size, seq_len_dec, vocab_size)

        return decoder_output

    def single_step(
        self,
        decoder_input: tf.Tensor,
        states: List[tf.Tensor],
        encoder_output: tf.Tensor
    ) -> tuple[Tensor, list[Tensor]]:
        """
        Performs a single decoding step.

        Parameters
        ----------
        decoder_input: tf.Tensor
            The input token at the current time step (shape: [batch_size, 1]).
        states: List[tf.Tensor]
            The initial state of the decoder LSTM layers (list of tensors).
        encoder_output: tf.Tensor
            The output from the encoder (shape: [batch_size, seq_len_enc, units]).

        Returns
        ----------
        decoder_output, new_states: Tuple[tf.Tensor]
            Tuple of (decoder_output, new_states)
        """
        # Embed the input
        decoder_output: tf.Tensor = self.embedding(decoder_input) # Shape: (batch_size, 1, decoder_embedding_dim)

        # Prepare the initial states
        num_states = len(states)
        expected_states = self.num_layers * 2  # hidden (h) and cell (c) states for each layer

        if num_states == 2:
            # Use initial_state (encoder final state) for the first LSTM layer
            states_list = [states] + [(None, None)] * (self.num_layers - 1)
        elif num_states == expected_states:
            # Prepare initial states for all layers, initialising to zeros if not provided
            states_list: List[Tuple[tf.Tensor, tf.Tensor]] = [
                (states[i], states[i + 1]) for i in range(0, num_states, 2)
            ]
        else:
            raise ValueError(f"Expected states length to be 2 or {expected_states}, got {num_states}")

        new_states: List[tf.Tensor] = []
        previous_output = decoder_output
        for i, (lstm_layer, dropout_layer, layer_norm_layer) in enumerate(
                zip(self.lstm_layers, self.dropout_layers, self.layer_norm_layers)
        ):
            state_h: tf.Tensor
            state_c: tf.Tensor
            state_h, state_c = states_list[i]
            if state_h is None or state_c is None:
                batch_size = tf.shape(decoder_output)[0]
                state_h = tf.zeros((batch_size, self.units))
                state_c = tf.zeros((batch_size, self.units))
            decoder_output, state_h, state_c = lstm_layer(
                decoder_output,
                initial_state=[state_h, state_c],
                training=False
            )
            new_states.extend([state_h, state_c]) # Shape: (batch_size, 1, units)

            # Apply Layer Normalization
            decoder_output = layer_norm_layer(decoder_output)

            # Apply residual connection from the second layer onwards
            if i > 0:
                decoder_output += previous_output

            # Update previous_output
            previous_output = decoder_output

            # Apply dropout
            decoder_output = dropout_layer(decoder_output, training=False)

        # Apply attention mechanism
        context_vector: tf.Tensor  # Shape: (batch_size, 1, enc_units)
        attention_weights: tf.Tensor  # Shape: (batch_size, 1, seq_len_enc)
        context_vector, attention_weights = self.attention(
            inputs=[encoder_output, decoder_output],
            mask=None  # No mask during inference
        )

        # Concatenate decoder outputs and context vector
        concat_output = tf.concat([decoder_output, context_vector], axis=-1) # Shape: (batch_size, 1, units + enc_units)

        # Generate output probabilities
        decoder_output = self.dense(concat_output) # Shape: (batch_size, 1, vocab_size)

        return decoder_output, new_states

    def compute_mask(
        self,
        inputs: Union[tf.Tensor, List[tf.Tensor]],
        mask: Optional[Any] = None
    ) -> None:
        """
        Propagates the mask forward by computing an output mask tensor for the layer.

        Parameters
        ----------
        inputs: Union[tf.Tensor, List[tf.Tensor]]
            A tensor or list of tensors.

        Returns
        ----------
        decoder_mask: tf.Tensor
            Mask tensor based on the decoder's mask.
        """
        decoder_input, initial_state, encoder_output = inputs
        decoder_mask = self.embedding.compute_mask(decoder_input)

        return decoder_mask

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.

        Returns
        ----------
        config: dict
            A Python dictionary containing the layer's configuration.
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'decoder_embedding_dim': self.embedding.output_dim,
            'units': self.units,
            'num_layers': self.num_layers,
            'attention_dim': self.attention_dim,
            'dropout_rate': self.dropout_rate,
            'weight_decay': self.weight_decay
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'StackedLSTMDecoder':
        """
        Creates a layer from its config.

        Parameters
        ----------
        config: dict
            A Python dictionary containing the layer's configuration.

        Returns
        ----------
        decoder: StackedLSTMDecoder
            A new instance of StackedLSTMDecoder configured using the provided config.
        """
        return cls(**config)
