from typing import Optional, Union, Tuple, List, Dict

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dropout,
    Dense,
    Layer,
    LayerNormalization
)

from encoders.encoder_interface import EncoderInterface


@tf.keras.utils.register_keras_serializable()
class StackedBidirectionalLSTMEncoder(EncoderInterface):
    """
    StackedBidirectionalLSTMEncoder

    A custom TensorFlow Keras layer that encodes input sequences into context-rich representations
    for a Seq2Seq model.

    It leverages stacked Bidirectional LSTM layers to capture information from both past and future tokens,
    enhancing the encoder's ability to understand the input sequence comprehensively.

    Advanced architectural features such as residual connections and layer normalization are integrated to
    improve model performance and training stability.

    Architecture:
        - Embedding Layer: Converts input token indices into dense embedding vectors, enabling the model
            to learn meaningful representations of tokens.
        - Stacked Bidirectional LSTM Layers: Processes embeddings in both forward and backward directions
            across multiple layers to capture intricate sequential
            dependencies and contextual information.
        - Dropout Layers: Applies dropout after each LSTM layer to prevent overfitting by randomly
            deactivating a subset of neurons during training.
        - Layer Normalization Layers: Normalizes the outputs of each LSTM layer to stabilize and accelerate
            the training process by reducing internal covariate shift.
        - Residual Connections: Implements skip connections from the input of each LSTM layer to its output
            to facilitate better gradient flow and mitigate vanishing gradient issues in deep networks.

    Parameters
    ----------
    vocab_size : int
        Size of the input vocabulary.
    encoder_embedding_dim : int
        Dimensionality of the embedding vectors.
    units : int
        Number of units in each LSTM layer.
    num_layers : int
        Number of stacked Bidirectional LSTM layers.
    dropout_rate : float, optional
        Dropout rate applied after each LSTM layer (default is 0.2).
    weight_decay : float, optional
        L2 regularization factor applied to the LSTM layers (default is 1e-4).
    **kwargs : Any
        Additional keyword arguments passed to the base `Layer` class.

    Attributes
    ----------
    _embedding : Embedding
        Embedding layer that transforms input token indices into dense vectors.
    _bidirectional_lstm_layers : List[Bidirectional]
        List of stacked Bidirectional LSTM layers.
    _dropout_layers : List[Dropout]
        List of Dropout layers applied after each LSTM layer.
    _layer_norm_layers : List[LayerNormalization]
        List of Layer Normalization layers applied after each LSTM layer.
    _supports_masking : bool
        Indicates that the layer supports masking.

    Methods
    -------
    call(encoder_input, training=False)
        Encodes the input sequence and returns the encoder outputs and final states.
    compute_mask(inputs, mask=None)
        Computes the mask for the encoder input.
    get_config()
        Returns the configuration of the layer for serialization.
    from_config(config)
        Creates a StackedBidirectionalLSTMEncoder layer from its configuration.

    Returns
    -------
    encoder_output : tf.Tensor
        Encoded sequence representations with shape (batch_size, seq_len, units * 2).
    final_state_h : tf.Tensor
        Concatenated hidden state from the last Bidirectional LSTM layer with shape (batch_size, units * 2).
    final_state_c : tf.Tensor
        Concatenated cell state from the last Bidirectional LSTM layer with shape (batch_size, units * 2).
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_embedding_dim: int,
        units: int,
        num_layers: int,
        dropout_rate: float = 0.2,
        weight_decay: float = 1e-4,
        supports_masking: Optional[bool] = True,
        **kwargs
    ) -> None:
        super(StackedBidirectionalLSTMEncoder, self).__init__(**kwargs)
        self._vocab_size = vocab_size
        self._embedding = Embedding(vocab_size, encoder_embedding_dim, mask_zero=True)
        self._units= units
        self._num_layers = num_layers
        self._dropout_rate = dropout_rate
        self._weight_decay = weight_decay

        self._supports_masking = supports_masking

        # Build first Bidirectional LSTM layer
        self._bidirectional_lstm_layers = []
        self._dropout_layers = []
        self._layer_norm_layers = []
        self._residual_projection_layers = []
        for i in range(num_layers):
            lstm_layer = Bidirectional(
                LSTM(
                    units,
                    return_sequences=True,
                    return_state=True,
                    kernel_regularizer=l2(weight_decay) if weight_decay is not None else None
                ),
                name=f'bidirectional_lstm_encoder_{i + 1}'
            )
            self._bidirectional_lstm_layers.append(lstm_layer)

            dropout_layer = Dropout(dropout_rate, name=f'encoder_dropout_{i + 1}')
            self._dropout_layers.append(dropout_layer)

            layer_norm_layer = LayerNormalization(name=f'encoder_layer_norm_{i + 1}')
            self._layer_norm_layers.append(layer_norm_layer)

    def call(
        self,
        encoder_input: tf.Tensor,
        training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Encodes the input sequence and returns the encoder outputs and final states.

        This method processes the input sequences through embedding, stacked Bidirectional LSTM layers,
        applies dropout and layer normalization, and generates the final hidden and cell states by
        concatenating the forward and backward states from the last LSTM layer.

        Parameters
        ----------
        encoder_input : tf.Tensor
            Input tensor for the encoder with shape (batch_size, seq_len).
        training : Optional[bool], default=None
            Training flag. If `True`, dropout layers are active. Defaults to `None`.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            - encoder_output : tf.Tensor
                Encoded sequence representations with shape (batch_size, seq_len, units * 2).
            - final_state_h : tf.Tensor
                Concatenated hidden state from the last Bidirectional LSTM layer with shape (batch_size, units * 2).
            - final_state_c : tf.Tensor
                Concatenated cell state from the last Bidirectional LSTM layer with shape (batch_size, units * 2).

        Raises
        ------
        ValueError
            If `encoder_input` is `None` or not a 2D tensor.
        """
        if encoder_input is None:
            raise ValueError("encoder_input must not be None.")
        if len(encoder_input.shape) != 2:
            raise ValueError("encoder_input must be a 2D tensor (batch_size, seq_len).")

        # Embed the input and obtain mask
        encoder_output: tf.Tensor = self._embedding(encoder_input) # Shape: (batch_size, seq_len, embedding_dim)
        encoder_mask: tf.Tensor = self._embedding.compute_mask(encoder_input) # Shape: (batch_size, seq_len)

        final_state_h: Union[None, tf.Tensor] = None
        final_state_c: Union[None, tf.Tensor] = None

        # Initialize previous_output with the embeddings
        previous_output = encoder_output

        # Iterate through each stacked Bidirectional LSTM layer
        for i, (lstm_layer, dropout_layer, layer_norm_layer) in enumerate(
                zip(self._bidirectional_lstm_layers, self._dropout_layers, self._layer_norm_layers)
        ):
            # Pass through the Bidirectional LSTM layer
            # encoder_output shape: (batch_size, seq_len, units * 2)
            # forward_h shape: (batch_size, units)
            # backward_h shape: (batch_size, units)
            # forward_c shape: (batch_size, units)
            # backward_c shape: (batch_size, units)
            encoder_output, forward_h, forward_c, backward_h, backward_c = lstm_layer(
                encoder_output, mask=encoder_mask, training=training
            )

            # Concatenate the final forward and backward hidden states
            final_state_h = tf.concat([forward_h, backward_h], axis=-1) # Shape: (batch_size, units * 2)

            # Concatenate the final forward and backward cell states
            final_state_c = tf.concat([forward_c, backward_c], axis=-1) # Shape: (batch_size, units * 2)

            # Apply Layer Normalization
            encoder_output = layer_norm_layer(encoder_output)

            # Apply residual connection from the second layer onwards
            if i > 0:
                encoder_output += previous_output

            # Update previous_output
            previous_output = encoder_output

            # Apply dropout to the encoder output
            encoder_output: tf.Tensor = dropout_layer(
                encoder_output,
                training=training
            ) # Shape: (batch_size, seq_len, units * 2)

        if final_state_h is None or final_state_c is None:
            raise ValueError("No encoder bidirectional LSTM layers detected; num_layers must be at least 1.")

        return encoder_output, final_state_h, final_state_c

    def compute_mask(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None
    ) -> Optional[tf.Tensor]:
        """
        Computes the mask for the encoder input.

        This method propagates the mask forward by computing an output mask tensor
        based on the encoder's input mask. It ensures that the masking information
        is correctly passed to subsequent layers.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor for the encoder with shape (batch_size, seq_len).
        mask : Optional[tf.Tensor], default=None
            Input encoder mask with shape (batch_size, seq_len). If provided, it will be used to compute
            the output mask.

        Returns
        -------
        Optional[tf.Tensor]
            The mask tensor based on the encoder's input mask. Returns `None` if no mask is computed.
        """
        return self._embedding.compute_mask(inputs, mask)

    def get_config(self) -> Dict:
        """
        Returns the configuration of the layer for serialization.

        This method enables the custom layer to be serialized and deserialized with its
        configuration parameters, facilitating model saving and loading.

        Returns
        -------
        config: Dict[str, Any]
            A dictionary containing the layer's configuration, including parameters like
            `vocab_size`, `encoder_embedding_dim`, `units`, `num_layers`, `dropout_rate`,
            and `weight_decay`.
        """
        config = super(StackedBidirectionalLSTMEncoder, self).get_config()
        config.update({
            'vocab_size': self._embedding.input_dim,
            'encoder_embedding_dim': self._embedding.output_dim,
            'units': self._units,
            'num_layers': self._num_layers,
            'dropout_rate': self._dropout_rate,
            'weight_decay': self._weight_decay
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'StackedBidirectionalLSTMEncoder':
        """
        Creates a StackedBidirectionalLSTMEncoder layer from its configuration.

        This class method allows the creation of a new `StackedBidirectionalLSTMEncoder` instance
        from a configuration dictionary, enabling model reconstruction from saved configurations.

        Parameters
        ----------
        config : Dict[str, Any]
            A dictionary containing the configuration of the layer.

        Returns
        -------
        StackedBidirectionalLSTMEncoder
            A new instance of `StackedBidirectionalLSTMEncoder` configured using the provided config.
        """
        return cls(**config)

    @property
    def embedding(self) -> Embedding:
        """
        Returns the Embedding layer that transforms target token indices into dense vectors.

        Returns
        -------
        Embedding
            The decoders embedding layer.
        """
        return self._embedding

    @property
    def num_layers(self):
        """
        Returns the number of stacked Bidirectional LSTM layers in the encoder.

        Returns
        -------
        int
            Number of stacked Bidirectional LSTM layers.
        """
        return self._num_layers