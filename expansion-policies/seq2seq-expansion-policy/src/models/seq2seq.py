from typing import Optional, Any, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from encoders.lstm_encoders import StackedBidirectionalLSTMEncoder
from decoders.lstm_decoders import StackedLSTMDecoder


class RetrosynthesisSeq2SeqModel(Model):
    """
    Retrosynthesis Seq2Seq Model using LSTM layers.

    Model consists of an encoder and a decoder with attention mechanism for predicting reaction precursors
    for a target molecule.

    Parameters
    ----------
    input_vocab_size : int
        Size of the input vocabulary.
    output_vocab_size : int
        Size of the output vocabulary.
    encoder_embedding_dim : int
        Dimension of the embedding space for the encoder.
    decoder_embedding_dim : int
        Dimension of the embedding space for the decoder.
    units : int
        Number of units in LSTM layers.
    num_encoder_layers : int, optional
        Number of stacked layers in the encoder (default is 2).
    num_decoder_layers : int, optional
        Number of stacked layers in the decoder (default is 4).
    dropout_rate : float, optional
        Dropout rate for regularization (default is 0.2).
    **kwargs
        Additional keyword arguments for the Model superclass.

    Attributes
    ----------
    units : int
        Number of units in LSTM layers.
    encoder : StackedBidirectionalLSTMEncoder
        The encoder part of the Seq2Seq model.
    decoder : StackedLSTMDecoder
        The decoder part of the Seq2Seq model.
    input_vocab_size : int
        Size of the input vocabulary.
    output_vocab_size : int
        Size of the output vocabulary.
    enc_state_h : Dense
        Dense layer to map encoder hidden state to decoder initial hidden state.
    enc_state_c : Dense
        Dense layer to map encoder cell state to decoder initial cell state.
    encoder_data_processor : Any
        Data preprocessor for the encoder inputs (to be set externally).
    decoder_data_processor : Any
        Data preprocessor for the decoder inputs (to be set externally).
    dropout_rate : float
        Dropout rate for regularization.

    Methods
    -------
    call(inputs, training=None)
        Forward pass of the Seq2Seq model.
    get_config()
        Returns the configuration of the model.
    """
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        encoder_embedding_dim: int,
        decoder_embedding_dim: int,
        units: int,
        attention_dim: int,
        num_encoder_layers = 2,
        num_decoder_layers: int = 4,
        dropout_rate: float = 0.2,
        weight_decay: float = 1e-4,
        **kwargs
    ):
        super(RetrosynthesisSeq2SeqModel, self).__init__(**kwargs)

        self.units: int = units
        self.attention_dim: int = attention_dim

        # Encoder layer
        self.encoder: StackedBidirectionalLSTMEncoder = StackedBidirectionalLSTMEncoder(
            vocab_size=input_vocab_size,
            encoder_embedding_dim=encoder_embedding_dim,
            units=units,
            num_layers=num_encoder_layers,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )

        # Decoder layer
        self.decoder: StackedLSTMDecoder = StackedLSTMDecoder(
            vocab_size=output_vocab_size,
            decoder_embedding_dim=decoder_embedding_dim,
            attention_dim=attention_dim,
            units=units,
            num_layers=num_decoder_layers,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )

        self.input_vocab_size: int = input_vocab_size
        self.output_vocab_size: int = output_vocab_size

        # Mapping encoder final states to decoder initial states
        self.enc_state_h: Dense = Dense(units, name='enc_state_h')
        self.enc_state_c: Dense = Dense(units, name='enc_state_c')

        # Data processors to be set externally
        self.encoder_data_processor: Optional[Any] = None
        self.decoder_data_processor: Optional[Any] = None

        self.dropout_rate: float = dropout_rate
        self.weight_decay: float = weight_decay

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Forward pass of the Seq2Seq model.

        Parameters
        ----------
        inputs : tuple of tf.Tensor
            Tuple containing encoder_input and decoder_input:
            - encoder_input : tf.Tensor of shape (batch_size, seq_len_enc)
            - decoder_input : tf.Tensor of shape (batch_size, seq_len_dec)
        training : bool, optional
            Indicates whether the model is in training mode.

        Returns
        -------
        tf.Tensor
            Output predictions from the decoder.
            Shape: (batch_size, seq_len_dec, vocab_size)
        """
        # Unpack inputs
        encoder_input: tf.Tensor
        decoder_input: tf.Tensor
        encoder_input, decoder_input = inputs

        # Encoder input sequence processing
        encoder_output: tf.Tensor
        state_h: tf.Tensor
        state_c: tf.Tensor
        encoder_output, state_h, state_c = self.encoder(encoder_input, training=training)

        # Map encoder final states to decoder initial states
        decoder_initial_state_h: tf.Tensor = self.enc_state_h(state_h)  # Shape: (batch_size, units)
        decoder_initial_state_c: tf.Tensor = self.enc_state_c(state_c)  # Shape: (batch_size, units)
        decoder_initial_state: Tuple[tf.Tensor, tf.Tensor] = (decoder_initial_state_h, decoder_initial_state_c)

        # Prepare decoder inputs
        decoder_inputs = (
            decoder_input,
            decoder_initial_state,
            encoder_output
        )

        # Extract encoder mask
        encoder_mask: Optional[tf.Tensor] = self.encoder.compute_mask(encoder_input)

        # Decoder input sequence processing
        output: tf.Tensor = self.decoder(
            decoder_inputs,
            training=training,
            mask=encoder_mask
        )

        return output

    def predict_sequence(self, encoder_input, max_length=100, start_token_id=None, end_token_id=None):
        batch_size = tf.shape(encoder_input)[0]

        # Encode the input sequence
        encoder_output, state_h, state_c = self.encoder(encoder_input, training=False)

        # Map encoder final states to decoder initial states
        decoder_state_h = self.enc_state_h(state_h)
        decoder_state_c = self.enc_state_c(state_c)
        decoder_states = [decoder_state_h, decoder_state_c]

        # Prepare initial decoder input (start tokens)
        if start_token_id is None:
            start_token = self.decoder_data_processor.smiles_tokenizer.start_token
            start_token_id = self.decoder_data_processor.tokenizer.word_index[start_token]
        if end_token_id is None:
            end_token = self.decoder_data_processor.smiles_tokenizer.end_token
            end_token_id = self.decoder_data_processor.tokenizer.word_index[end_token]

        decoder_input = tf.fill([batch_size, 1], start_token_id)
        sequences = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        finished = tf.zeros([batch_size], dtype=tf.bool)

        for t in range(max_length):
            # Run decoder for one time step
            decoder_output, decoder_states = self.decoder.single_step(
                decoder_input,
                decoder_states,
                encoder_output
            )
            # Get the predicted token id
            predicted_id = tf.argmax(decoder_output, axis=-1, output_type=tf.int32)
            # Append to sequences
            sequences = sequences.write(t, predicted_id[:, 0])

            # Update finished status
            finished = tf.logical_or(finished, tf.equal(predicted_id[:, 0], end_token_id))

            # Break if all sequences are finished
            if tf.reduce_all(finished):
                break

            # Prepare next decoder input
            decoder_input = predicted_id

        sequences = sequences.stack()
        sequences = tf.transpose(sequences, [1, 0])  # shape (batch_size, seq_len)
        return sequences

    def get_config(self) -> dict:
        """
        Returns the configuration of the model for serialization.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        config = {
            'input_vocab_size': self.input_vocab_size,
            'output_vocab_size': self.output_vocab_size,
            'encoder_embedding_dim': self.encoder.embedding.output_dim,
            'decoder_embedding_dim': self.decoder.embedding.output_dim,
            'units': self.units,
            'attention_dim': self.attention_dim,
            'num_encoder_layers': self.encoder.num_layers,
            'num_decoder_layers': self.decoder.num_layers,
            'dropout_rate': self.dropout_rate,
            'weight_decay': self.weight_decay,
            'name': self.name,
        }
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'RetrosynthesisSeq2SeqModel':
        """
        Creates an instance of the model from its configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        RetrosynthesisSeq2SeqModel
            An instance of the model.
        """
        return cls(**config)
