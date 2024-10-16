from typing import Optional, Any, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from data.utils.preprocessing import SmilesTokenizer
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

        # Smiles tokenizer to be set externally
        self.smiles_tokenizer: Optional[SmilesTokenizer] = None

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

        # Compute masks
        encoder_mask:Optional[tf.Tensor]  = self.encoder.compute_mask(encoder_input)
        decoder_mask: Optional[tf.Tensor]  = self.decoder.compute_mask([decoder_input, None, None])

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

        encoder_mask: Optional[tf.Tensor] = self.encoder.compute_mask(encoder_input)

        # Decoder input sequence processing
        output: tf.Tensor = self.decoder(
            decoder_inputs,
            training=training,
            mask=[decoder_mask, encoder_mask]
        )

        return output

    def predict_sequence(self, encoder_input, max_length=100, start_token_id=None, end_token_id=None):
        batch_size, encoder_output, decoder_states, start_token_id, end_token_id = self._encode_and_initialize(
            encoder_input,
            start_token_id,
            end_token_id
        )

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

    def predict_sequence_beam_search(
            self,
            encoder_input,
            beam_width=3,
            max_length=100,
            start_token_id=None,
            end_token_id=None
    ):
        batch_size, encoder_output, initial_decoder_states, start_token_id, end_token_id = self._encode_and_initialize(
            encoder_input,
            start_token_id,
            end_token_id
        )

        # Initialize beams
        sequences = []
        for i in range(batch_size):
            for _ in range(beam_width):
                sequences.append([start_token_id])  # Total length: batch_size * beam_width

        # Initialize scores
        scores = tf.zeros([batch_size, beam_width], dtype=tf.float32)  # Shape: (batch_size, beam_width)

        # Tile the initial decoder states across the beam width
        decoder_state_h, decoder_state_c = initial_decoder_states
        decoder_state_h = tf.tile(tf.expand_dims(decoder_state_h, 1),
                                  [1, beam_width, 1])  # Shape: (batch_size, beam_width, units)
        decoder_state_c = tf.tile(tf.expand_dims(decoder_state_c, 1), [1, beam_width, 1])
        states = [decoder_state_h, decoder_state_c]  # Each of shape (batch_size, beam_width, units)

        # Tile encoder outputs
        encoder_outputs = tf.tile(tf.expand_dims(encoder_output, 1),
                                  [1, beam_width, 1, 1])  # Shape: (batch_size, beam_width, seq_len_enc, enc_units)

        # Initialize finished flags
        finished = tf.zeros([batch_size, beam_width], dtype=tf.bool)  # Shape: (batch_size, beam_width)

        for t in range(max_length):
            # Prepare inputs
            last_tokens = [seq[-1] for seq in sequences]  # Length: batch_size * beam_width
            last_tokens = tf.convert_to_tensor(last_tokens, dtype=tf.int32)  # Shape: (batch_size * beam_width,)
            last_tokens = tf.reshape(last_tokens, [-1, 1])  # Shape: (batch_size * beam_width, 1)

            # Prepare states
            flat_states = []
            for s in states:
                # s shape: (batch_size, beam_width, units)
                s = tf.reshape(s, [batch_size * beam_width, -1])
                flat_states.append(s)

            # Prepare encoder outputs
            flat_encoder_outputs = tf.reshape(encoder_outputs, [batch_size * beam_width, -1, encoder_output.shape[-1]])

            # Run decoder
            decoder_output, new_states = self.decoder.single_step(
                last_tokens,
                flat_states,
                flat_encoder_outputs
            )

            # Get log probabilities
            log_probs = tf.math.log(decoder_output[:, 0, :] + 1e-10)  # Shape: (batch_size * beam_width, vocab_size)
            vocab_size = log_probs.shape[-1]

            # Reshape to (batch_size, beam_width, vocab_size)
            log_probs = tf.reshape(log_probs, [batch_size, beam_width, vocab_size])
            total_scores = tf.expand_dims(scores, 2) + log_probs  # Shape: (batch_size, beam_width, vocab_size)

            # Flatten beams
            flat_scores = tf.reshape(total_scores, [batch_size, -1])  # Shape: (batch_size, beam_width * vocab_size)

            # Get top beam_width scores and indices
            topk_scores, topk_indices = tf.nn.top_k(flat_scores, k=beam_width)

            # Compute new sequences, states, and finished flags
            new_sequences = []
            new_states_h = []
            new_states_c = []
            new_finished = []

            for i in range(batch_size):
                seqs = []
                states_h = []
                states_c = []
                fin = []
                for j in range(beam_width):
                    index = topk_indices[i, j].numpy()
                    beam_idx = index // vocab_size
                    token_idx = index % vocab_size

                    seq_idx = i * beam_width + beam_idx

                    # Get the previous sequence and append the new token
                    seq = sequences[seq_idx] + [token_idx]
                    seqs.append(seq)

                    # Get the corresponding state
                    state_idx = i * beam_width + beam_idx
                    states_h.append(new_states[0][state_idx])
                    states_c.append(new_states[1][state_idx])

                    # Check if finished
                    fin.append(token_idx == end_token_id)
                new_sequences.extend(seqs)
                new_states_h.extend(states_h)
                new_states_c.extend(states_c)
                new_finished.append(fin)

            # Update sequences, states, scores, and finished
            sequences = new_sequences  # Length: batch_size * beam_width
            scores = topk_scores  # Shape: (batch_size, beam_width)
            states = [
                tf.reshape(tf.stack(new_states_h, axis=0), [batch_size, beam_width, -1]),
                tf.reshape(tf.stack(new_states_c, axis=0), [batch_size, beam_width, -1])
            ]
            finished = tf.convert_to_tensor(new_finished, dtype=tf.bool)  # Shape: (batch_size, beam_width)

            # If all sequences are finished, break
            if tf.reduce_all(finished):
                break

        # Select best sequences for each batch item
        best_sequences = []
        for i in range(batch_size):
            # The best sequence is the one with the highest score
            best_idx = tf.argmax(scores[i]).numpy()
            seq_idx = i * beam_width + best_idx
            best_seq = sequences[seq_idx]
            best_sequences.append(best_seq)

        return best_sequences

    def _encode_and_initialize(
            self,
            encoder_input,
            start_token_id=None,
            end_token_id=None
    ):
        batch_size = tf.shape(encoder_input)[0]

        # Encode the input sequence
        encoder_output, state_h, state_c = self.encoder(encoder_input, training=False)

        # Map encoder final states to decoder initial states
        decoder_state_h = self.enc_state_h(state_h)
        decoder_state_c = self.enc_state_c(state_c)
        initial_decoder_states = [decoder_state_h, decoder_state_c]

        # Prepare start and end token IDs
        if start_token_id is None:
            start_token = self.smiles_tokenizer.start_token
            start_token_id = self.smiles_tokenizer.word_index[start_token]
        if end_token_id is None:
            end_token = self.smiles_tokenizer.end_token
            end_token_id = self.smiles_tokenizer.word_index[end_token]

        return batch_size, encoder_output, initial_decoder_states, start_token_id, end_token_id

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
