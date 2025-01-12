from typing import Optional, Any, Tuple, List, Dict

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from data.utils.preprocessing import SmilesTokeniser
from encoders.lstm_encoders import StackedBidirectionalLSTMEncoder
from decoders.lstm_decoders import StackedLSTMDecoder
from inference.beam_search_decoder import BeamSearchDecoder


class RetrosynthesisSeq2SeqModel(Model):
    """
    Retrosynthesis Seq2Seq Model using LSTM layers.

    The model consists of an encoder and a decoder with an attention mechanism for predicting reaction precursors
    for a target molecule.

    This architecture leverages stacked Bidirectional LSTM layers in the encoder to capture contextual information
    from both directions, enhancing the model's understanding of the input sequences. The decoder employs stacked
    LSTM layers with attention to generate accurate and contextually relevant output sequences.

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments for the Model superclass.

    Attributes
    ----------
    units : int
        Number of units in LSTM layers.
    attention_dim : int
        Dimensionality of the attention mechanism.
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
    smiles_tokenizer : Optional[SmilesTokeniser]
        Data preprocessor for the encoder and decoder inputs.
    dropout_rate : float
        Dropout rate for regularization.
    weight_decay : Optional[float]
        L2 regularization factor applied to the LSTM and Dense layers.

    Methods
    -------
    call(inputs, training=None)
        Forward pass of the Seq2Seq model.
    predict_sequence(encoder_input, max_length=100, start_token_id=None, end_token_id=None)
        Generate a single sequence prediction using greedy decoding.
    predict_sequence_beam_search(encoder_input, beam_width=5, max_length=140, start_token_id=None, end_token_id=None)
        Generate sequence predictions using beam search decoding.
    _encode_and_initialize(encoder_input, start_token_id=None, end_token_id=None)
        Helper method to encode input and initialize decoder states.
    get_config()
        Returns the configuration of the model.
    from_config(config)
        Creates an instance of the model from its configuration.
    """
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        encoder_embedding_dim: int,
        decoder_embedding_dim: int,
        units: int,
        attention_dim: int,
        smiles_tokenizer: SmilesTokeniser,
        num_encoder_layers = 2,
        num_decoder_layers: int = 4,
        dropout_rate: float = 0.2,
        weight_decay: float = None,
        **kwargs
    ):
        super(RetrosynthesisSeq2SeqModel, self).__init__(**kwargs)

        self.units = units
        self.attention_dim = attention_dim

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

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        # Initialise Dense layers to pass encoder final states through
        self.enc_state_h: Dense = Dense(units, name='enc_state_h')
        self.enc_state_c: Dense = Dense(units, name='enc_state_c')

        # Smiles tokenizer to be set externally
        self.smiles_tokenizer = smiles_tokenizer
        self.tokenizer_vocab_size = self.smiles_tokenizer.vocab_size

        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Forward pass of the Seq2Seq model.

        Parameters
        ----------
        inputs : Tuple[tf.Tensor, tf.Tensor]
            Tuple containing encoder_input and decoder_input:
                - encoder_input : tf.Tensor of shape (batch_size, seq_len_enc)
                - decoder_input : tf.Tensor of shape (batch_size, seq_len_dec)
        training : bool, optional
            Indicates whether the model is in training mode.

        Returns
        -------
        output : tf.Tensor
            Output predictions from the decoder.
            Shape: (batch_size, seq_len_dec, vocab_size)

        Raises
        ------
        ValueError
            If `smiles_tokenizer` is not set.
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

        # Pass encoder final states through Dense layers
        # and map encoder final states to initial states for the decoder's first layer
        decoder_initial_state_h: tf.Tensor = self.enc_state_h(state_h)  # Shape: (batch_size, units)
        decoder_initial_state_c: tf.Tensor = self.enc_state_c(state_c)  # Shape: (batch_size, units)
        decoder_initial_state: List[tf.Tensor] = [decoder_initial_state_h, decoder_initial_state_c]

        # Prepare initial states for all decoder layers
        decoder_initial_state = (decoder_initial_state +
                                 [tf.zeros_like(decoder_initial_state_h)] * (self.decoder.num_layers * 2 - 2))

        # Prepare decoder inputs
        decoder_inputs: Tuple[tf.Tensor, List[tf.Tensor], tf.Tensor] = (
            decoder_input,
            decoder_initial_state,
            encoder_output
        )

        # Decoder input sequence processing
        output: tf.Tensor = self.decoder(
            decoder_inputs,
            training=training,
            mask=[decoder_mask, encoder_mask]
        )

        return output

    @tf.function
    def predict_sequence_tf(
            self,
            encoder_input: tf.Tensor,
            max_length: int = 140,
            start_token_id: int = None,
            end_token_id: int = None
    ) -> tf.Tensor:
        """
        Generate a sequence prediction using a TF while_loop for greedy decoding.

        This refactors the step-by-step for-loop into a compiled graph.
        It should reduce Python overhead and allow more efficient GPU usage.

        Parameters
        ----------
        encoder_input : tf.Tensor
            Tensor of shape (batch_size, seq_len_enc) containing encoder input sequences.
        max_length : int, optional
            Maximum length of the generated sequences (default=100).
        start_token_id : int, optional
            The token ID representing the start of a sequence.
            If None, it is retrieved from the tokenizer in _encode_and_initialize().
        end_token_id : int, optional
            The token ID representing the end of a sequence.
            If None, it is retrieved from the tokenizer in _encode_and_initialize().

        Returns
        -------
        tf.Tensor
            Tensor of shape (batch_size, generated_seq_length)
            containing the generated token IDs.
        """
        # 1) Encode and set up initial states
        batch_size, encoder_output, initial_decoder_states, start_token_id, end_token_id = self._encode_and_initialize(
            encoder_input,
            start_token_id=start_token_id,
            end_token_id=end_token_id
        )

        # 2) Prepare storage for output tokens via a TensorArray
        #    We write one time-step’s predictions at index i
        sequences_ta = tf.TensorArray(
            dtype=tf.int32,
            size=max_length,
            dynamic_size=False
        )

        # 3) Initialize loop variables
        idx = tf.constant(0, dtype=tf.int32)
        finished = tf.zeros([batch_size], dtype=tf.bool)
        # Decoder input at first step is just the <start> token
        decoder_input = tf.fill([batch_size, 1], start_token_id)
        max_length_tf = tf.constant(max_length, dtype=tf.int32)

        # 4) Define the loop condition and body as lambda
        cond_func = lambda idx_, sequences_ta_, finished_, decoder_input_, decoder_states_: (
            self._predict_sequence_cond(
                idx_, sequences_ta_, finished_, decoder_input_, decoder_states_, max_length_tf
            )
        )
        body_func = lambda idx_, sequences_ta_, finished_, decoder_input_, decoder_states_: (
            self._predict_sequence_body(
                idx_, sequences_ta_, finished_, decoder_input_, decoder_states_,
                encoder_output, end_token_id
            )
        )

        # 5) Execute the `tf.while_loop`
        _, sequences_ta, _, _, _ = tf.while_loop(
            cond=cond_func,
            body=body_func,
            loop_vars=[idx, sequences_ta, finished, decoder_input, initial_decoder_states]
        )

        # 7) Finalize the output sequence shape
        sequences = sequences_ta.stack()  # (max_length, batch_size)
        sequences = tf.transpose(sequences, [1, 0])  # (batch_size, max_length)

        return sequences

    def predict_sequence(
        self,
        encoder_input: tf.Tensor,
        max_length: int = 140,
        start_token_id: Optional[int] = None,
        end_token_id: Optional[int] = None
    ) -> tf.Tensor:
        """
        Generate a single sequence prediction using greedy decoding.

        Parameters
        ----------
        encoder_input : tf.Tensor
            Tensor of shape `(batch_size, seq_len_enc)` containing encoder input sequences.
        max_length : int, optional
            Maximum length of the generated sequences (default is 100).
        start_token_id : Optional[int], optional
            The token ID representing the start of a sequence. If `None`, it is retrieved from the tokenizer.
        end_token_id : Optional[int], optional
            The token ID representing the end of a sequence. If `None`, it is retrieved from the tokenizer.

        Returns
        -------
        sequences : tf.Tensor
            Tensor of shape `(batch_size, generated_seq_length)` containing the generated sequences.

        Raises
        ------
        ValueError
            If `smiles_tokenizer` is not set.
        """
        batch_size, encoder_output, decoder_states, start_token_id, end_token_id = self._encode_and_initialize(
            encoder_input,
            start_token_id,
            end_token_id
        )

        decoder_input = tf.fill([batch_size, 1], start_token_id)
        sequences = tf.TensorArray(tf.int32, size=max_length, dynamic_size=False)
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

            # Check for end token
            if tf.reduce_all(tf.equal(predicted_id[:, 0], end_token_id)):
                break

        sequences = sequences.stack()
        sequences = tf.transpose(sequences, [1, 0])  # shape (batch_size, seq_len)
        return sequences

    def predict_sequence_beam_search(
        self,
        encoder_input,
        beam_width: int = 5,
        max_length: int = 140,
        start_token_id: Optional[int] = None,
        end_token_id: Optional[int] = None,
        return_top_n=1
    ) -> Tuple[List[List[List[int]]], List[List[float]]]:
        """
        Generate sequence predictions using beam search decoding.

        Parameters
        ----------
        encoder_input : tf.Tensor
            Tensor of shape `(batch_size, seq_len_enc)` containing encoder input sequences.
        beam_width : int, optional
            The number of beams to keep during search (default is 5).
        max_length : int, optional
            Maximum length of the generated sequences (default is 140).
        start_token_id : Optional[int], optional
            The token ID representing the start of a sequence. If `None`, it is retrieved from the tokenizer.
        end_token_id : Optional[int], optional
            The token ID representing the end of a sequence. If `None`, it is retrieved from the tokenizer.

        Returns
        -------
        best_sequences : List[List[int]]
            The best decoded sequences for each item in the batch, represented as lists of token IDs.

        Raises
        ------
        ValueError
            If `smiles_tokenizer` is not set.
        """
        batch_size, encoder_output, initial_decoder_states, start_token_id, end_token_id = self._encode_and_initialize(
            encoder_input,
            start_token_id,
            end_token_id
        )

        # Initialize BeamSearchDecoder
        beam_search_decoder = BeamSearchDecoder(
            decoder=self.decoder,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            beam_width=beam_width,
            max_length=max_length,
            return_top_n=return_top_n
        )

        # Perform beam search decoding
        best_sequences, best_scores = beam_search_decoder.search(
            encoder_output=encoder_output,
            initial_decoder_states=initial_decoder_states
        )

        return best_sequences, best_scores

    def _encode_and_initialize(
        self,
        encoder_input: tf.Tensor,
        start_token_id: Optional[int] = None,
        end_token_id: Optional[int] = None
    ) -> Tuple[int, tf.Tensor, List[tf.Tensor], int, int]:
        """
        Helper method to encode input and initialize decoder states.

        Parameters
        ----------
        encoder_input : tf.Tensor
            Tensor of shape `(batch_size, seq_len_enc)` containing encoder input sequences.
        start_token_id : Optional[int], optional
            The token ID representing the start of a sequence. If `None`, it is retrieved from the tokenizer.
        end_token_id : Optional[int], optional
            The token ID representing the end of a sequence. If `None`, it is retrieved from the tokenizer.

        Returns
        -------
        Tuple[int, tf.Tensor, List[tf.Tensor], int, int]
            - batch_size : int
                Number of samples in the batch.
            - encoder_output : tf.Tensor
                Encoder output tensor.
            - initial_decoder_states : List[tf.Tensor]
                Initial states for the decoder.
            - start_token_id : int
                Token ID representing the start of a sequence.
            - end_token_id : int
                Token ID representing the end of a sequence.

        Raises
        ------
        ValueError
            If `smiles_tokenizer` is not set.
            If `start_token_id` or `end_token_id` cannot be found in the tokenizer.
        """
        # 1) Encode the input sequence
        batch_size = tf.shape(encoder_input)[0]
        encoder_output, state_h, state_c = self.encoder(encoder_input, training=False)

        # 2) Map encoder final states to first-layer decoder states
        decoder_state_h = self.enc_state_h(state_h)
        decoder_state_c = self.enc_state_c(state_c)
        initial_decoder_states = [decoder_state_h, decoder_state_c]

        # 3) Create a full set of (2 * `num_decoder_layers`) states
        #    The first pair are the mapped encoder final states, and the rest of zero-initialised
        initial_decoder_states = [decoder_state_h, decoder_state_c]
        for _ in range(self.decoder.num_layers - 1):
            initial_decoder_states.append(tf.zeros_like(decoder_state_h))
            initial_decoder_states.append(tf.zeros_like(decoder_state_c))

        # 4) Prepare start and end token IDs
        if start_token_id is None:
            start_token = self.smiles_tokenizer.start_token
            start_token_id = self.smiles_tokenizer.word_index[start_token]
            if start_token_id is None:
                raise ValueError(f"Start token '{start_token}' not found in tokenizer's word_index.")

        if end_token_id is None:
            end_token = self.smiles_tokenizer.end_token
            end_token_id = self.smiles_tokenizer.word_index[end_token]
            if end_token_id is None:
                raise ValueError(f"End token '{end_token}' not found in tokenizer's word_index.")

        return (
            batch_size,
            encoder_output,
            initial_decoder_states,
            start_token_id,
            end_token_id
        )

    @tf.function
    def _predict_sequence_cond(
        self,
        i: tf.Tensor,
        sequences_ta: tf.TensorArray,
        finished: tf.Tensor,
        decoder_input: tf.Tensor,
        decoder_states: List[tf.Tensor],
        max_length: tf.Tensor
    ) -> tf.Tensor:
        """
        Condition function for `tf.while_loop`: checks whether
        we have reached `max_length` or whether all sequences are finished.
        """
        not_all_finished = tf.logical_not(tf.reduce_all(finished))
        return tf.logical_and(i < max_length, not_all_finished)

    @tf.function
    def _predict_sequence_body(
        self,
        i: tf.Tensor,
        sequences_ta: tf.TensorArray,
        finished: tf.Tensor,
        decoder_input: tf.Tensor,
        decoder_states: List[tf.Tensor],
        encoder_output: tf.Tensor,
        end_token_id: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.TensorArray, tf.Tensor, tf.Tensor, List[tf.Tensor]]:
        """
        Body function for `tf.while_loop`:
        1) Calls `single_step`
        2) Argmax for predicted token
        3) Updates 'finished'
        4) Writes predictions to `sequences_ta`
        5) Returns updated loop variables
        """
        # 1) single-step decode
        decoder_output, new_states = self.decoder.single_step(
            decoder_input, decoder_states, encoder_output
        )
        # shape: (batch_size, 1, vocab_size)

        # 2) Argmax to get predicted token, then squeeze shape -> (batch_size,)
        predicted_id = tf.argmax(decoder_output, axis=-1, output_type=tf.int32)
        predicted_id = tf.squeeze(predicted_id, axis=-1)  # shape: (batch_size,)

        # 3) Update 'finished' status
        newly_finished = tf.equal(predicted_id, end_token_id)
        finished = tf.logical_or(finished, newly_finished)

        # 4) Write predictions to the TensorArray
        sequences_ta = sequences_ta.write(i, predicted_id)

        # 5) Next input must be shape (batch_size, 1)
        next_input = tf.expand_dims(predicted_id, axis=-1)

        return i + 1, sequences_ta, finished, next_input, new_states

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the model for serialization.

        This configuration can be used to re-instantiate the model with the same parameters.

        Returns
        -------
        config : Dict[str, Any]
            Configuration dictionary containing all necessary parameters to recreate the model.
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
    def from_config(cls, config: Dict[str, Any]) -> 'RetrosynthesisSeq2SeqModel':
        """
        Creates an instance of the model from its configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        RetrosynthesisSeq2SeqModel
            An instance of the model configured as per the provided dictionary.
        """
        return cls(**config)
