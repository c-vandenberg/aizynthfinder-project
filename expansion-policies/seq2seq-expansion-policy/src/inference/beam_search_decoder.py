from typing import List, Optional, Tuple, Any

import tensorflow as tf

from decoders.decoder_interface import DecoderInterface


class BeamSearchDecoder:
    """
    BeamSearchDecoder

    Implements beam search decoding for Seq2Seq models using a specified decoder.

    Beam search efficiently explores multiple potential output sequences, maintaining the top `beam_width`
    candidates at each decoding step based on their cumulative log probabilities. This approach balances
    exploration and exploitation, improving the quality of generated sequences compared to greedy decoding.

    Architecture:
        - Decoder Integration: Utilizes a provided `DecoderInterface` instance to generate predictions at each
                                decoding step.
        - Beam Width: Maintains multiple hypotheses (beams) simultaneously, allowing the exploration of diverse
                                sequence possibilities.
        - Length Penalty: Applies a penalty to longer sequences to balance between sequence length and overall
                                probability, preventing the model from favoring excessively long outputs.
        - Early Stopping: Terminates the search early if all beams have generated the end-of-sequence token,
                                reducing unnecessary computations.

    Parameters
    ----------
    decoder : DecoderInterface
        The decoder instance used for generating predictions.
    beam_width : int, optional
        The number of beams to keep during search (default is 5).
    max_length : int, optional
        The maximum length of the generated sequences (default is 140).
    start_token_id : int, optional
        The token ID representing the start of a sequence.
    end_token_id : int, optional
        The token ID representing the end of a sequence.
    length_penalty : float, optional
        The penalty applied to longer sequences to balance between length and probability (default is 1.0).

    Attributes
    ----------
    decoder : DecoderInterface
        The decoder instance used for generating predictions.
    beam_width : int
        The number of beams to keep during search.
    max_length : int
        The maximum length of the generated sequences.
    start_token_id : Optional[int]
        The token ID representing the start of a sequence.
    end_token_id : Optional[int]
        The token ID representing the end of a sequence.
    length_penalty : float
        The penalty applied to longer sequences to balance between length and probability.

    Methods
    -------
    search(encoder_output, initial_decoder_states)
        Perform beam search decoding to generate the best sequences.
    """

    def __init__(
            self,
            decoder: 'DecoderInterface',
            beam_width: int = 5,
            max_length: int = 140,
            start_token_id: int = None,
            end_token_id: int = None,
        length_penalty: float = 1.0,
            return_top_n: int = 1
    ) -> None:
        self.decoder = decoder
        self.beam_width = beam_width
        self.max_length = max_length
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.length_penalty = length_penalty
        self.return_top_n = return_top_n

    def search(
        self,
        encoder_output: tf.Tensor,
        initial_decoder_states: List[tf.Tensor],
    ) -> Tuple[List[List[List[int]]], List[List[float]]]:
        """
        Perform beam search decoding to generate the best sequences.

        This method generates sequences by iteratively expanding the top `beam_width` candidates at each
        decoding step. It maintains a list of active beams and completed sequences, updating them based
        on the decoder's predictions and applying a length penalty to balance sequence length and
        cumulative log probabilities.

        Parameters
        ----------
        encoder_output : tf.Tensor
            The output from the encoder with shape `(batch_size, seq_len_enc, enc_units)`.
        initial_decoder_states : List[tf.Tensor]
            The initial state of the decoder LSTM layers (list of tensors).

        Returns:
        --------
        best_sequences : List[List[int]]
            The best decoded sequences for each item in the batch. Each sequence is represented as a
            list of token IDs.

        Raises
        ------
        ValueError
            If `encoder_output` is not a 3D tensor.
            If `initial_decoder_states` is empty or contains non-tensor elements.
        """
        batch_size = tf.shape(encoder_output)[0]

        # Initialize sequences with the start token
        start_tokens = tf.fill(
            [batch_size],
            self.start_token_id
        )  # Shape: [batch_size]

        # Initialize beam variables
        sequences = [[[]] * self.beam_width for _ in range(batch_size)]
        scores = tf.zeros([batch_size, self.beam_width], dtype=tf.float32)

        # Expand encoder outputs and initial states for beam search
        encoder_outputs = tf.repeat(encoder_output, repeats=self.beam_width, axis=0)
        decoder_states = [tf.repeat(state, repeats=self.beam_width, axis=0) for state in initial_decoder_states]

        # Initialize decoder inputs
        decoder_input = tf.fill([batch_size * self.beam_width, 1], self.start_token_id)

        # Beam search decoding loop
        for t in range(self.max_length):
            # Run decoder for one time step
            decoder_output, decoder_states = self.decoder.single_step(
                decoder_input,
                decoder_states,
                encoder_outputs
            )  # decoder_output: [batch_size * beam_width, 1, vocab_size]

            # Get log probabilities
            vocab_size = decoder_output.shape[-1]
            log_probs = tf.math.log(decoder_output[:, 0, :] + 1e-10)  # Shape: [batch_size * beam_width, vocab_size]

            # Reshape to [batch_size, beam_width, vocab_size]
            log_probs = tf.reshape(log_probs, [batch_size, self.beam_width, vocab_size])

            # Compute total scores
            scores_expanded = tf.expand_dims(scores, axis=2)  # Shape: [batch_size, beam_width, 1]
            total_scores = scores_expanded + log_probs  # Shape: [batch_size, beam_width, vocab_size]

            # Apply length penalty
            total_scores = total_scores / tf.pow((5.0 + tf.cast(t + 1, tf.float32)) / 6.0, self.length_penalty)

            # Flatten to [batch_size, beam_width * vocab_size]
            total_scores_flat = tf.reshape(total_scores, [batch_size, -1])

            # Get top beam_width scores and indices
            top_scores, top_indices = tf.math.top_k(total_scores_flat, k=self.beam_width)

            # Compute beam and token indices
            beam_indices = top_indices // vocab_size  # Shape: [batch_size, beam_width]
            token_indices = top_indices % vocab_size  # Shape: [batch_size, beam_width]

            # Prepare for the next iteration
            new_sequences = []
            all_finished = True
            for i in range(batch_size):
                seqs = []
                for j in range(self.beam_width):
                    beam_idx = beam_indices[i, j].numpy()
                    token_idx = token_indices[i, j].numpy()
                    seq = sequences[i][beam_idx] + [token_idx]
                    seqs.append(seq)
                new_sequences.append(seqs)

                # Check if all sequences have ended
                for seq in seqs:
                    if seq[-1] != self.end_token_id:
                        all_finished = False

            sequences = new_sequences
            scores = top_scores  # Shape: [batch_size, beam_width]

            # Prepare next decoder input
            decoder_input = tf.reshape(token_indices, [batch_size * self.beam_width, 1])

            # Early stopping if all sequences have ended
            if all_finished:
                break

        # Collect top sequences
        best_sequences = []
        best_scores = []
        for i in range(batch_size):
            # Collect sequences and scores
            seq_score_pairs = list(zip(sequences[i], scores[i].numpy()))
            # Sort by score
            seq_score_pairs.sort(key=lambda x: x[1], reverse=True)
            top_n = min(self.return_top_n, len(seq_score_pairs))
            best_sequences.append([seq for seq, _ in seq_score_pairs[:top_n]])
            best_scores.append([score for _, score in seq_score_pairs[:top_n]])

        return best_sequences, best_scores
