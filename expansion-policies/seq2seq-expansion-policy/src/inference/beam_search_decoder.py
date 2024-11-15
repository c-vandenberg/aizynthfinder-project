from typing import List, Optional, Tuple

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
        start_tokens = tf.fill([batch_size, 1], self.start_token_id)  # Shape: [batch_size, 1]

        # Initialize sequences with the start token repeated beam_width times
        sequences = tf.tile(start_tokens, [1, self.beam_width])  # Shape: [batch_size, beam_width]

        # Initialize scores with zeros
        scores = tf.zeros([batch_size, self.beam_width], dtype=tf.float32)  # Shape: [batch_size, beam_width]

        # Initialize completed sequences and their scores
        completed_sequences = [[] for _ in range(batch_size)]
        completed_scores = [[] for _ in range(batch_size)]

        # Expand encoder outputs for beam search
        encoder_outputs = tf.expand_dims(encoder_output, axis=1)  # Shape: [batch_size, 1, seq_len_enc, enc_units]
        encoder_outputs = tf.tile(encoder_outputs, [1, self.beam_width, 1, 1])
        flat_encoder_outputs = tf.reshape(
            encoder_outputs,
            [batch_size * self.beam_width, -1, encoder_output.shape[-1]]
        )

        # Tile the initial decoder states for beam search
        tiled_initial_states = []
        for state in initial_decoder_states:
            tiled_state = tf.expand_dims(state, axis=1)  # Shape: [batch_size, 1, units]
            tiled_state = tf.tile(
                tiled_state,
                [1, self.beam_width, 1]
            )  # Shape: [batch_size, beam_width, units]
            tiled_initial_states.append(tf.reshape(tiled_state, [batch_size * self.beam_width, -1]))

        # Flatten initial decoder states
        flat_decoder_states = tiled_initial_states  # List of tensors

        for t in range(self.max_length):
            # Reshape current sequences to [batch_size * beam_width, current_seq_length]
            flat_sequences = tf.reshape(sequences, [batch_size * self.beam_width, -1])

            # Get the last token from each sequence
            last_tokens = flat_sequences[:, -1]  # Shape: [batch_size * beam_width]

            # Prepare decoder input
            decoder_input = tf.expand_dims(last_tokens, axis=1)  # Shape: [batch_size * beam_width, 1]

            # Run decoder single step
            decoder_output, new_states = self.decoder.single_step(
                decoder_input,
                flat_decoder_states,
                flat_encoder_outputs
            )  # decoder_output: [batch_size * beam_width, 1, vocab_size]

            # Squeeze the time dimension
            decoder_output = tf.squeeze(decoder_output, axis=1)  # Shape: [batch_size * beam_width, vocab_size]

            # Compute log probabilities
            log_probs = tf.math.log(decoder_output + 1e-10)  # Shape: [batch_size * beam_width, vocab_size]

            # Reshape log_probs to [batch_size, beam_width, vocab_size]
            log_probs = tf.reshape(log_probs, [batch_size, self.beam_width, -1])

            # Add current scores to log_probs
            scores_expanded = tf.expand_dims(scores, axis=2)  # Shape: [batch_size, beam_width, 1]
            total_scores = scores_expanded + log_probs  # Shape: [batch_size, beam_width, vocab_size]

            # Apply length penalty
            total_scores = total_scores / tf.pow((tf.cast(t + 1, tf.float32)), self.length_penalty)

            # Reshape to [batch_size, beam_width * vocab_size]
            total_scores_flat = tf.reshape(total_scores, [batch_size, -1])

            # Get the top beam_width scores and their indices
            topk_scores, topk_indices = tf.math.top_k(total_scores_flat, k=self.beam_width, sorted=True)

            # Calculate beam indices and token indices
            vocab_size = tf.shape(decoder_output)[1]
            beam_indices = topk_indices // vocab_size  # Shape: [batch_size, beam_width]
            token_indices = topk_indices % vocab_size  # Shape: [batch_size, beam_width]

            # Gather the sequences corresponding to beam_indices
            batch_offsets = tf.range(batch_size) * self.beam_width  # Shape: [batch_size]
            beam_indices_flat = beam_indices + tf.reshape(batch_offsets,
                                                          [batch_size, 1])  # Shape: [batch_size, beam_width]
            beam_indices_flat = tf.reshape(
                beam_indices_flat,
                [batch_size * self.beam_width]
            )  # Shape: [batch_size * beam_width]

            # Gather the new sequences
            selected_sequences = tf.gather(
                flat_sequences,
                beam_indices_flat
            )  # Shape: [batch_size * beam_width, seq_len]
            new_tokens = tf.reshape(
                token_indices,
        [batch_size * self.beam_width, 1]
            )  # Shape: [batch_size * beam_width, 1]
            new_sequences = tf.concat(
                [selected_sequences, new_tokens],
                axis=1
            )  # Shape: [batch_size * beam_width, seq_len + 1]

            # Check which sequences have finished
            is_finished = tf.equal(new_tokens, self.end_token_id)  # Shape: [batch_size * beam_width, 1]
            is_finished = tf.reshape(
                is_finished,
                [batch_size, self.beam_width]
            )  # Shape: [batch_size, beam_width]

            # Update completed sequences and their scores
            for i in range(batch_size):
                for j in range(self.beam_width):
                    if is_finished[i, j]:
                        seq = new_sequences[i * self.beam_width + j].numpy().tolist()
                        score = topk_scores[i, j].numpy()
                        if seq not in completed_sequences[i]:
                            completed_sequences[i].append(seq)
                            completed_scores[i].append(score)
                        # Set score to negative infinity to avoid re-selection
                        topk_scores = tf.tensor_scatter_nd_update(
                            topk_scores,
                            [[i, j]],
                            [float('-inf')]
                        )

            # Update sequences and scores
            sequences = tf.reshape(
                new_sequences,
                [batch_size, self.beam_width, -1]
            )  # Shape: [batch_size, beam_width, seq_len + 1]
            scores = topk_scores  # Shape: [batch_size, beam_width]

            # Update decoder states
            flat_decoder_states = []
            for state in new_states:
                state = tf.reshape(
                    state, [batch_size, self.beam_width, -1]
                )  # Shape: [batch_size, beam_width, units]
                flat_decoder_states.append(tf.reshape(state, [batch_size * self.beam_width, -1]))

            # Early stopping if all beams have finished
            if all(len(completed_sequences[i]) >= self.return_top_n for i in range(batch_size)):
                break

        # Collect top sequences
        best_sequences = []
        best_scores = []
        for i in range(batch_size):
            seq_score_pairs = list(zip(completed_sequences[i], completed_scores[i]))
            # If there are not enough completed sequences, fill with current sequences
            if len(seq_score_pairs) < self.return_top_n:
                remaining = self.return_top_n - len(seq_score_pairs)
                current_seqs = sequences[i].numpy().tolist()
                current_scores = scores[i].numpy().tolist()
                current_seq_score_pairs = list(zip(current_seqs, current_scores))
                # Sort current sequences by score
                current_seq_score_pairs.sort(key=lambda x: x[1], reverse=True)
                seq_score_pairs.extend(current_seq_score_pairs[:remaining])

            # Sort all sequences by score
            seq_score_pairs.sort(key=lambda x: x[1], reverse=True)
            top_n = min(self.return_top_n, len(seq_score_pairs))
            best_sequences.append([seq for seq, _ in seq_score_pairs[:top_n]])
            best_scores.append([score for _, score in seq_score_pairs[:top_n]])

        return best_sequences, best_scores
