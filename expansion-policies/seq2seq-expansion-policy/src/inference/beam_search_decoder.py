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
        start_token_id: int,
        end_token_id: int,
        vocab_size: int,
        beam_width: int = 5,
        max_length: int = 140,
        length_penalty: float = 1.0,
        return_top_n: int = 1
    ) -> None:
        self.decoder = decoder
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.vocab_size = vocab_size
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.return_top_n = return_top_n

    def search(
            self,
            encoder_output: tf.Tensor,
            initial_decoder_states: List[tf.Tensor],
    ) -> Tuple[List[List[List[int]]], List[List[float]]]:
        batch_size = tf.shape(encoder_output)[0]

        # Initialize sequences with the start token
        start_tokens = tf.fill([batch_size], self.start_token_id)  # Shape: [batch_size]
        sequences = tf.expand_dims(start_tokens, axis=1)  # Shape: [batch_size, 1]

        # Initialize scores with zeros
        scores = tf.zeros([batch_size], dtype=tf.float32)  # Shape: [batch_size]

        # Initialize lengths (after first token)
        lengths = tf.ones([batch_size], dtype=tf.int32)  # Shape: [batch_size]

        # Initialize completed sequences and their scores
        completed_sequences = [[] for _ in range(batch_size)]
        completed_scores = [[] for _ in range(batch_size)]

        # Initial decoder states
        decoder_states = initial_decoder_states  # List of tensors, each of shape [batch_size, units]

        # First time step (t = 0)
        t = 0

        # Run decoder single step
        decoder_input = tf.expand_dims(start_tokens, axis=1)  # Shape: [batch_size, 1]
        decoder_output, decoder_states = self.decoder.single_step(
            decoder_input,
            decoder_states,
            encoder_output
        )  # decoder_output: [batch_size, 1, vocab_size]

        # Squeeze the time dimension
        decoder_output = tf.squeeze(decoder_output, axis=1)  # Shape: [batch_size, vocab_size]

        # Compute log probabilities
        log_probs = tf.math.log(decoder_output + 1e-10)  # Shape: [batch_size, vocab_size]

        # Expand scores to add log_probs
        scores_expanded = tf.expand_dims(scores, axis=1)  # Shape: [batch_size, 1]

        # Compute cumulative log probabilities
        adjusted_scores = scores_expanded + log_probs  # Shape: [batch_size, vocab_size]

        # Increment lengths (for the next step)
        lengths = lengths + 1  # Lengths become 2

        # Apply length penalty
        penalties = tf.pow((5.0 + tf.cast(lengths, tf.float32)) / 6.0, self.length_penalty)
        penalties = tf.expand_dims(penalties, axis=1)  # Shape: [batch_size, 1]
        total_scores = adjusted_scores / penalties  # Shape: [batch_size, vocab_size]

        # Get the top k scores and their indices
        topk_scores, topk_indices = tf.math.top_k(total_scores, k=self.beam_width, sorted=True)
        # Shapes: [batch_size, beam_width]

        # Expand sequences
        sequences = tf.expand_dims(sequences, axis=1)  # Shape: [batch_size, 1, seq_length]
        sequences = tf.tile(sequences, [1, self.beam_width, 1])  # Shape: [batch_size, beam_width, seq_length]
        sequences = tf.concat([sequences, tf.expand_dims(topk_indices, axis=2)], axis=2)  # Append new tokens
        sequences = tf.reshape(sequences,
                               [batch_size * self.beam_width, -1])  # Shape: [batch_size * beam_width, seq_length]

        # Update scores
        scores = tf.reshape(topk_scores, [-1])  # Shape: [batch_size * beam_width]

        # Update decoder states
        # For each layer in decoder_states, select the corresponding states for the top beams
        decoder_states = [tf.expand_dims(state, axis=1) for state in
                          decoder_states]  # Each state: [batch_size, 1, units]
        decoder_states = [tf.tile(state, [1, self.beam_width, 1]) for state in
                          decoder_states]  # [batch_size, beam_width, units]
        decoder_states = [tf.reshape(state, [batch_size * self.beam_width, -1]) for state in decoder_states]

        # Expand lengths to match beams
        lengths = tf.expand_dims(lengths, axis=1)  # Shape: [batch_size, 1]
        lengths = tf.tile(lengths, [1, self.beam_width])  # Shape: [batch_size, beam_width]
        lengths = tf.reshape(lengths, [batch_size * self.beam_width])  # Shape: [batch_size * beam_width]

        # Encoder outputs are repeated for each beam
        encoder_outputs = tf.repeat(encoder_output, repeats=self.beam_width,
                                    axis=0)  # Shape: [batch_size * beam_width, seq_len_enc, enc_units]

        # Debugging: Print initial sequences and scores
        print(f"\nTime step {t}:")
        print(f"Sequences shape: {sequences.shape}")
        print(f"Sequences: {sequences.numpy().tolist()}")
        print(f"Scores: {scores.numpy().tolist()}")
        print(f"Lengths: {lengths.numpy().tolist()}")

        # Limit the number of time steps to print for debugging
        max_print_time_steps = 5

        for t in range(1, self.max_length):
            # Get the last token from each sequence
            last_tokens = sequences[:, -1]  # Shape: [batch_size * beam_width]

            # Prepare decoder input
            decoder_input = tf.expand_dims(last_tokens, axis=1)  # Shape: [batch_size * beam_width, 1]

            # Run decoder single step
            decoder_output, new_states = self.decoder.single_step(
                decoder_input,
                decoder_states,
                encoder_outputs
            )  # decoder_output: [batch_size * beam_width, 1, vocab_size]

            # Squeeze the time dimension
            decoder_output = tf.squeeze(decoder_output, axis=1)  # Shape: [batch_size * beam_width, vocab_size]

            # Compute log probabilities
            log_probs = tf.math.log(decoder_output + 1e-10)  # Shape: [batch_size * beam_width, vocab_size]

            # Expand scores to add log_probs
            scores_expanded = tf.expand_dims(scores, axis=1)  # Shape: [batch_size * beam_width, 1]

            # Compute cumulative log probabilities
            adjusted_scores = scores_expanded + log_probs  # Shape: [batch_size * beam_width, vocab_size]

            # Apply length penalty
            penalties = tf.pow((5.0 + tf.cast(lengths, tf.float32)) / 6.0, self.length_penalty)
            penalties = tf.expand_dims(penalties, axis=1)  # Shape: [batch_size * beam_width, 1]
            total_scores = adjusted_scores / penalties

            # Reshape for top_k selection
            total_scores = tf.reshape(total_scores, [batch_size, self.beam_width * self.vocab_size])

            # Get the top k scores and their indices
            topk_scores, topk_indices = tf.math.top_k(total_scores, k=self.beam_width, sorted=True)

            # Calculate beam indices and token indices
            beam_indices = topk_indices // self.vocab_size  # Shape: [batch_size, beam_width]
            token_indices = topk_indices % self.vocab_size  # Shape: [batch_size, beam_width]

            # Flatten indices for gathering
            batch_offsets = tf.range(batch_size) * self.beam_width  # Shape: [batch_size]
            beam_indices_flat = beam_indices + tf.expand_dims(batch_offsets, axis=1)  # Shape: [batch_size, beam_width]
            beam_indices_flat = tf.reshape(beam_indices_flat, [-1])  # Shape: [batch_size * beam_width]

            # Gather sequences and append new tokens
            sequences = tf.gather(sequences, beam_indices_flat)
            sequences = tf.concat([sequences, tf.reshape(token_indices, [-1, 1])], axis=1)

            # Gather decoder states
            decoder_states = [tf.gather(state, beam_indices_flat) for state in new_states]

            # Update scores
            scores = tf.reshape(topk_scores, [-1])

            # Update lengths
            lengths = tf.gather(lengths, beam_indices_flat)
            # Check for end token to prevent increasing length after EOS
            is_finished = tf.equal(token_indices, self.end_token_id)  # Shape: [batch_size, beam_width]
            is_finished_flat = tf.reshape(is_finished, [-1])
            lengths = tf.where(is_finished_flat, lengths, lengths + 1)

            # Debugging: Print sequences and scores at current time step
            if t <= max_print_time_steps:
                print(f"\nTime step {t}:")
                print(f"Sequences shape: {sequences.shape}")
                sequences_list = sequences.numpy().tolist()
                print(f"Sequences: {sequences_list}")
                print(f"Scores: {scores.numpy().tolist()}")
                print(f"Lengths: {lengths.numpy().tolist()}")

            # Handle completed sequences
            for i in range(batch_size * self.beam_width):
                if is_finished_flat[i]:
                    seq = sequences[i].numpy().tolist()
                    score = scores[i].numpy()
                    length = lengths[i].numpy()
                    batch_idx = i // self.beam_width
                    # Adjust score with final length penalty
                    final_penalty = tf.pow((5.0 + tf.cast(length, tf.float32)) / 6.0, self.length_penalty)
                    final_score = scores[i].numpy() / final_penalty.numpy()
                    completed_sequences[batch_idx].append(seq)
                    completed_scores[batch_idx].append(final_score)
                    scores = tf.tensor_scatter_nd_update(
                        scores,
                        [[i]],
                        [float('-inf')]
                    )

            # Early stopping if all sequences are completed
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
                # Get current sequences and scores for this batch item
                start_idx = i * self.beam_width
                end_idx = start_idx + self.beam_width
                current_seqs = sequences[start_idx:end_idx].numpy().tolist()
                current_scores = scores[start_idx:end_idx].numpy().tolist()
                current_lengths = lengths[start_idx:end_idx].numpy().tolist()
                # Adjust current scores with length penalty
                for idx in range(len(current_scores)):
                    final_penalty = tf.pow((5.0 + tf.cast(current_lengths[idx], tf.float32)) / 6.0, self.length_penalty)
                    current_scores[idx] = current_scores[idx] / final_penalty.numpy()
                current_seq_score_pairs = list(zip(current_seqs, current_scores))
                # Sort current sequences by score
                current_seq_score_pairs.sort(key=lambda x: x[1], reverse=True)
                seq_score_pairs.extend(current_seq_score_pairs[:remaining])

            # Sort all sequences by score
            seq_score_pairs.sort(key=lambda x: x[1], reverse=True)
            best_sequences.append([seq for seq, _ in seq_score_pairs[:self.return_top_n]])
            best_scores.append([score for _, score in seq_score_pairs[:self.return_top_n]])

        return best_sequences, best_scores
