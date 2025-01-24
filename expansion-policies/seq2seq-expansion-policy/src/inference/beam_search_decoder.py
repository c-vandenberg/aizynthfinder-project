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

    Attributes
    ----------
    _decoder : DecoderInterface
        The decoder instance used for generating predictions.
    _beam_width : int
        The number of beams to keep during search.
    _num_groups : int
        The number of groups for diversity in beam search.
    _group_size : int
        The size of each group in beam search.
    _diversity_strength : float
        The strength of the diversity penalty.
    _max_length : int
        The maximum length of the generated sequences.
    _start_token_id : Optional[int]
        The token ID representing the start of a sequence.
    _end_token_id : Optional[int]
        The token ID representing the end of a sequence.
    _length_penalty : float
        The penalty applied to longer sequences.
    _return_top_n : int
        The number of top sequences to return.


    Methods
    -------
    search(encoder_output, initial_decoder_states)
        Perform beam search decoding to generate the best sequences.
    """
    def __init__(
        self,
        decoder: 'DecoderInterface',
        beam_width: int = 5,
        num_groups: int = 1,
        diversity_strength: float = 2.0,
        max_length: int = 140,
        start_token_id: int = None,
        end_token_id: int = None,
        length_penalty: float = 1.0,
        return_top_n: int = 1
    ) -> None:
        self._decoder = decoder
        self._beam_width = beam_width
        self._num_groups = num_groups
        self._group_size = beam_width // num_groups
        self._diversity_strength = diversity_strength
        self._max_length = max_length
        self._start_token_id = start_token_id
        self._end_token_id = end_token_id
        self._length_penalty = length_penalty
        self._return_top_n = return_top_n

    def search(
        self,
        encoder_output: tf.Tensor,
        initial_decoder_states: List[tf.Tensor],
    ) -> Tuple[List[List[List[int]]], List[List[float]]]:
        """
        Performs beam search decoding to generate the best sequences.

        Parameters
        ----------
        encoder_output : tf.Tensor
            The output tensor from the encoder of shape `[batch_size, seq_len, hidden_dim]`.
        initial_decoder_states : list of tf.Tensor
            The initial states for the decoder.

        Returns
        -------
        best_sequences : List[List[List[int]]]
            The best sequences generated for each sample in the batch.
        best_scores : List[List[float]]
            The scores corresponding to the best sequences.

        Notes
        -----
        This method implements beam search with optional group-based diversity to
        promote varied outputs. It maintains multiple hypotheses (beams) simultaneously
        and applies a length penalty to balance between sequence length and probability.
        """
        batch_size = tf.shape(encoder_output)[0]

        # Initialize sequences with the start token
        start_tokens = tf.fill([batch_size, 1], self._start_token_id)
        sequences = tf.tile(start_tokens, [1, self._beam_width])

        # Initialize scores with zeros
        scores = tf.zeros([batch_size, self._beam_width], dtype=tf.float32)

        # Initialize completed sequences and their scores
        completed_sequences = [[] for _ in range(batch_size)]
        completed_scores = [[] for _ in range(batch_size)]

        # Initialize finished flags
        finished = tf.zeros([batch_size, self._beam_width], dtype=tf.bool)

        # Expand encoder outputs for beam search
        encoder_outputs = tf.expand_dims(encoder_output, axis=1)
        encoder_outputs = tf.tile(encoder_outputs, [1, self._beam_width, 1, 1])
        flat_encoder_outputs = tf.reshape(
            encoder_outputs,
            [batch_size * self._beam_width, -1, encoder_output.shape[-1]]
        )

        # Tile the initial decoder states for beam search
        tiled_initial_states = []
        for state in initial_decoder_states:
            tiled_state = tf.expand_dims(state, axis=1)
            tiled_state = tf.tile(
                tiled_state,
                [1, self._beam_width, 1]
            )
            tiled_initial_states.append(tf.reshape(tiled_state, [batch_size * self._beam_width, -1]))

        # Flatten initial decoder states
        flat_decoder_states = tiled_initial_states

        for t in range(self._max_length):
            # Initialize per-batch, per-group tokens to keep track of tokens selected in each group at current time step
            group_tokens = [[ [] for _ in range(batch_size) ] for _ in range(self._num_groups)]

            # Reshape current sequences to [batch_size * beam_width, current_seq_length]
            flat_sequences = tf.reshape(sequences, [batch_size * self._beam_width, -1])

            # Get the last token from each sequence
            last_tokens = flat_sequences[:, -1]

            # Prepare decoder input
            decoder_input = tf.expand_dims(last_tokens, axis=1)

            # Run decoder single step
            decoder_output, new_states = self._decoder.single_step(
                decoder_input,
                flat_decoder_states,
                flat_encoder_outputs
            )

            # Squeeze the time dimension
            decoder_output = tf.squeeze(decoder_output, axis=1)  # Shape: [batch_size * beam_width, vocab_size]

            # Compute log probabilities
            log_probs = tf.math.log(decoder_output + 1e-10)

            # Apply temperature scaling to log_probs
            temperature = 2.0  # Adjust as needed
            log_probs = log_probs / temperature

            # Reshape log_probs to [batch_size, beam_width, vocab_size]
            log_probs = tf.reshape(log_probs, [batch_size, self._beam_width, -1])

            # Add random noise to promote diversity
            epsilon = 1e-1  # Increased epsilon
            noise = tf.random.uniform(tf.shape(log_probs), minval=0, maxval=epsilon)
            log_probs = log_probs + noise

            # Set log_probs of finished beams to -inf to prevent further expansion
            log_probs = tf.where(
                tf.expand_dims(finished, axis=2),
                tf.fill(tf.shape(log_probs), float('-inf')),
                log_probs
            )

            # Initialize lists to collect group results
            all_topk_scores = []
            all_topk_indices = []
            all_beam_indices = []

            for group_idx in range(self._num_groups):
                # Compute group offsets
                group_start = group_idx * self._group_size
                group_end = group_start + self._group_size

                # Extract log_probs and scores for this group
                group_log_probs = log_probs[:, group_start:group_end, :]
                group_scores = scores[:, group_start:group_end]

                # Add current scores to log_probs
                group_scores_expanded = tf.expand_dims(group_scores, axis=2)
                adjusted_scores = group_scores_expanded + group_log_probs

                # Build penalty mask
                penalty_mask = tf.zeros_like(adjusted_scores)
                if group_idx > 0:
                    for batch_idx in range(batch_size):
                        penalized_tokens = []
                        for prev_group_idx in range(group_idx):
                            penalized_tokens.extend(group_tokens[prev_group_idx][batch_idx])

                        if penalized_tokens:
                            penalized_tokens = tf.constant(penalized_tokens, dtype=tf.int32)
                            updates = tf.ones_like(penalized_tokens, dtype=tf.float32) * self._diversity_strength

                            # Build indices
                            indices = tf.stack(
                                [
                                    tf.repeat(tf.range(self._group_size), len(penalized_tokens)),
                                    tf.tile(penalized_tokens, [self._group_size])
                                ],
                                axis=-1
                            )

                            # Create penalty mask using scatter_nd
                            penalty_mask[batch_idx] = tf.tensor_scatter_nd_add(
                                penalty_mask[batch_idx],
                                indices,
                                updates
                            )

                # Subtract diversity penalty
                adjusted_scores -= penalty_mask

                # Apply length penalty
                adjusted_scores = adjusted_scores / tf.pow(tf.cast(t + 1, tf.float32), self._length_penalty)

                # Reshape to [batch_size, group_size * vocab_size]
                adjusted_scores_flat = tf.reshape(adjusted_scores, [batch_size, -1])

                # Get the top k scores and their indices for this group
                topk_scores, topk_indices = tf.math.top_k(adjusted_scores_flat, k=self._group_size, sorted=True)

                # Map topk_indices back to beam and token indices
                vocab_size = adjusted_scores.shape[2]
                beam_indices = topk_indices // vocab_size
                token_indices = topk_indices % vocab_size

                # Adjust beam indices to account for group offset
                beam_indices += group_start

                # Append to all beams
                all_topk_scores.append(topk_scores)
                all_topk_indices.append(token_indices)
                all_beam_indices.append(beam_indices)

                # Update group tokens
                for batch_idx in range(batch_size):
                    tokens = token_indices[batch_idx].numpy().tolist()
                    group_tokens[group_idx][batch_idx].extend(tokens)

            # Concatenate group results
            scores = tf.concat(all_topk_scores, axis=1)
            token_indices = tf.concat(all_topk_indices, axis=1)
            beam_indices = tf.concat(all_beam_indices, axis=1)

            # Gather the sequences corresponding to beam_indices
            batch_offsets = tf.range(batch_size)[:, None] * self._beam_width
            beam_indices_flat = beam_indices + batch_offsets
            beam_indices_flat = tf.reshape(beam_indices_flat, [batch_size * self._beam_width])

            # Gather the new sequences
            selected_sequences = tf.gather(flat_sequences, beam_indices_flat)
            new_tokens = tf.reshape(token_indices, [batch_size * self._beam_width, 1])
            new_sequences = tf.concat([selected_sequences, new_tokens], axis=1)

            # Update the finished mask
            is_finished = tf.equal(new_tokens, self._end_token_id)
            is_finished = tf.reshape(is_finished, [batch_size, self._beam_width])
            finished = tf.logical_or(finished, is_finished)

            # Update completed sequences and their scores
            for i in range(batch_size):
                for j in range(self._beam_width):
                    idx = i * self._beam_width + j
                    if is_finished[i, j]:
                        seq = new_sequences[idx].numpy().tolist()
                        score = scores[i, j].numpy()
                        if seq not in completed_sequences[i]:
                            completed_sequences[i].append(seq)
                            completed_scores[i].append(score)

            # Update sequences
            sequences = tf.reshape(new_sequences, [batch_size, self._beam_width, -1])

            # Update decoder states
            flat_decoder_states = []
            for state in new_states:
                state = tf.reshape(state, [batch_size * self._beam_width, -1])
                selected_states = tf.gather(state, beam_indices_flat)
                flat_decoder_states.append(selected_states)

            # Early stopping if all beams have finished
            if tf.reduce_all(finished):
                break

        # Collect top sequences
        best_sequences = []
        best_scores = []
        for i in range(batch_size):
            seq_score_pairs = list(zip(completed_sequences[i], completed_scores[i]))
            if len(seq_score_pairs) < self._return_top_n:
                remaining = self._return_top_n - len(seq_score_pairs)
                current_seqs = sequences[i].numpy().tolist()
                current_scores = scores[i].numpy().tolist()
                current_seq_score_pairs = list(zip(current_seqs, current_scores))
                seq_score_pairs.extend(current_seq_score_pairs)
            seq_score_pairs.sort(key=lambda x: x[1], reverse=True)
            top_n = min(self._return_top_n, len(seq_score_pairs))
            best_sequences.append([seq for seq, _ in seq_score_pairs[:top_n]])
            best_scores.append([score for _, score in seq_score_pairs[:top_n]])

        return best_sequences, best_scores
