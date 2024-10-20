from typing import List

import tensorflow as tf

from decoders.decoder_interface import DecoderInterface


class BeamSearchDecoder:
    def __init__(
        self,
        decoder: DecoderInterface,
        start_token_id: int,
        end_token_id: int,
        beam_width: int = 5,
        max_length: int = 140
    ) -> None:
        """
        Initializes the BeamSearchDecoder.

        """
        self.decoder = decoder
        self.beam_width = beam_width
        self.max_length = max_length
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id

    def search(
        self,
        encoder_output: tf.Tensor,
        initial_decoder_states: List[tf.Tensor],
    ) -> List[List[int]]:
        """
        Perform beam search decoding.

        Parameters:
        -----------
        encoder_output: tf.Tensor
            The output from the encoder (shape: [batch_size, seq_len_enc, enc_units]).
        initial_decoder_states: List[tf.Tensor]
            The initial state of the decoder LSTM layers (list of tensors).

        Returns:
        --------
        best_sequences: List[List[int]]
            The best decoded sequences for each item in the batch.
        """
        batch_size = tf.shape(encoder_output)[0]

        # Initialize beams
        sequences = [[self.start_token_id] for _ in range(batch_size * self.beam_width)]  # List of sequences
        scores = tf.zeros([batch_size, self.beam_width], dtype=tf.float32)  # Shape: (batch_size, beam_width)

        # Tile the initial decoder states and encoder outputs across the beam width
        tiled_states = []
        for state in initial_decoder_states:
            state = tf.expand_dims(state, axis=1)  # Shape: (batch_size, 1, units)
            state = tf.tile(state, [1, self.beam_width, 1])  # Shape: (batch_size, beam_width, units)
            tiled_states.append(state)

        # Flatten the beam dimension into the batch dimension
        flat_states = [tf.reshape(state, [batch_size * self.beam_width, -1]) for state in tiled_states]

        encoder_outputs = tf.expand_dims(encoder_output, axis=1)  # Shape: (batch_size, 1, seq_len_enc, enc_units)
        encoder_outputs = tf.tile(encoder_outputs, [1, self.beam_width, 1, 1])  # Shape: (batch_size, beam_width, seq_len_enc, enc_units)
        flat_encoder_outputs = tf.reshape(encoder_outputs, [batch_size * self.beam_width, -1, encoder_output.shape[-1]])

        # Initialize finished flags
        finished = tf.zeros([batch_size, self.beam_width], dtype=tf.bool)  # Shape: (batch_size, beam_width)

        for t in range(self.max_length):
            # Prepare decoder input
            last_tokens = [seq[-1] for seq in sequences]  # List of last tokens
            decoder_input = tf.convert_to_tensor(last_tokens, dtype=tf.int32)  # Shape: (batch_size * beam_width,)
            decoder_input = tf.expand_dims(decoder_input, axis=1)  # Shape: (batch_size * beam_width, 1)

            # Run decoder single step
            decoder_output, new_states = self.decoder.single_step(
                decoder_input,
                flat_states,
                flat_encoder_outputs
            )

            # Get log probabilities
            decoder_output = tf.squeeze(decoder_output, axis=1)  # Shape: (batch_size * beam_width, vocab_size)
            log_probs = tf.math.log(decoder_output + 1e-10)  # Shape: (batch_size * beam_width, vocab_size)
            vocab_size = log_probs.shape[-1]

            # Reshape to (batch_size, beam_width, vocab_size)
            log_probs = tf.reshape(log_probs, [batch_size, self.beam_width, vocab_size])

            # Compute total scores
            total_scores = tf.expand_dims(scores, 2) + log_probs  # Shape: (batch_size, beam_width, vocab_size)

            # Flatten beams
            flat_scores = tf.reshape(total_scores, [batch_size, -1])  # Shape: (batch_size, beam_width * vocab_size)

            # Get top beam_width scores and indices
            topk_scores, topk_indices = tf.nn.top_k(flat_scores, k=self.beam_width)

            # Compute new sequences, states, and finished flags
            new_sequences = []
            new_states_list = [[] for _ in range(len(new_states))]  # Prepare a list for each state tensor
            new_finished = []

            for i in range(batch_size):
                seqs = []
                fin = []
                for j in range(self.beam_width):
                    index = topk_indices[i, j].numpy()
                    beam_idx = index // vocab_size
                    token_idx = index % vocab_size

                    seq_idx = i * self.beam_width + beam_idx

                    # Get the previous sequence and append the new token
                    seq = sequences[seq_idx] + [token_idx]
                    seqs.append(seq)

                    # Get the corresponding state
                    state_idx = seq_idx
                    for k in range(len(new_states)):
                        state_slice = new_states[k][state_idx]
                        new_states_list[k].append(state_slice)

                    # Check if finished
                    fin.append(token_idx == self.end_token_id)
                new_sequences.extend(seqs)
                new_finished.append(fin)

            # Update sequences, states, scores, and finished
            sequences = new_sequences  # Length: batch_size * beam_width
            scores = topk_scores  # Shape: (batch_size, beam_width)

            # Reconstruct flat_states from new_states_list
            flat_states = []
            for k in range(len(new_states_list)):
                state_tensor = tf.stack(new_states_list[k], axis=0)  # Shape: (batch_size * beam_width, units)
                flat_states.append(state_tensor)

            finished = tf.convert_to_tensor(new_finished, dtype=tf.bool)  # Shape: (batch_size, beam_width)

            # If all sequences are finished, break
            if tf.reduce_all(finished):
                break

        # Select best sequences for each batch item
        best_sequences = []
        for i in range(batch_size):
            # The best sequence is the one with the highest score
            best_idx = tf.argmax(scores[i]).numpy()
            seq_idx = i * self.beam_width + best_idx
            best_seq = sequences[seq_idx]
            best_sequences.append(best_seq)

        return best_sequences
