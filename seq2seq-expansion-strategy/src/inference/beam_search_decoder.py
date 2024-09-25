import numpy as np
import tensorflow as tf
from inference.inference_decoder_interface import InferenceDecoderInterface

class BeamSearchDecoder(InferenceDecoderInterface):
    def __init__(self, model, tokenizer, beam_width=5, max_seq_len=100):
        """
        Initializes the BeamSearchDecoder.

        Args:
            model: The trained seq2seq model.
            tokenizer: The tokenizer used for encoding and decoding.
            beam_width: The number of beams to keep during decoding.
            max_seq_len: The maximum length of the decoded sequences.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.max_seq_len = max_seq_len
        self.start_token_index = tokenizer.word_index['<START>']
        self.end_token_index = tokenizer.word_index['<END>']
        self.vocab_size = len(tokenizer.word_index) + 1  # +1 for padding

    def decode(self, input_sequence):
        """
        Performs beam search decoding on the input sequence.

        Args:
            input_sequence: The input sequence (encoder input).

        Returns:
            List of tuples: Each tuple contains a decoded sequence and its score.
        """
        # Encode the input sequence to get the encoder output and states
        encoder_input = tf.expand_dims(input_sequence, axis=0)  # Add batch dimension
        encoder_output, state_h, state_c = self.model.encoder(encoder_input)

        # Prepare initial decoder input (start token) and states
        decoder_input = tf.expand_dims([self.start_token_index], 0)  # Shape: (1, 1)
        decoder_state_h = self.model.enc_state_h(state_h)
        decoder_state_c = self.model.enc_state_c(state_c)

        # Initialize states for all layers
        zero_state = tf.zeros_like(decoder_state_h)
        decoder_states = [decoder_state_h, decoder_state_c, zero_state, zero_state, zero_state, zero_state, zero_state,
                          zero_state]

        # Initialize the beam with the start token sequence
        sequences = [([self.start_token_index], 0.0, decoder_states)]  # (sequence, score, states)

        for _ in range(self.max_seq_len):
            all_candidates = []
            for seq, score, states in sequences:
                if seq[-1] == self.end_token_index:
                    # If the last token is the end token, add the sequence as is
                    all_candidates.append((seq, score, states))
                    continue

                # Prepare decoder input
                decoder_input = tf.expand_dims([seq[-1]], 0)  # Shape: (1, 1)
                decoder_input = tf.cast(decoder_input, tf.int32)

                # Run the decoder for one step
                decoder_output, decoder_states = self.model.decoder.single_step(
                    decoder_input, states, encoder_output
                )

                # Get the log probabilities and top candidates
                log_probs = tf.math.log(decoder_output[0, -1] + 1e-9).numpy()  # Shape: (vocab_size,)

                # Get the top beam_width candidates
                top_k_indices = np.argsort(log_probs)[-self.beam_width:]

                for idx in top_k_indices:
                    candidate_seq = seq + [idx]
                    candidate_score = score + log_probs[idx]
                    candidate_states = decoder_states
                    all_candidates.append((candidate_seq, candidate_score, candidate_states))

            # Select the top beam_width sequences
            sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:self.beam_width]

            # Check if all sequences have reached the end token
            if all(seq[-1] == self.end_token_index for seq, _, _ in sequences):
                break

        # Return the sequences sorted by score
        final_sequences = sorted(sequences, key=lambda tup: tup[1], reverse=True)
        return final_sequences