from inference.beam_search_decoder import BeamSearchDecoder
import tensorflow as tf


class InferenceModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict_with_beam_search(self, input_smiles, beam_width=5, max_seq_len=100):
        """
        Predicts the reactant SMILES given a product SMILES using beam search.

        Args:
            input_smiles: The input product SMILES string.
            self.model: The trained seq2seq model.
            self.tokenizer: The tokenizer used for encoding and decoding.
            beam_width: The number of beams to keep during decoding.
            max_seq_len: The maximum length of the decoded sequences.

        Returns:
            List of tuples: Each tuple contains a decoded SMILES and its score.
        """
        # Tokenize and preprocess the input SMILES
        input_tokens = self.tokenizer.tokenize(input_smiles)
        input_sequence = self.tokenizer.texts_to_sequences([input_tokens])[0]
        input_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [input_sequence],
            maxlen=self.model.encoder_data_processor.max_seq_length,
            padding='post'
        )[0]

        # Initialize the BeamSearchDecoder
        beam_search_decoder = BeamSearchDecoder(self.model, self.tokenizer, beam_width, max_seq_len)

        # Perform beam search decoding
        final_sequences = beam_search_decoder.decode(input_sequence)

        # Convert sequences of token indices back to SMILES strings
        decoded_smiles = []
        for seq, score, _ in final_sequences:
            # Remove start and end tokens
            token_indices = seq[1:]  # Remove start token
            if self.tokenizer.word_index['<END>'] in token_indices:
                end_idx = token_indices.index(self.tokenizer.word_index['<END>'])
                token_indices = token_indices[:end_idx]

            tokens = self.tokenizer.sequences_to_texts([token_indices])[0].split()
            smiles = ''.join(tokens)
            decoded_smiles.append((smiles, score))

        return decoded_smiles