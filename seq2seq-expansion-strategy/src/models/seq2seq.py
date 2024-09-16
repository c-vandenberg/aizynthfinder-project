import tensorflow as tf
import tf2onnx
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout


class RetrosynthesisSeq2SeqModel(tf.keras.Model):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, units, *args, **kwargs):
        super(RetrosynthesisSeq2SeqModel, self).__init__(*args, **kwargs)

        self.units = units

        # Save the vocabulary sizes
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        # Sve the embedding dimension
        self.embedding_dim = embedding_dim

        # Define the embedding layers
        self.embedding_encoder = Embedding(input_vocab_size, embedding_dim, mask_zero=True)
        self.embedding_decoder = Embedding(output_vocab_size, embedding_dim, mask_zero=True)

        # Encoder: 2-layer Bidirectional LSTM with dropout rate of 0.2
        self.encoder_layers = []
        for _ in range(2):
            self.encoder_layers.append(
                Bidirectional(
                    LSTM(units, return_sequences=True, return_state=True, dropout=0.2)
                )
            )

        # Decoder: 4-layer LSTM with dropout rate of 0.2
        self.decoder_layers = []
        for _ in range(4):
            self.decoder_layers.append(
                LSTM(units, return_sequences=True, return_state=True, dropout=0.2)
            )

        # Define Attention layers
        self.attention_dense1 = Dense(units)
        self.attention_dense2 = Dense(units)
        self.attention_v = Dense(1)

        # Dense layer to produce output
        self.dense = Dense(output_vocab_size, activation='softmax')

        # Mapping encoder final states to decoder initial states
        self.enc_state_h = Dense(units)
        self.enc_state_c = Dense(units)

    def call(self, inputs, training=None, mask=None):
        # Extract encoder and decoder inputs
        encoder_input, decoder_input = inputs

        # Generate masks (if needed)
        encoder_mask = tf.math.not_equal(encoder_input, 0)
        decoder_mask = tf.math.not_equal(decoder_input, 0)

        # Encoder
        encoder_input_embedded = self.embedding_encoder(encoder_input)
        encoder_output = encoder_input_embedded

        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            encoder_output, forward_h, forward_c, backward_h, backward_c = encoder_layer(encoder_output)

            # Concatenate encoder forward and backward states
            state_h = tf.concat([forward_h, backward_h], axis=-1)  # Shape: (batch_size, units * 2)
            state_c = tf.concat([forward_c, backward_c], axis=-1)

        # Map encoder final states to decoder initial states
        decoder_initial_state_h = self.enc_state_h(state_h)  # Shape: (batch_size, units)
        decoder_initial_state_c = self.enc_state_c(state_c)  # Shape: (batch_size, units)
        decoder_initial_state = [decoder_initial_state_h, decoder_initial_state_c]

        # Decoder
        decoder_input_embedded = self.embedding_decoder(decoder_input)
        decoder_output = decoder_input_embedded

        for idx, decoder_layer in enumerate(self.decoder_layers):
            if idx == 0:
                # Use the mapped encoder states as initial state
                decoder_output, state_h, state_c = decoder_layer(decoder_output, initial_state=decoder_initial_state)
            else:
                decoder_output, state_h, state_c = decoder_layer(decoder_output)

        # Attention Mechanism
        # Calculate attention scores
        # Expand dimensions to match the shapes for broadcasting
        encoder_output_expanded = tf.expand_dims(encoder_output, 1)  # Shape: (batch_size, 1, seq_len_encoder, units)
        decoder_output_expanded = tf.expand_dims(decoder_output, 2)  # Shape: (batch_size, seq_len_decoder, 1, units)

        # Compute the score
        score = tf.nn.tanh(
            self.attention_dense1(encoder_output_expanded) + self.attention_dense2(decoder_output_expanded)
        )  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, units)

        attention_weights = tf.nn.softmax(self.attention_v(score),
                                          axis=2)  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, 1)

        # Compute the context vector
        context_vector = (attention_weights *
                          encoder_output_expanded)  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, units)
        context_vector = tf.reduce_sum(context_vector, axis=2)  # Shape: (batch_size, seq_len_decoder, units)

        # Concatenate attention output
        concat_output = tf.concat([decoder_output, context_vector], axis=-1)

        # Dense layer to produce output
        output = self.dense(concat_output)
        return output

    def train(self, x_train, y_train, epochs=10, batch_size=64):
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.fit([x_train, y_train], y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def save_model(self, save_path):
        """
        Save the model in TensorFlow SavedModel format.
        """
        self.save(save_path)
        print(f"Model saved to {save_path}")

    @staticmethod
    def convert_to_onnx(saved_model_path, onnx_file_path):
        """
        Convert the TensorFlow SavedModel to ONNX format and save it.
        """
        # Load the TensorFlow model
        model = tf.keras.models.load_model(saved_model_path)

        # Convert the TensorFlow model to ONNX format
        onnx_model = tf2onnx.convert.from_keras(model)

        # Save the ONNX model to a file
        with open(onnx_file_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX model saved to {onnx_file_path}")
