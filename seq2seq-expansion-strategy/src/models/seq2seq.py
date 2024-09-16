import tensorflow as tf
import tf2onnx
from tensorflow.keras.layers import Input


class RetrosynthesisSeq2SeqModel(tf.keras.Model):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, units, *args, **kwargs):
        super(RetrosynthesisSeq2SeqModel, self).__init__(*args, **kwargs)

        self.units = units

        # Save the vocabulary size for the Dense layer
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim

        # Define the layers
        self.embedding_encoder = tf.keras.layers.Embedding(input_vocab_size, embedding_dim, mask_zero=True)
        self.embedding_decoder = tf.keras.layers.Embedding(output_vocab_size, embedding_dim, mask_zero=True)
        self.encoder = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)

        # Attention layers
        self.attention_dense1 = tf.keras.layers.Dense(units)
        self.attention_dense2 = tf.keras.layers.Dense(units)
        self.attention_v = tf.keras.layers.Dense(1)

        self.dense = tf.keras.layers.Dense(output_vocab_size, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        encoder_input, decoder_input = inputs

        # Encoder
        encoder_input_embedded = self.embedding_encoder(encoder_input)
        encoder_output, encoder_state_h, encoder_state_c = self.encoder(encoder_input_embedded)

        # Decoder
        decoder_input_embedded = self.embedding_decoder(decoder_input)
        decoder_output, _, _ = self.decoder(decoder_input_embedded, initial_state=[encoder_state_h, encoder_state_c])

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
        context_vector = attention_weights * encoder_output_expanded  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, units)
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
