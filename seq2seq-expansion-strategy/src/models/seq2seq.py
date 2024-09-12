import tensorflow as tf
import tf2onnx
from tensorflow.keras.layers import Input


class Seq2SeqModel(tf.keras.Model):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, units, *args, **kwargs):
        super(Seq2SeqModel, self).__init__(*args, **kwargs)

        # Save the vocabulary size for the Dense layer
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim

        # Define the layers
        self.embedding_encoder = tf.keras.layers.Embedding(input_vocab_size, embedding_dim)
        self.embedding_decoder = tf.keras.layers.Embedding(output_vocab_size, embedding_dim)
        self.encoder = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(output_vocab_size, activation='softmax')

    def call(self, inputs):
        encoder_input, decoder_input = inputs

        # Apply embeddings
        encoder_input_embedded = self.embedding_encoder(encoder_input)
        decoder_input_embedded = self.embedding_decoder(decoder_input)

        # Encoder
        encoder_output, encoder_state_h, encoder_state_c = self.encoder(encoder_input_embedded)

        # Decoder
        decoder_output, _, _ = self.decoder(decoder_input_embedded, initial_state=[encoder_state_h, encoder_state_c])

        # Dense layer to produce output
        outputs = self.dense(decoder_output)
        return outputs

    def train(self, x_train, y_train, epochs=10):
        y_train = tf.expand_dims(y_train, -1)  # Add an extra dimension for the target variable
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.fit([x_train, y_train], y_train, epochs=epochs, validation_split=0.2)

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
