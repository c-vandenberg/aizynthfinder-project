import tensorflow as tf
import tf2onnx
from tensorflow.keras.layers import Input


class Seq2SeqModel(tf.keras.Model):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, units, *args, **kwargs):
        super(Seq2SeqModel, self).__init__(*args, **kwargs)

        # Save the vocabulary size for the Dense layer
        self.output_vocab_size = output_vocab_size

        # Define the layers
        self.encoder = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(output_vocab_size,
                                           activation='softmax')

    def call(self, inputs):
        encoder_input, decoder_input = inputs

        # Encoder
        encoder_output, encoder_state_h, encoder_state_c = self.encoder(encoder_input)

        # Decoder
        decoder_output, _, _ = self.decoder(decoder_input, initial_state=[encoder_state_h, encoder_state_c])

        # Dense layer to produce output
        outputs = self.dense(decoder_output)
        return outputs

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
