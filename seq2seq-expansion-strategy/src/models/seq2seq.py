import tensorflow as tf
import tf2onnx
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout


class RetrosynthesisSeq2SeqModel(tf.keras.Model):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, units, dropout_rate=0.2, *args, **kwargs):
        super(RetrosynthesisSeq2SeqModel, self).__init__(*args, **kwargs)

        self.units = units

        # Save the vocabulary sizes
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        # Sve the embedding dimension
        self.embedding_dim = embedding_dim

        self.dropout_rate = dropout_rate

        # Define the embedding layers
        self.embedding_encoder = Embedding(input_vocab_size, embedding_dim, mask_zero=True)
        self.embedding_decoder = Embedding(output_vocab_size, embedding_dim, mask_zero=True)

        # Encoder: 2-layer Bidirectional LSTM without internal Dropout
        self.encoder_layers = tf.keras.Sequential([
            Bidirectional(LSTM(units, return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0),
                          name='bidirectional_lstm_1'),
            Dropout(dropout_rate, name='encoder_dropout_1'),
            Bidirectional(LSTM(units, return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0),
                          name='bidirectional_lstm_2'),
            Dropout(dropout_rate, name='encoder_dropout_2')
        ], name='encoder_layers')

        # Decoder: 4-layer LSTM without internal Dropout
        self.decoder_layers = tf.keras.Sequential([
            LSTM(units, return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0,
                 name='lstm_decoder_1'),
            Dropout(dropout_rate, name='decoder_dropout_1'),
            LSTM(units, return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0,
                 name='lstm_decoder_2'),
            Dropout(dropout_rate, name='decoder_dropout_2'),
            LSTM(units, return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0,
                 name='lstm_decoder_3'),
            Dropout(dropout_rate, name='decoder_dropout_3'),
            LSTM(units, return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0,
                 name='lstm_decoder_4'),
            Dropout(dropout_rate, name='decoder_dropout_4')
        ], name='decoder_layers')

        # Define Attention layers
        self.attention_dense1 = Dense(units, name='attention_dense1')
        self.attention_dense2 = Dense(units, name='attention_dense2')
        self.attention_v = Dense(1, name='attention_v')

        # Dense layer to produce output
        self.dense = Dense(output_vocab_size, activation='softmax')

        # Mapping encoder final states to decoder initial states
        self.enc_state_h = Dense(units, name='enc_state_h')
        self.enc_state_c = Dense(units, name='enc_state_c')

    def build(self, input_shape):
        # Define the input shapes for encoder and decoder
        encoder_input_shape, decoder_input_shape = input_shape

        # Pass a dummy input through encoder and decoder to initialize weights
        encoder_dummy = tf.zeros(encoder_input_shape)
        decoder_dummy = tf.zeros(decoder_input_shape)

        # Forward pass to build the model
        self.call((encoder_dummy, decoder_dummy), training=False)

        # Mark the model as built
        super(RetrosynthesisSeq2SeqModel, self).build(input_shape)

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
        # Since encoder_layers is a Sequential model, it will process both Bidirectional LSTMs sequentially
        # However, to extract states, we need to manually iterate through them
        # Therefore, it's better to separate the layers or adjust accordingly
        # Alternatively, you can define the encoder_layers as a list of layers inside a custom Layer
        # For clarity, we'll process layers one by one
        states_h = []
        states_c = []
        for layer in self.encoder_layers.layers:
            if isinstance(layer, Bidirectional):
                # For Bidirectional layers, get forward and backward states
                encoder_output, forward_h, forward_c, backward_h, backward_c = layer(encoder_output, training=training)
                # Concatenate forward and backward states
                state_h = tf.concat([forward_h, backward_h], axis=-1)  # Shape: (batch_size, units * 2)
                state_c = tf.concat([forward_c, backward_c], axis=-1)
            elif isinstance(layer, Dropout):
                # Apply Dropout
                encoder_output = layer(encoder_output, training=training)
                continue
            else:
                encoder_output, state_h, state_c = layer(encoder_output, training=training)
            states_h.append(state_h)
            states_c.append(state_c)

        # Use the last layer's states
        state_h = states_h[-1]
        state_c = states_c[-1]

        # Map encoder final states to decoder initial states
        decoder_initial_state_h = self.enc_state_h(state_h)  # Shape: (batch_size, units)
        decoder_initial_state_c = self.enc_state_c(state_c)  # Shape: (batch_size, units)
        decoder_initial_state = [decoder_initial_state_h, decoder_initial_state_c]

        # Decoder
        decoder_input_embedded = self.embedding_decoder(decoder_input)
        decoder_output = decoder_input_embedded

        for layer in self.decoder_layers.layers:
            if isinstance(layer, LSTM):
                if layer.name == 'lstm_decoder_1':
                    # Use the mapped encoder states as initial state for the first decoder LSTM layer
                    decoder_output, state_h, state_c = layer(decoder_output, initial_state=decoder_initial_state,
                                                             training=training)
                else:
                    decoder_output, state_h, state_c = layer(decoder_output, training=training)
            elif isinstance(layer, Dropout):
                # Apply Dropout
                decoder_output = layer(decoder_output, training=training)

        # Attention Mechanism
        # Calculate attention scores
        # Expand dimensions to match the shapes for broadcasting
        encoder_output_expanded = tf.expand_dims(encoder_output,
                                                 1)  # Shape: (batch_size, 1, seq_len_encoder, units*2)
        decoder_output_expanded = tf.expand_dims(decoder_output,
                                                 2)  # Shape: (batch_size, seq_len_decoder, 1, units)

        # Adjust attention_dense layers to handle the dimensions
        # Since encoder_output has units*2, you might need to adjust attention_dense1 accordingly
        # For simplicity, let's assume units*2 is compatible with attention_dense1

        # Compute the score
        score = tf.nn.tanh(
            self.attention_dense1(encoder_output_expanded) + self.attention_dense2(decoder_output_expanded)
        )  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, units)

        attention_weights = tf.nn.softmax(self.attention_v(score),
                                          axis=2)  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, 1)

        # Compute the context vector
        context_vector = attention_weights * encoder_output_expanded  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, units*2)
        context_vector = tf.reduce_sum(context_vector, axis=2)  # Shape: (batch_size, seq_len_decoder, units*2)

        # Concatenate attention output
        concat_output = tf.concat([decoder_output, context_vector],
                                  axis=-1)  # Shape: (batch_size, seq_len_decoder, units + units*2)

        # Dense layer to produce output
        output = self.dense(concat_output)  # Shape: (batch_size, seq_len_decoder, output_vocab_size)

        return output

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


class CustomCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager):
        super(CustomCheckpointCallback, self).__init__()
        self.checkpoint_manager = checkpoint_manager
        self.best_val_loss = float('inf')  # Initialize with infinity

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                save_path = self.checkpoint_manager.save()
                print(f"\nEpoch {epoch+1}: Validation loss improved to {current_val_loss:.4f}. Saving checkpoint to {save_path}")
