import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout
from models.interfaces import EncoderInterface


class RetrosynthesisEncoder(tf.keras.layers.Layer, EncoderInterface):
    def __init__(self, vocab_size, embedding_dim, units, dropout_rate=0.2):
        super(RetrosynthesisEncoder, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)

        # Encoder: 2-layer Bidirectional LSTM without internal Dropout
        self.encoder_layers = tf.keras.Sequential([
            Bidirectional(LSTM(units, return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0),
                          name='bidirectional_lstm_1'),
            Dropout(dropout_rate, name='encoder_dropout_1'),
            Bidirectional(LSTM(units, return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0),
                          name='bidirectional_lstm_2'),
            Dropout(dropout_rate, name='encoder_dropout_2')
        ])

    def call(self, encoder_input, training=False):
        # Embed the input
        encoder_output = self.embedding(encoder_input)
        states_h = []
        states_c = []

        # Process through encoder layers
        # Since encoder_layers is a Sequential model, it will process both Bidirectional LSTMs sequentially
        # However, to extract states, we need to manually iterate through them
        # Therefore, it's better to separate the layers or adjust accordingly
        # Alternatively, you can define the encoder_layers as a list of layers inside a custom Layer
        # For clarity, we'll process layers one by one
        for layer in self.encoder_layers.layers:
            if isinstance(layer, Bidirectional):
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

        # Use the last layer's states for final states
        final_state_h = states_h[-1]
        final_state_c = states_c[-1]

        return encoder_output, final_state_h, final_state_c
