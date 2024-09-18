import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from models.interfaces import DecoderInterface
from models.attention import RetrosynthesisAttention


class RetrosynthesisDecoder(tf.keras.layers.Layer, DecoderInterface):
    def __init__(self, vocab_size, embedding_dim, units, dropout_rate=0.2):
        super(RetrosynthesisDecoder, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)

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
        ])

        # Attention Mechanism
        self.attention = RetrosynthesisAttention(units)

        # Output layer
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, inputs, training=False, **kwargs):
        # Extract initial state and encoder output from inputs
        decoder_input, initial_state, encoder_output = inputs

        if decoder_input is None or initial_state is None or encoder_output is None:
            raise ValueError('decoder_input, initial_state and encoder_output must be provided to the Decoder.')

        # Embed the input
        decoder_output = self.embedding(decoder_input)

        # Process through decoder layers
        for layer in self.decoder_layers.layers:
            if isinstance(layer, LSTM):
                if layer.name == 'lstm_decoder_1':
                    # Use the mapped encoder states as initial state for the first decoder LSTM layer
                    decoder_output, state_h, state_c = layer(decoder_output, initial_state=initial_state,
                                                             training=training)
                else:
                    decoder_output, state_h, state_c = layer(decoder_output, training=training)
            elif isinstance(layer, Dropout):
                # Apply Dropout
                decoder_output = layer(decoder_output, training=training)

        # Apply attention
        context_vector, attention_weights = self.attention([encoder_output, decoder_output])

        # Concatenate decoder outputs and context vector
        concat_output = tf.concat([decoder_output, context_vector], axis=-1)  # (batch_size, seq_len_dec, units + units_enc)

        # Generate outputs
        decoder_output = self.dense(concat_output)  # (batch_size, seq_len_dec, vocab_size)

        return decoder_output
