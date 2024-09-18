import tensorflow as tf
from tensorflow.keras.layers import Dense
from models.interfaces import AttentionInterface


class RetrosynthesisAttention(tf.keras.layers.Layer, AttentionInterface):
    def __init__(self, units):
        super(RetrosynthesisAttention, self).__init__()
        self.units = units
        self.attention_dense1 = Dense(units, name='attention_dense1')
        self.attention_dense2 = Dense(units, name='attention_dense2')
        self.attention_v = Dense(1, name='attention_v')

    def call(self, outputs):
        encoder_output, decoder_output = outputs

        # Attention Mechanism
        # Calculate attention scores
        # Expand dimensions to match the shapes for broadcasting
        encoder_output_expanded = tf.expand_dims(encoder_output,
                                                 1)  # Shape: (batch_size, 1, seq_len_encoder, units*2)
        decoder_output_expanded = tf.expand_dims(decoder_output,
                                                 2)  # Shape: (batch_size, seq_len_decoder, 1, units)

        # Compute the attention scores
        score = tf.nn.tanh(
            self.attention_dense1(encoder_output_expanded) + self.attention_dense2(decoder_output_expanded)
        )  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, units)

        attention_weights = tf.nn.softmax(self.attention_v(score),
                                          axis=2)  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, 1)

        # Compute the context vector
        context_vector = attention_weights * encoder_output_expanded  # Shape: (batch_size, seq_len_decoder, seq_len_encoder, units*2)
        context_vector = tf.reduce_sum(context_vector, axis=2)  # Shape: (batch_size, seq_len_decoder, units*2)

        return context_vector, attention_weights
