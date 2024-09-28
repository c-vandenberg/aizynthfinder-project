#!/usr/bin/env python3

import numpy as np
from models.seq2seq import RetrosynthesisSeq2SeqModel
from tensorflow.keras.optimizers import Adam


def main():
    input_vocab_size = 1000
    output_vocab_size = 1000
    encoder_embedding_dim = 32
    decoder_embedding_dim = 64
    units = 128
    dropout_rate = 0.2

    model = RetrosynthesisSeq2SeqModel(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        encoder_embedding_dim=encoder_embedding_dim,
        decoder_embedding_dim=decoder_embedding_dim,
        units=units,
        dropout_rate=dropout_rate
    )

    encoder_input_shape = (1, 20)  # (batch_size, sequence_length)
    decoder_input_shape = (1, 20)  # (batch_size, sequence_length)

    model.build([encoder_input_shape, decoder_input_shape])

    sample_encoder_input = np.random.randint(0, input_vocab_size, size=(1, 20))
    sample_decoder_input = np.random.randint(0, output_vocab_size, size=(1, 20))

    learning_rate: float = 0.0001
    optimizer: Adam = Adam(learning_rate=learning_rate, clipnorm=5.0)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    output = model([sample_encoder_input, sample_decoder_input])
    print("Model output shape:", output.shape)

    model.save('minimal_seq2seq_model.keras')
    print("Model saved successfully.")


if __name__ == '__main__':
    main()
