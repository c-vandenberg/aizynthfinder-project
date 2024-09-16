#!/usr/bin/env python3

import sys
import os
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from aizynthfinder.aizynthfinder import AiZynthFinder
from models.seq2seq import RetrosynthesisSeq2SeqModel
from models.utils import (
    smiles_tokenizer,
    preprocess_smiles,
    create_smiles_tokenizer,
    masked_sparse_categorical_crossentropy
)


def main():
    # 1. Load data
    with open('../data/processed/pande_et_al/products_smiles', 'r') as file:
        products_x_dataset = [line.strip() for line in file.readlines()]

    with open('../data/processed/pande_et_al/reactants_smiles', 'r') as file:
        reactants_y_dataset = [line.strip() for line in file.readlines()]

    with open('../data/processed/pande_et_al/validation_products_smiles', 'r') as file:
        products_x_valid_dataset = [line.strip() for line in file.readlines()]

    with open('../data/processed/pande_et_al/validation_reactants_smiles', 'r') as file:
        reactants_y_valid_dataset = [line.strip() for line in file.readlines()]

    # Ensure that the datasets have the same length
    assert len(products_x_dataset) == len(reactants_y_dataset), "Mismatch in dataset lengths."
    assert len(products_x_valid_dataset) == len(reactants_y_valid_dataset), "Mismatch in validation dataset lengths."

    # 2. Create the Tokenizer
    # Combine all SMILES strings to build a common tokenizer
    all_smiles = (products_x_dataset + reactants_y_dataset +
                  products_x_valid_dataset + reactants_y_valid_dataset)
    tokenizer = create_smiles_tokenizer(all_smiles)

    # Save the tokenizer for future use
    tokenizer_path = '../data/tokenizers/pande_et_al_data_tokenizer.json'
    with open(tokenizer_path, 'w') as f:
        f.write(tokenizer.to_json())

    # 3. Split data into training and test data sets
    (seq2seq_model_x_train_data, seq2seq_model_x_test_data,
     seq2seq_model_y_train_data, seq2seq_model_y_test_data) = train_test_split(
        products_x_dataset,
        reactants_y_dataset,
        test_size=0.3,  # 30% of the data is reserved for testing, and 70% is used for training
        random_state=42  # Random number generator seed
    )

    # 4. Determine Maximum Sequence Lengths
    # Tokenize to find the maximum lengths
    max_encoder_seq_length = max(len(smiles_tokenizer(smiles)) for smiles in seq2seq_model_x_train_data)
    max_decoder_seq_length = max(len(smiles_tokenizer(smiles)) for smiles in seq2seq_model_y_train_data)

    # Add buffer to max lengths
    max_encoder_seq_length += 10
    max_decoder_seq_length += 10

    # 5. Preprocess the SMILES Data
    # Preprocess encoder input training data
    encoder_input_data = preprocess_smiles(seq2seq_model_x_train_data, tokenizer, max_encoder_seq_length)

    # Preprocess encoder input validation data
    encoder_input_valid = preprocess_smiles(products_x_valid_dataset, tokenizer, max_encoder_seq_length)

    # Preprocess decoder input and target training data
    decoder_input_data_full = preprocess_smiles(seq2seq_model_y_train_data, tokenizer, max_decoder_seq_length)

    # Preprocess decoder input and target validation data
    decoder_input_valid_full = preprocess_smiles(reactants_y_valid_dataset, tokenizer, max_decoder_seq_length)

    # Prepare decoder input and target training data by shifting
    decoder_input_data = decoder_input_data_full[:, :-1]
    decoder_target_data = decoder_input_data_full[:, 1:]

    # Prepare decoder input and target validation data by shifting
    decoder_input_valid = decoder_input_valid_full[:, :-1]
    decoder_target_valid = decoder_input_valid_full[:, 1:]

    # 6. Get Vocabulary Size (+1 for padding token)
    vocab_size = len(tokenizer.word_index) + 1

    # 7. Initialize the Model
    model = RetrosynthesisSeq2SeqModel(
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
        embedding_dim=64,
        units=128
    )

    # 8. Compile the Model with the Custom Loss Function
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=masked_sparse_categorical_crossentropy, metrics=['accuracy'])

    # 9. Define callbacks for early stopping and checkpointing
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True
    )

    # 10. Train the Model
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=32,
        epochs=1,
        validation_data=(
            [encoder_input_valid, decoder_input_valid],
            decoder_target_valid
        ),
        callbacks=[early_stopping, checkpoint]
    )

    # 11. Save the Trained Model
    model_save_path = '../data/models/pande_et_al/init_hyperparameters'
    tf.saved_model.save(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # 12. Evaluate the model
    # Preprocess test data
    encoder_input_test = preprocess_smiles(seq2seq_model_x_test_data, tokenizer, max_encoder_seq_length)
    decoder_input_test_full = preprocess_smiles(seq2seq_model_y_test_data, tokenizer, max_decoder_seq_length)
    decoder_input_test = decoder_input_test_full[:, :-1]
    decoder_target_test = decoder_input_test_full[:, 1:]

    test_loss, test_accuracy = model.evaluate(
        [encoder_input_test, decoder_input_test],
        decoder_target_test
    )
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")


def remove_rx_pattern(line):
    return re.sub(r'<RX_\d+>', '', line).strip()


if __name__ == '__main__':
    main()
