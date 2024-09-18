#!/usr/bin/env python3

import random
import os
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from aizynthfinder.aizynthfinder import AiZynthFinder
from models.seq2seq import RetrosynthesisSeq2SeqModel, CustomCheckpointCallback
from models.utils import Seq2SeqModelUtils
from data.utils.data_loader import DataLoader
from data.utils.preprocessing import DataPreprocessor
from data.utils.tokenization import SmilesTokenizer

random.seed(42)
tf.random.set_seed(42)


def main():
    smiles_tokenizer = SmilesTokenizer()

    # 1. Load data
    with open('../data/processed/pande-et-al/products_smiles', 'r') as file:
        products_x_dataset = [line.strip() for line in file.readlines()]

    with open('../data/processed/pande-et-al/reactants_smiles', 'r') as file:
        reactants_y_dataset = [line.strip() for line in file.readlines()]

    with open('../data/processed/pande-et-al/validation_products_smiles', 'r') as file:
        products_x_valid_dataset = [line.strip() for line in file.readlines()]

    with open('../data/processed/pande-et-al/validation_reactants_smiles', 'r') as file:
        reactants_y_valid_dataset = [line.strip() for line in file.readlines()]

    # Ensure that the datasets have the same length
    assert len(products_x_dataset) == len(reactants_y_dataset), "Mismatch in dataset lengths."
    assert len(products_x_valid_dataset) == len(reactants_y_valid_dataset), "Mismatch in validation dataset lengths."

    num_samples = 100

    # 2. Create the Tokenizer
    # Tokenize the datasets
    tokenized_products_x_dataset = smiles_tokenizer.tokenize_list(products_x_dataset)
    tokenized_products_x_dataset = tokenized_products_x_dataset[:num_samples]

    tokenized_reactants_y_dataset = smiles_tokenizer.tokenize_list(reactants_y_dataset)
    tokenized_reactants_y_dataset = tokenized_reactants_y_dataset[:num_samples]

    tokenized_products_x_valid_dataset = smiles_tokenizer.tokenize_list(products_x_valid_dataset)
    tokenized_products_x_valid_dataset = tokenized_products_x_valid_dataset[:num_samples]

    tokenized_reactants_y_valid_dataset = smiles_tokenizer.tokenize_list(reactants_y_valid_dataset)
    tokenized_reactants_y_valid_dataset = tokenized_reactants_y_valid_dataset[:num_samples]

    # Combine all tokenized SMILES strings to build a common tokenizer
    all_tokenized_smiles = (tokenized_products_x_dataset + tokenized_reactants_y_dataset +
                            tokenized_products_x_valid_dataset + tokenized_reactants_y_valid_dataset)

    tokenizer = smiles_tokenizer.create_tokenizer(all_tokenized_smiles)

    # Save the tokenizer for future use
    tokenizer_path = '../data/tokenizers/pande_et_al_tokenizer.json'
    with open(tokenizer_path, 'w') as f:
        f.write(tokenizer.to_json())

    # 3. Split data into training and test data sets
    (tokenized_products_x_train_data, tokenized_products_x_test_data,
     tokenized_reactants_y_train_data, tokenized_reactants_y_test_data) = train_test_split(
        tokenized_products_x_dataset,
        tokenized_reactants_y_dataset,
        test_size=0.3,  # 30% of the data is reserved for testing, and 70% is used for training
        random_state=42  # Random number generator seed
    )

    # 4. Determine Maximum Sequence Lengths
    # Tokenize to find the maximum lengths
    max_encoder_seq_length = 140
    max_decoder_seq_length = 140

    # Add buffer to max lengths
    max_encoder_seq_length += 10
    max_decoder_seq_length += 10

    # 5. Preprocess the SMILES Data
    # Preprocess encoder input training data
    encoder_data_processor = DataPreprocessor(tokenizer, max_encoder_seq_length)
    decoder_data_processor = DataPreprocessor(tokenizer, max_decoder_seq_length)

    encoder_input_train_data = encoder_data_processor.preprocess_smiles(tokenized_products_x_train_data)

    # Preprocess encoder input validation data
    encoder_input_valid_data = encoder_data_processor.preprocess_smiles(tokenized_products_x_valid_dataset)

    # Preprocess encoder input testing data
    encoder_input_test_data = encoder_data_processor.preprocess_smiles(tokenized_products_x_test_data)

    # Preprocess decoder full training data (input and target training data)
    decoder_full_train_data = encoder_data_processor.preprocess_smiles(tokenized_reactants_y_train_data)

    # Preprocess decoder full validation data (input and target validation data)
    decoder_full_valid_data = encoder_data_processor.preprocess_smiles(tokenized_reactants_y_valid_dataset)

    # Preprocess decoder full testing data (input and target training data)
    decoder_full_test_data = encoder_data_processor.preprocess_smiles(tokenized_reactants_y_test_data)

    # Prepare decoder input and target training data by shifting
    decoder_input_train_data = decoder_full_train_data[:, :-1]
    decoder_target_train_data = decoder_full_train_data[:, 1:]

    # Prepare decoder input and target validation data by shifting
    decoder_input_valid_data = decoder_full_valid_data[:, :-1]
    decoder_target_valid_data = decoder_full_valid_data[:, 1:]

    # 6. Get Vocabulary Size (+1 for padding token)
    vocab_size = len(tokenizer.word_index) + 1

    # 7. Initialize the Model
    model = RetrosynthesisSeq2SeqModel(
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
        embedding_dim=256,
        units=256
    )

    # 8. Compile the Model with the Custom Loss Function
    seq2seq_model_utils = Seq2SeqModelUtils()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=5.0)
    model.compile(optimizer=optimizer, loss=seq2seq_model_utils.masked_sparse_categorical_crossentropy, metrics=['accuracy'])

    build_seq2seq_model(model, encoder_input_train_data, decoder_input_train_data)

    # 9. Define callbacks for early stopping, checkpointing and visualisation of training and validation metrics
    # over epochs.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../logs/pande-et-al')

    # 9.1 Initialize TensorFlow Checkpoint and CheckpointManager
    # Use `tf.train.Checkpoint` to manage saving and loading
    # Use `tf.train.CheckpointManager` to manage multiple checkpoints
    checkpoint_dir = '../data/training/pande-et-al/checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=checkpoint_dir,
        max_to_keep=5  # Keeps the latest 5 checkpoints
    )

    # Restore the latest checkpoint if it exists
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(f"Restored from {checkpoint_manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    checkpoint = CustomCheckpointCallback(checkpoint_manager)

    # 10. Train the Model
    model.fit(
        [encoder_input_train_data, decoder_input_train_data],
        decoder_target_train_data,
        batch_size=16,
        epochs=1,
        validation_data=(
            [encoder_input_valid_data, decoder_input_valid_data],
            decoder_target_valid_data
        ),
        callbacks=[early_stopping, checkpoint, tensorboard_callback]
    )

    # 11. Save the Trained Model
    model_save_path = '../data/training/pande-et-al/models/init_hyperparameters'
    tf.saved_model.save(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # 12. Evaluate the model

    decoder_input_test_data = decoder_full_test_data[:, :-1]
    decoder_target_test_data = decoder_full_test_data[:, 1:]

    test_loss, test_accuracy = model.evaluate(
        [encoder_input_test_data, decoder_input_test_data],
        decoder_target_test_data
    )
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")


def build_seq2seq_model(model, encoder_input_data, decoder_input_data):
    print("Building the model with sample data to initialize variables...")
    sample_encoder_input = tf.constant(encoder_input_data[:1])  # (1, max_encoder_seq_length)
    sample_decoder_input = tf.constant(decoder_input_data[:1])  # (1, max_decoder_seq_length - 1)
    model([sample_encoder_input, sample_decoder_input])
    print("Model built successfully.\n")


if __name__ == '__main__':
    main()
