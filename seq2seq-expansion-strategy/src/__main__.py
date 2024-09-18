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

    # Paths to your data files
    products_file = '../data/processed/pande-et-al/products_smiles'
    reactants_file = '../data/processed/pande-et-al/reactants_smiles'
    products_valid_file = '../data/processed/pande-et-al/validation_products_smiles'
    reactants_valid_file = '../data/processed/pande-et-al/validation_reactants_smiles'

    # Initialize DataLoader
    data_loader = DataLoader(
        products_file=products_file,
        reactants_file=reactants_file,
        products_valid_file=products_valid_file,
        reactants_valid_file=reactants_valid_file,
        num_samples=100,
        max_encoder_seq_length=150,
        max_decoder_seq_length=150,
        batch_size=16,
        test_size=0.3,
        random_state=42
    )

    # Load and prepare data
    data_loader.load_and_prepare_data()

    # Access the tokenizer and vocab size
    tokenizer = data_loader.tokenizer
    vocab_size = data_loader.vocab_size

    # Save the tokenizer for future use
    tokenizer_path = '../data/tokenizers/pande_et_al_tokenizer.json'
    with open(tokenizer_path, 'w') as f:
        f.write(tokenizer.to_json())

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

    build_seq2seq_model(model, data_loader)

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

    # Get datasets
    train_dataset = data_loader.get_train_dataset()
    valid_dataset = data_loader.get_valid_dataset()
    test_dataset = data_loader.get_test_dataset()

    # 10. Train the Model
    model.fit(
        train_dataset,
        epochs=1,
        validation_data=valid_dataset,
        callbacks=[early_stopping, checkpoint, tensorboard_callback]
    )

    # 11. Save the Trained Model
    model_save_path = '../data/training/pande-et-al/models/init_hyperparameters'
    tf.saved_model.save(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # 12. Evaluate the model
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")


def build_seq2seq_model(model, data_loader):
    print("Building the model with sample data to initialize variables...")
    # Get a batch from the training dataset
    for batch in data_loader.get_train_dataset().take(1):
        (sample_encoder_input, sample_decoder_input), _ = batch
        model([sample_encoder_input, sample_decoder_input])
        break
    print("Model built successfully.\n")


if __name__ == '__main__':
    main()
