import re
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit, cross_validate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List


def load_smiles_data_from_csv(csv_file, smiles_column='Smiles'):
    df = pd.read_csv(csv_file)

    smiles_col = smiles_column.lower()

    if smiles_column.lower() not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in the CSV file.")

    # Drop NaN values and convert to a list
    smiles_list = df[smiles_column].dropna().tolist()

    return smiles_list


def smiles_tokenizer(smiles):
    pattern = r"(\[.*?\])|Cl?|Br?|Si|@@?|==?|[B-NOP-Zb-nop-z0-9]|\S"
    tokens = re.findall(pattern, smiles)
    tokens = ['<START>'] + tokens + ['<END>']
    return tokens


def preprocess_smiles(smiles_list, tokenizer, max_length):
    # Tokenize SMILES strings
    tokenized_smiles = [smiles_tokenizer(smiles) for smiles in smiles_list]

    # Convert tokens to sequences of integers
    sequences = tokenizer.texts_to_sequences(tokenized_smiles)

    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences


def create_smiles_tokenizer(smiles_list):
    # Tokenize each SMILES string in dataset
    tokenized_smiles = [smiles_tokenizer(smiles) for smiles in smiles_list]

    # Flatten the tokens list for fitting the tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False, oov_token='<OOV>')
    tokenizer.fit_on_texts(tokenized_smiles)
    return tokenizer


def seq2seq_cross_validator(n_splits: int, test_size: float, random_state: int, seq2seq_model, feature_matrix, target_matrix):
    cross_validator: ShuffleSplit = ShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state
    )

    scoring_metrics: List[str] = ['r2', 'neg_mean_absolute_error']

    return cross_validate(
        seq2seq_model,
        feature_matrix,
        target_matrix,
        scoring=scoring_metrics,
        cv=cross_validator
    )


def masked_sparse_categorical_crossentropy(real, pred):
    # Create a mask to ignore padding tokens (assumed to be 0)
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    # Define and instantiate the loss object
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    # Compute the loss for each token
    loss = loss_object(real, pred)

    # Cast mask to the same dtype as loss and apply it
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    # Return the mean loss over non-padding tokens
    return tf.reduce_mean(loss)
