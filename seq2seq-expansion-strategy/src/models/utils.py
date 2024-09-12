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


def preprocess_smiles(smiles_list, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(smiles_list)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences


def create_tokenizer(smiles_list):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, filters='')
    tokenizer.fit_on_texts(smiles_list)
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
