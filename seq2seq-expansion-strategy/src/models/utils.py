import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess_smiles(smiles_list, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(smiles_list)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences


def create_tokenizer(smiles_list):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, filters='')
    tokenizer.fit_on_texts(smiles_list)
    return tokenizer
