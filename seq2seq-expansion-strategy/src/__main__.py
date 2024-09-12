#!/usr/bin/env python3

import sys
import os
import re
from sklearn.model_selection import train_test_split
from aizynthfinder.aizynthfinder import AiZynthFinder
from models.utils import load_smiles_data_from_csv, preprocess_smiles, create_tokenizer
from models.seq2seq import Seq2SeqModel


def main():
    with open('../data/processed/products', 'r') as file:
        products_x_dataset = file.readlines()

    with open('../data/processed/reactants', 'r') as file:
        reactants_y_dataset = file.readlines()

    # Split data into training and test data sets using `train_test_split()` function
    (seq2seq_model_x_train_data, seq2seq_model_x_test_data,
     seq2seq_model_y_train_data, seq2seq_model_y_test_data) = train_test_split(
        products_x_dataset,
        reactants_y_dataset,
        test_size=0.3,  # 30% of the data is reserved for testing, and 70% is used for training
        random_state=42  # Random number generator seed
    )

    x_train_data_tokenizer = create_tokenizer(smiles_list=seq2seq_model_x_train_data)
    y_train_data_tokenizer = create_tokenizer(smiles_list=seq2seq_model_y_train_data)

    x_input_data = preprocess_smiles(
        smiles_list=seq2seq_model_x_train_data,
        tokenizer=x_train_data_tokenizer,
        max_length=100
    )

    y_output_data = preprocess_smiles(
        smiles_list=seq2seq_model_y_train_data,
        tokenizer=x_train_data_tokenizer,
        max_length=100
    )
    input_vocab_size = len(x_train_data_tokenizer.word_index) + 1
    output_vocab_size = len(y_train_data_tokenizer.word_index) + 1

    model = Seq2SeqModel(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        embedding_dim=128,
        units=256
    )

    model.train(x_input_data, y_output_data)

    model.save_model('../data/models')


if __name__ == '__main__':
    main()
