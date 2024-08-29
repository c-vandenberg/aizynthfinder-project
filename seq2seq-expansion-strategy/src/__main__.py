#!/usr/bin/env python3

import sys
import os
from aizynthfinder.aizynthfinder import AiZynthFinder
from models.seq2seq import Seq2SeqModel


def main():
    vocab_size = 5000  # Example vocabulary size
    model = Seq2SeqModel(input_vocab_size=vocab_size, output_vocab_size=vocab_size, embedding_dim=256, units=512)


if __name__ == '__main__':
    main()