from abc import abstractmethod, ABCMeta
from tensorflow.keras.layers import Layer, Embedding


class EncoderInterface(Layer, metaclass=ABCMeta):
    def __init__(self, vocab_size: int, embedding_dim: int, units: int):
        super(EncoderInterface, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim)

    @abstractmethod
    def call(self, encoder_inputs, training=None):
        raise NotImplementedError
