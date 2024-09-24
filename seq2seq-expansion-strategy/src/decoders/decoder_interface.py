from abc import abstractmethod, ABCMeta
from tensorflow.keras.layers import Layer, Embedding


class DecoderInterface(Layer, metaclass=ABCMeta):
    def __init__(self, vocab_size: int, embedding_dim: int, units: int):
        super(DecoderInterface, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim)

    @abstractmethod
    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError('Decoder layer subclasses must implement `call` method')
