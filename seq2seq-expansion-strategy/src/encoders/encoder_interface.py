from abc import abstractmethod, ABCMeta
from tensorflow.keras.layers import Layer, Embedding


class EncoderInterface(Layer, metaclass=ABCMeta):
    def __init__(self, vocab_size: int, embedding_dim: int, units: int, **kwargs):
        super(EncoderInterface, self).__init__(**kwargs)
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim)

    @abstractmethod
    def call(self, encoder_inputs, training=None):
        raise NotImplementedError('Encoder layer subclasses must implement `call` method')
