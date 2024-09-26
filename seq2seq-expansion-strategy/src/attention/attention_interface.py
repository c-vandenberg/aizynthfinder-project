from abc import abstractmethod, ABCMeta
from tensorflow.keras.layers import Layer, Embedding

class AttentionInterface(Layer, metaclass=ABCMeta):
    def __init__(self, units: int, **kwargs):
        super(AttentionInterface, self).__init__(**kwargs)
        self.units = units

    @abstractmethod
    def call(self, outputs):
        raise NotImplementedError('Attention layer subclasses must implement `call` method')
