from abc import abstractmethod, ABCMeta
from tensorflow.keras.layers import Layer, Embedding

class AttentionInterface(Layer, metaclass=ABCMeta):
    def __init__(self, units: int):
        super(AttentionInterface, self).__init__()
        self.units = units

    @abstractmethod
    def call(self, outputs):
        pass