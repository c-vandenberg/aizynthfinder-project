from abc import ABC, abstractmethod


class EncoderInterface(ABC):
    @abstractmethod
    def call(self, encoder_inputs, training=False):
        pass


class DecoderInterface(ABC):
    @abstractmethod
    def call(self, inputs, training=False, **kwargs):
        pass


class AttentionInterface(ABC):
    @abstractmethod
    def call(self, outputs):
        pass
