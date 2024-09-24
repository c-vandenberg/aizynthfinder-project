from abc import abstractmethod, ABCMeta


class InferenceDecoderInterface(metaclass=ABCMeta):
    @abstractmethod
    def decode(self, encoder_output):
        """
        Decode the encoder output into a target sequence.

        Args:
            encoder_output: The output from the encoder.

        Returns:
            Decoded sequences.
        """
        raise NotImplementedError('Inference decoder subclasses must implement `decode` method')
