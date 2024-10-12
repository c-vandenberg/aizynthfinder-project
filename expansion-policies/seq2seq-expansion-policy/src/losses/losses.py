import tensorflow as tf
from tensorflow.keras.losses import Loss


@tf.keras.utils.register_keras_serializable()
class MaskedSparseCategoricalCrossentropy(Loss):
    """
    Masked Sparse Categorical Crossentropy Loss Function.

    Computes the sparse categorical cross-entropy loss while ignoring padding tokens.

    Parameters
    ----------
    padding_idx : int, optional
        The index used for padding tokens (default is 0).
    name : str, optional
        Name for the loss function (default is 'masked_sparse_categorical_crossentropy').
    **kwargs
        Additional keyword arguments for the base Loss class.

    Attributes
    ----------
    padding_idx : int
        The index used for padding tokens.
    loss_function : tf.keras.losses.Loss
        The underlying loss function used to compute the loss.

    Methods
    -------
    call(y_true, y_pred)
        Computes the masked sparse categorical cross-entropy loss.
    get_config()
        Returns the configuration of the loss function.
    """
    def __init__(
        self,
        padding_idx: int = 0,
        name: str = "masked_sparse_categorical_crossentropy",
        **kwargs
    ) -> None:
        super(MaskedSparseCategoricalCrossentropy, self).__init__(name=name, **kwargs)
        self.padding_idx = padding_idx
        self.reduction = tf.keras.losses.Reduction.NONE
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            reduction=self.reduction
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the masked sparse categorical cross-entropy loss.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth tensor of shape (batch_size, sequence_length).
        y_pred : tf.Tensor
            Predicted tensor of shape (batch_size, sequence_length, vocab_size).

        Returns
        -------
        tf.Tensor
            Scalar tensor representing the mean loss over non-padding tokens.
        """
        # Compute the loss for each token
        loss = self.loss_function(y_true, y_pred)  # Shape: (batch_size, sequence_length)

        # Create a mask to ignore padding tokens
        mask = tf.not_equal(y_true, self.padding_idx) # Shape: (batch_size, sequence_length)
        mask = tf.cast(mask, dtype=loss.dtype) # Cast mask to match y_pred's dtype

        # Apply the mask to the loss
        loss *= mask

        # Compute the mean loss over non-padding tokens
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def get_config(self) -> dict:
        """
        Returns the configuration of the loss function for serialization.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        config = super(MaskedSparseCategoricalCrossentropy, self).get_config()
        config.update({
            'padding_idx': self.padding_idx,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'MaskedSparseCategoricalCrossentropy':
        """
        Creates an instance of the loss function from its configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        MaskedSparseCategoricalCrossentropy
            An instance of the loss function.
        """
        return cls(**config)