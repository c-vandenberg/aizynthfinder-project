import tensorflow as tf
from tensorflow.keras.losses import Loss

@tf.keras.utils.register_keras_serializable()
class MaskedSparseCategoricalCrossentropy(Loss):
    def __init__(self, padding_idx: int = 0, name: str = "masked_sparse_categorical_crossentropy", **kwargs):
        """
        Initializes the MaskedSparseCategoricalCrossentropy loss.

        Args:
            padding_idx (int, optional): The index used for padding tokens. Defaults to 0.
            name (str, optional): Name for the loss function. Defaults to "masked_sparse_categorical_crossentropy".
            **kwargs: Additional keyword arguments for the base class.
        """
        super(MaskedSparseCategoricalCrossentropy, self).__init__(name=name, **kwargs)
        self.padding_idx = padding_idx
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the masked sparse categorical cross-entropy loss.

        Args:
            y_true (tf.Tensor): Ground truth tensor of shape (batch_size, sequence_length).
            y_pred (tf.Tensor): Predicted tensor of shape (batch_size, sequence_length, vocab_size).

        Returns:
            tf.Tensor: Scalar tensor representing the mean loss over non-padding tokens.
        """
        # Create a mask to ignore padding tokens
        mask = tf.not_equal(y_true, self.padding_idx)  # Shape: (batch_size, sequence_length)
        mask = tf.cast(mask, dtype=y_pred.dtype)       # Cast mask to match y_pred's dtype

        # Compute the loss for each token
        loss = self.loss_object(y_true, y_pred)       # Shape: (batch_size, sequence_length)

        # Apply the mask to the loss
        loss *= mask

        # Compute the mean loss over non-padding tokens
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def get_config(self) -> dict:
        """
        Returns the configuration of the loss function for serialization.

        Returns:
            dict: Configuration dictionary.
        """
        config = super(MaskedSparseCategoricalCrossentropy, self).get_config()
        config.update({
            'padding_idx': self.padding_idx,
            # Note: `from_logits` and `reduction` are fixed in this implementation.
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        """
        Creates an instance of the loss function from its config.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            MaskedSparseCategoricalCrossentropy: A new instance of the loss function.
        """
        return cls(**config)