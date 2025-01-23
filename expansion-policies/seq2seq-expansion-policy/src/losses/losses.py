from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras.losses import Loss

@tf.keras.utils.register_keras_serializable()
class WeightedSparseCategoricalCrossEntropy(Loss):
    """
    A custom Weighted Sparse Categorical Crossentropy class.
    This loss applies a per-token weight factor and masks out padding tokens.

    Parameters
    ----------
    token_to_weight_map : tf.Tensor
        A 1D tensor of shape [vocab_size] where token_to_weight_map[i]
        is the weight for token i.
    pad_token_id : int
        The token ID used for padding. Will be excluded from the loss.
    from_logits : bool
        Whether y_pred is expected to be logits or probabilities (softmax).
    name : str
        Optional name for the loss.
    """

    def __init__(
        self,
        token_to_weight_map: tf.Tensor,
        pad_token_id: int = 0,
        from_logits: bool = False,
        name: str = "WeightedSparseCategoricalCrossEntropy"
    ):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.token_to_weight_map = token_to_weight_map
        self.pad_token_id = pad_token_id
        self.from_logits = from_logits

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=self.from_logits,
            reduction=tf.keras.losses.Reduction.NONE
        )

    def call(self, y_true, y_pred):
        """
        y_true: (batch_size, seq_length)
        y_pred: (batch_size, seq_length, vocab_size)
        """
        # 1) Compute raw per-token crossentropy, shape = (batch_size, seq_length)
        per_token_loss = self.loss(y_true, y_pred)

        # 2) Gather weights for each token in y_true
        weights = tf.gather(self.token_to_weight_map, y_true)  # shape: (batch, seq_len)

        # 3) Multiply: Multiply raw loss by these weights
        weighted_loss = per_token_loss * weights

        # 4) Mask out pad tokens (zero their loss contribution)
        mask = tf.cast(tf.not_equal(y_true, self.pad_token_id), tf.float32)
        weighted_loss *= mask

        # 5) Reduce to a single scalar (average over all non-padding tokens)
        return tf.reduce_mean(weighted_loss)


@tf.keras.utils.register_keras_serializable()
class MaskedSparseCategoricalCrossEntropy(Loss):
    """
    Masked Sparse Categorical Crossentropy Loss Function.

    Computes the sparse categorical cross-entropy loss while ignoring padding tokens.

    Padding tokens are used in sequence modeling tasks to ensure uniform input lengths. By masking these tokens,
    the loss calculation focuses only on meaningful parts of the sequences, improving training efficiency and
    performance.

    Parameters
    ----------
    padding_idx : int, optional
        The index used for padding tokens (default is 0).
    label_smoothing : float, optional
        The amount of label smoothing to apply (default is 0.0). Label smoothing can help prevent the model
        from becoming over-confident.
    name : str, optional
        Name for the loss function (default is 'masked_sparse_categorical_crossentropy').
    **kwargs : Any
        Additional keyword arguments for the base `Loss` class.

    Attributes
    ----------
    padding_idx : int
        The index used for padding tokens.
    label_smoothing : float
        The amount of label smoothing applied.
    loss_function : tf.keras.losses.Loss
        The underlying loss function used to compute the loss.

    Methods
    -------
    call(y_true, y_pred)
        Computes the masked sparse categorical cross-entropy loss.
    get_config()
        Returns the configuration of the loss function for serialization.
    from_config(config)
        Creates an instance of the loss function from its configuration.

    Returns
    -------
    tf.Tensor
        Scalar tensor representing the mean loss over non-padding tokens.
    """
    def __init__(
        self,
        padding_idx: int = 0,
        label_smoothing: float = 0.0,
        name: str = "masked_sparse_categorical_crossentropy",
        **kwargs
    ) -> None:
        super(MaskedSparseCategoricalCrossEntropy, self).__init__(name=name, **kwargs)
        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing
        self.reduction = tf.keras.losses.Reduction.NONE

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the masked sparse categorical cross-entropy loss.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth tensor of shape `(batch_size, sequence_length)`.
            Each entry should be an integer representing the correct class.
        y_pred : tf.Tensor
            Predicted tensor of shape `(batch_size, sequence_length, vocab_size)`.
            Represents the probability distribution over classes for each timestep.

        Returns
        -------
        tf.Tensor
            Scalar tensor representing the mean loss over non-padding tokens.

        Raises
        ------
        ValueError
            If `y_true` is not a 2D tensor or `y_pred` is not a 3D tensor.
        """
        # Validate input dimensions
        if y_true.ndim != 2:
            raise ValueError(f"y_true must be a 2D tensor, got {y_true.ndim}D tensor.")
        if y_pred.ndim != 3:
            raise ValueError(f"y_pred must be a 3D tensor, got {y_pred.ndim}D tensor.")

        # Flatten y_true and y_pred for simplicity
        vocab_size = tf.shape(y_pred)[-1]
        y_true = tf.cast(y_true, tf.int32)
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1, vocab_size])

        # Create mask to ignore padding tokens
        mask = tf.not_equal(y_true_flat, self.padding_idx)

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = tf.cast(vocab_size, y_pred.dtype)
            label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)
            smooth_positives = 1.0 - label_smoothing
            smooth_negatives = label_smoothing / (num_classes - 1)

            # One-hot encode y_true
            y_true_one_hot = tf.one_hot(y_true_flat, depth=vocab_size, dtype=y_pred.dtype)
            y_true_smoothed = y_true_one_hot * smooth_positives + smooth_negatives

            # Compute loss
            loss = tf.keras.losses.categorical_crossentropy(
                y_true_smoothed,
                y_pred_flat,
                from_logits=False
            )
        else:
            # Compute loss without label smoothing
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true_flat,
                y_pred_flat,
                from_logits=False,
                reduction=self.reduction
            )

        # Apply the mask
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        # Compute mean loss over non-padding tokens
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the loss function for serialization.

        This configuration can be used to re-instantiate the loss function with the same parameters.

        Returns
        -------
        config : Dict[str, Any]
            Configuration dictionary containing all necessary parameters to recreate the loss function.
        """
        config = super(MaskedSparseCategoricalCrossEntropy, self).get_config()
        config.update({
            'padding_idx': self.padding_idx,
            'label_smoothing': self.label_smoothing,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'MaskedSparseCategoricalCrossEntropy':
        """
        Creates an instance of the loss function from its configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        MaskedSparseCategoricalCrossEntropy
            An instance of the loss function configured as per the provided dictionary.
        """
        return cls(**config)