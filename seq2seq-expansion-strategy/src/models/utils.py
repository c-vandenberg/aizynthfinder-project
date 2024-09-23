import random
import tensorflow as tf
import tf2onnx
from sklearn.model_selection import ShuffleSplit, cross_validate
from typing import List

random.seed(42)
tf.random.set_seed(42)


class Seq2SeqModelUtils:
    @staticmethod
    def seq2seq_cross_validator(n_splits: int, test_size: float, random_state: int, seq2seq_model, feature_matrix,
                                target_matrix):
        cross_validator: ShuffleSplit = ShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state
        )

        scoring_metrics: List[str] = ['r2', 'neg_mean_absolute_error']

        return cross_validate(
            seq2seq_model,
            feature_matrix,
            target_matrix,
            scoring=scoring_metrics,
            cv=cross_validator
        )

    @staticmethod
    def masked_sparse_categorical_crossentropy(real, pred):
        """
        Computes the sparse categorical cross-entropy loss while masking out padding tokens.

        Args:
            real (tf.Tensor): The ground truth tensor.
            pred (tf.Tensor): The predicted tensor.

        Returns:
            tf.Tensor: The computed loss.
        """
        # Create a mask to ignore padding tokens (assumed to be 0)
        mask = tf.math.logical_not(tf.math.equal(real, 0))

        # Define and instantiate the loss object
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

        # Compute the loss for each token
        loss = loss_object(real, pred)

        # Cast mask to the same dtype as loss and apply it
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        # Return the mean loss over non-padding tokens
        return tf.reduce_mean(loss)

    @staticmethod
    def save_model(model, save_path):
        """
        Save the model in TensorFlow SavedModel format.
        """
        model.save(save_path)
        print(f"Model saved to {save_path}")

    @staticmethod
    def convert_to_onnx(saved_model_path, onnx_file_path):
        """
        Convert the TensorFlow SavedModel to ONNX format and save it.
        """
        # Load the TensorFlow model
        model = tf.keras.models.load_model(saved_model_path)

        # Convert the TensorFlow model to ONNX format
        onnx_model = tf2onnx.convert.from_keras(model)

        # Save the ONNX model to a file
        with open(onnx_file_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX model saved to {onnx_file_path}")
