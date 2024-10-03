from typing import List

import tf2onnx
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit, cross_validate

class Seq2SeqModelUtils:
    @staticmethod
    def seq2seq_cross_validator(
        n_splits: int,
        test_size: float,
        random_state: int,
        seq2seq_model,
        feature_matrix,
        target_matrix
    ):
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
    def inspect_model_layers(model) -> None:
        """
        Recursively inspect each layer and sublayer in the model and print their configuration.
        """
        def _inspect_layer(layer, indent=0):
            indent_str = "  " * indent
            print(f"{indent_str}Layer: {layer.name}")
            config = layer.get_config()
            for key, value in config.items():
                print(f"{indent_str}  - {key}: {value}")

            # Recursively inspect sublayers if any
            if hasattr(layer, 'layers'):  # For layers like Bidirectional, TimeDistributed, etc.
                for sublayer in layer.layers:
                    _inspect_layer(sublayer, indent + 1)
            elif hasattr(layer, 'layer'):  # For layers like RNN that contain a single layer
                _inspect_layer(layer.layer, indent + 1)

        for layer in model.layers:
            _inspect_layer(layer)

    @staticmethod
    def convert_to_onnx(saved_model_path, onnx_file_path) -> None:
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
