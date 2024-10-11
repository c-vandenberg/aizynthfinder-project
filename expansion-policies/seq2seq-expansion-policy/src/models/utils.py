import os
import sys
import subprocess
from typing import List

from tensorflow.keras import Model
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
    def inspect_model_layers(model: Model) -> None:
        """
        Recursively inspect each layer and sublayer in the model and print their configuration.
        """
        def _inspect_layer(model_layer, indent=0):
            indent_str = "  " * indent
            print(f"{indent_str}Layer: {model_layer.name}")
            config = model_layer.get_config()
            for key, value in config.items():
                print(f"{indent_str}  - {key}: {value}")

            # Recursively inspect sublayers if any
            if hasattr(model_layer, 'layers'):  # For layers like Bidirectional, TimeDistributed, etc.
                for sublayer in model_layer.layers:
                    _inspect_layer(sublayer, indent + 1)
            elif hasattr(model_layer, 'layer'):  # For layers like RNN that contain a single layer
                _inspect_layer(model_layer.layer, indent + 1)

        for layer in model.layers:
            _inspect_layer(layer)

    @staticmethod
    def save_saved_model_format(model_save_path: str, model: Model) -> None:
        os.makedirs(model_save_path, exist_ok=True)
        model.export(model_save_path)
        print(f"Model saved to {model_save_path}")

    @staticmethod
    def convert_saved_model_to_onnx_cli(saved_model_path: str, onnx_output_dir: str) -> None:
        """
        Convert SavedModel to ONNX format.

        Conversion of SavedModel format via tf2onnx Python API has been deprecated. However, conversion to ONNX via CLI
        is still supported.

        :param saved_model_path:
        :param onnx_output_dir:
        :return:
        """
        os.makedirs(onnx_output_dir, exist_ok=True)
        onnx_output_path = os.path.join(onnx_output_dir, "seq2seq_model.onnx")

        tf2onnx_cli_command = [
            sys.executable, '-m', 'tf2onnx.convert',
            '--saved-model', saved_model_path,
            '--output', onnx_output_path,
            '--opset', '13',
        ]

        result = subprocess.run(
            tf2onnx_cli_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            print(f"Model has been converted to ONNX format at {onnx_output_path}")
        else:
            print(f"Error converting model to ONNX:\n{result.stderr}")
