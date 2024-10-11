import os
from typing import List, Any, Union, Tuple, Dict

import onnx
import tf2onnx
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from sklearn.model_selection import ShuffleSplit, cross_validate


class Seq2SeqModelUtils:
    """
    Utility class for handling Seq2Seq Keras models, including cross-validation,
    model inspection, and saving models in various formats.
    """
    @staticmethod
    def seq2seq_cross_validator(
        n_splits: int,
        test_size: float,
        random_state: int,
        seq2seq_model: Any,
        feature_matrix: Union[np.ndarray, List[Any]],
        target_matrix: Union[np.ndarray, List[Any]]
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a Seq2Seq model using ShuffleSplit.

        Parameters
        ----------
        n_splits : int
            Number of re-shuffling and splitting iterations.
        test_size : float
            Proportion of the dataset to include in the test split.
        random_state : int
            Controls the randomness of the training and testing indices.
        seq2seq_model : Any
            The Seq2Seq model to be evaluated. This should be compatible with scikit-learn's cross_validate.
        feature_matrix : Union[np.ndarray, List[Any]]
            Feature data for the model.
        target_matrix : Union[np.ndarray, List[Any]]
            Target data for the model.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing cross-validation results, including scores for each metric.
        """
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
        Recursively inspect each layer and sublayer in the Keras model and print their configurations.

        Parameters
        ----------
        model : tensorflow.keras.Model
            The Keras model whose layers are to be inspected.

        Returns
        -------
        None
            This function prints the layer configurations to the console.
        """
        def _inspect_layer(model_layer: tf.keras.layers.Layer, indent: int = 0) -> None:
            """
            Helper function to recursively inspect a single layer and its sublayers.

            Parameters
            ----------
            model_layer : tensorflow.keras.layers.Layer
                The Keras layer to inspect.
            indent : int, optional
                Indentation level for printing sublayers, by default 0.

            Returns
            -------
            None
                This function prints the layer configuration to the console.
            """
            indent_str: str = "  " * indent
            print(f"{indent_str}Layer: {model_layer.name}")
            config: Dict[str, Any] = model_layer.get_config()
            for key, value in config.items():
                print(f"{indent_str}  - {key}: {value}")

            # Recursively inspect sublayers if present
            if hasattr(model_layer, 'layers'):  # For layers like Bidirectional, TimeDistributed, etc.
                for sublayer in model_layer.layers:
                    _inspect_layer(sublayer, indent + 1)
            elif hasattr(model_layer, 'layer'):  # For layers like RNN that contain a single layer
                _inspect_layer(model_layer.layer, indent + 1)

        for layer in model.layers:
            _inspect_layer(layer)

    @staticmethod
    def model_save_keras_format(keras_save_dir: str, model:Model) -> None:
        """
        Save the Keras model in the native Keras `.keras` format.

        Parameters
        ----------
        keras_save_dir : str
            Directory path where the `.keras` model will be saved.
        model : tensorflow.keras.Model
            The Keras model to be saved.

        Returns
        -------
        None
            The function saves the model to the specified directory and prints a confirmation message.
        """
        os.makedirs(keras_save_dir, exist_ok=True)
        keras_save_path: str = os.path.join(keras_save_dir, 'seq2seq_model.keras')
        model.save(keras_save_path)

        print(f"Model Keras V3 format save successful. Save file path: {keras_save_path}.")

    @staticmethod
    def model_save_hdf5_format(hdf5_save_dir: str, model:Model) -> None:
        """
        Save the Keras model in HDF5 `.h5` format.

        Parameters
        ----------
        hdf5_save_dir : str
            Directory path where the `.h5` model will be saved.
        model : tensorflow.keras.Model
            The Keras model to be saved.

        Returns
        -------
        None
            The function saves the model to the specified directory and prints a confirmation message.
        """
        os.makedirs(hdf5_save_dir, exist_ok=True)
        hdf5_save_path: str = os.path.join(hdf5_save_dir, 'seq2seq_model.h5')
        model.save(hdf5_save_path)

        print(f"Model HDF5 format save successful. Save file path: {hdf5_save_path}.")

    @staticmethod
    def model_save_onnx_format(
        onnx_output_dir: str,
        model: Model,
        max_encoder_seq_length: int,
        max_decoder_seq_length: int
    ) -> None:
        """
        Convert and save the Keras Seq2Seq model to ONNX format.

        Parameters
        ----------
        onnx_output_dir : str
            Directory path where the ONNX model will be saved.
        model : tensorflow.keras.Model
            The Keras model to be converted.
        max_encoder_seq_length : int
            Maximum sequence length for the encoder input.
        max_decoder_seq_length : int
            Maximum sequence length for the decoder input.

        Returns
        -------
        None
            The function converts the model to ONNX format, saves it, and prints a confirmation message.
        """
        os.makedirs(onnx_output_dir, exist_ok=True)
        onnx_save_path: str = os.path.join(onnx_output_dir, 'seq2seq_model.onnx')

        # Define the input signature for the function
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, max_encoder_seq_length], dtype=tf.int32, name='encoder_input'),
            tf.TensorSpec(shape=[None, max_decoder_seq_length], dtype=tf.int32, name='decoder_input'),
        ])
        def model_func(encoder_input, decoder_input):
            """
            TensorFlow function to define the models input signature.

            Parameters
            ----------
            encoder_input : tf.Tensor
                Tensor representing encoder inputs.
            decoder_input : tf.Tensor
                Tensor representing decoder inputs.

            Returns
            -------
            tf.Tensor
                Output tensor from the model.
            """
            return model([encoder_input, decoder_input])

        # Convert the function to ONNX
        model_proto: onnx.ModelProto
        try:
            model_proto, _ = tf2onnx.convert.from_function(
                model_func,
                input_signature=model_func.input_signature,
                opset=13,
                output_path=onnx_save_path
            )
            print(f"Model successfully converted to ONNX. Save file path: {onnx_save_path}")
        except Exception as e:
            print(f"Failed to convert model to ONNX format: {e}")
            raise

    @staticmethod
    def save_saved_model_format(model_save_path: str, model: Model) -> None:
        """
        Save the Keras model in the TensorFlow SavedModel format.

        Parameters
        ----------
        model_save_path : str
            Directory path where the SavedModel will be saved.
        model : tensorflow.keras.Model
            The Keras model to be saved.

        Returns
        -------
        None
            The function saves the model to the specified directory and prints a confirmation message.
        """
        os.makedirs(model_save_path, exist_ok=True)
        model.export(model_save_path)
        print(f"Model SavedModel format save successful. Save file path: {model_save_path}")
