import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer, Embedding, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Optional, Tuple, Any

from encoders.lstm_encoders import StackedBidirectionalLSTMEncoder
from decoders.lstm_decoders import StackedLSTMDecoder
import onnx
import onnxruntime as ort
import numpy as np

from trainers.trainer import Trainer

def main():
    onnx_model = onnx.load("data/training/pande-et-al/model-v7/model.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    # Create a runtime session
    ort_session = ort.InferenceSession("data/training/pande-et-al/model-v7/model.onnx")

    # Prepare dummy input data
    dummy_encoder_input = np.random.randint(0, 40, size=(32, 140)).astype(np.float32)
    dummy_decoder_input = np.random.randint(0, 40, size=(32, 139)).astype(np.float32)

    # Run inference
    inputs = {
        'inputs': dummy_encoder_input,
        'inputs_1': dummy_decoder_input
    }

    outputs = ort_session.run(None, inputs)
    print("ONNX model predictions shape:", outputs[0].shape)

if __name__ == '__main__':
    main()