import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.train import Checkpoint, CheckpointManager
from tensorflow.keras.callbacks import Callback
from encoders.lstm_encoders import StackedBidirectionalLSTMEncoder
from decoders.lstm_decoders import StackedLSTMDecoder
from typing import Optional, Any, Tuple


class RetrosynthesisSeq2SeqModel(Model):
    def __init__(self, input_vocab_size: int, output_vocab_size: int, encoder_embedding_dim: int,
                 decoder_embedding_dim: int, units: int, num_encoder_layers = 2, num_decoder_layers: int = 4,
                 dropout_rate: float = 0.2, *args, **kwargs):
        super(RetrosynthesisSeq2SeqModel, self).__init__(*args, **kwargs)

        # Save the number of units (neurons)
        self.units: int = units

        # Encoder layer
        self.encoder: StackedBidirectionalLSTMEncoder = StackedBidirectionalLSTMEncoder(
            vocab_size=input_vocab_size,
            encoder_embedding_dim=encoder_embedding_dim,
            units=units,
            num_layers=num_encoder_layers,
            dropout_rate=dropout_rate
        )

        # Decoder layer
        self.decoder: StackedLSTMDecoder = StackedLSTMDecoder(
            vocab_size=output_vocab_size,
            decoder_embedding_dim=decoder_embedding_dim,
            units=units,
            num_layers=num_decoder_layers,
            dropout_rate=dropout_rate
        )

        # Save the vocabulary sizes
        self.input_vocab_size: int = input_vocab_size
        self.output_vocab_size: int = output_vocab_size

        # Mapping encoder final states to decoder initial states
        self.enc_state_h: Dense = Dense(units, name='enc_state_h')
        self.enc_state_c: Dense = Dense(units, name='enc_state_c')

        # Store the data processors (to be set externally)
        self.encoder_data_processor: Optional[Any] = None
        self.decoder_data_processor: Optional[Any] = None

        # Save the dropout rate
        self.dropout_rate: float = dropout_rate

    def build(self, input_shape):
        # Define the input shapes for encoder and decoder
        encoder_input_shape, decoder_input_shape = input_shape

        # Pass a dummy input through encoder and decoder to initialize weights
        encoder_dummy = tf.zeros(encoder_input_shape)
        decoder_dummy = tf.zeros(decoder_input_shape)

        # Forward pass to build the model
        self.call((encoder_dummy, decoder_dummy), training=False)

        # Mark the model as built
        super(RetrosynthesisSeq2SeqModel, self).build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of the Seq2Seq model.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): Tuple containing encoder and decoder inputs.
            training (Optional[bool], optional): Training flag. Defaults to None.

        Returns:
            tf.Tensor: The output predictions from the decoder.
        """
        # Extract encoder and decoder inputs
        encoder_input, decoder_input = inputs

        # Encoder
        encoder_output, state_h, state_c = self.encoder(encoder_input, training=training)

        # Map encoder final states to decoder initial states
        decoder_initial_state_h: tf.Tensor = self.enc_state_h(state_h)  # (batch_size, units)
        decoder_initial_state_c: tf.Tensor = self.enc_state_c(state_c)  # (batch_size, units)
        decoder_initial_state: Tuple[tf.Tensor, tf.Tensor] = (decoder_initial_state_h, decoder_initial_state_c)

        # Prepare decoder inputs as a tuple
        decoder_inputs = (
            decoder_input,
            decoder_initial_state,
            encoder_output
        )

        # Extract encoder mask
        encoder_mask: Optional[tf.Tensor] = self.encoder.compute_mask(encoder_input)

        # Decoder
        output: tf.Tensor = self.decoder(
            decoder_inputs,
            training=training,
            mask=encoder_mask
        )

        return output

    def get_config(self) -> dict:
        config = {
            'input_vocab_size': self.input_vocab_size,
            'output_vocab_size': self.output_vocab_size,
            'encoder_embedding_dim': self.encoder.embedding.output_dim,
            'decoder_embedding_dim': self.decoder.embedding.output_dim,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'name': self.name,
        }
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'RetrosynthesisSeq2SeqModel':
        return cls(**config)

class BestValLossCheckpointCallback(Callback):
    def __init__(self, checkpoint_manager: CheckpointManager):
        super(BestValLossCheckpointCallback, self).__init__()
        self.checkpoint_manager: CheckpointManager = checkpoint_manager
        self.best_val_loss: float = float('inf')  # Initialize with infinity

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss: float = logs.get('val_loss')
        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                save_path: str = self.checkpoint_manager.save()
                print(
                    f"\nEpoch {epoch+1}: Validation loss improved to {current_val_loss:.4f}. "
                    f"Saving checkpoint to {save_path}"
                )
