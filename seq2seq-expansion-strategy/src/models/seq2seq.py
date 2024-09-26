import tensorflow as tf
from tensorflow.keras.layers import Dense
from encoders.lstm_encoders import StackedBidirectionalLSTMEncoder
from decoders.lstm_decoders import StackedLSTMDecoder


class RetrosynthesisSeq2SeqModel(tf.keras.Model):
    def __init__(self, input_vocab_size: int, output_vocab_size: int, encoder_embedding_dim: int,
                 decoder_embedding_dim: int, units: int, dropout_rate: float = 0.2,
                 *args, **kwargs):
        super(RetrosynthesisSeq2SeqModel, self).__init__(*args, **kwargs)

        self.units = units
        self.encoder = StackedBidirectionalLSTMEncoder(input_vocab_size, encoder_embedding_dim, units, dropout_rate)
        self.decoder = StackedLSTMDecoder(output_vocab_size, decoder_embedding_dim, units, dropout_rate)

        # Save the vocabulary sizes
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        # Mapping encoder final states to decoder initial states
        self.enc_state_h = Dense(units, name='enc_state_h')
        self.enc_state_c = Dense(units, name='enc_state_c')

        # Store the data processors (to be set externally)
        self.encoder_data_processor = None
        self.decoder_data_processor = None

        # Store dropout rate
        self.dropout_rate = dropout_rate

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

    def call(self, inputs, training=None):
        # Extract encoder and decoder inputs
        encoder_input, decoder_input = inputs

        # Encoder
        encoder_output, state_h, state_c = self.encoder.call(encoder_input, training=training)

        # Map encoder final states to decoder initial states
        decoder_initial_state_h = self.enc_state_h(state_h)  # (batch_size, units)
        decoder_initial_state_c = self.enc_state_c(state_c)  # (batch_size, units)
        decoder_initial_state = [decoder_initial_state_h, decoder_initial_state_c]

        # Prepare decoder inputs as a tuple
        decoder_inputs = (decoder_input, decoder_initial_state, encoder_output)
        encoder_mask = self.encoder.compute_mask(encoder_input)

        # Decoder
        output = self.decoder.call(
            decoder_inputs,
            training=training,
            mask=encoder_mask
        )

        return output

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict: A Python dictionary containing the layer's configuration.
        """
        config = super(RetrosynthesisSeq2SeqModel, self).get_config()
        config.update({
            'units': self.units,
            'input_vocab_size': self.input_vocab_size,
            'output_vocab_size': self.output_vocab_size,
            'encoder_embedding_dim': self.encoder.embedding.output_dim,
            'decoder_embedding_dim': self.decoder.embedding.output_dim,
            'dropout_rate': self.dropout_rate,
            'encoder': tf.keras.layers.serialize(self.encoder),
            'decoder': tf.keras.layers.serialize(self.decoder),
            'enc_state_h': tf.keras.layers.serialize(self.enc_state_h),
            'enc_state_c': tf.keras.layers.serialize(self.enc_state_c)
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a layer from its config.

        Args:
            config (dict): A Python dictionary containing the layer's configuration.

        Returns:
            RetrosynthesisSeq2SeqModel: A new instance of RetrosynthesisSeq2SeqModel configured using the provided
            config.
        """
        # Deserialize layers
        config['encoder'] = tf.keras.layers.deserialize(config['encoder'])
        config['decoder'] = tf.keras.layers.deserialize(config['decoder'])
        config['enc_state_h'] = tf.keras.layers.deserialize(config['enc_state_h'])
        config['enc_state_c'] = tf.keras.layers.deserialize(config['enc_state_c'])
        return cls(**config)


class BestValLossCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager):
        super(BestValLossCheckpointCallback, self).__init__()
        self.checkpoint_manager = checkpoint_manager
        self.best_val_loss = float('inf')  # Initialize with infinity

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                save_path = self.checkpoint_manager.save()
                print(
                    f"\nEpoch {epoch+1}: Validation loss improved to {current_val_loss:.4f}. "
                    f"Saving checkpoint to {save_path}"
                )
