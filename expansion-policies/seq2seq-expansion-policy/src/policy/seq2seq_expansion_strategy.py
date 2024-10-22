import tensorflow as tf
from rdkit import Chem
from aizynthfinder.context.policy.expansion_strategies import ExpansionStrategy
from aizynthfinder.chem import SmilesBasedRetroReaction
from aizynthfinder.chem import TreeMolecule
from aizynthfinder.chem.reaction import RetroReaction
from aizynthfinder.context.config import Configuration
from aizynthfinder.utils.type_utils import List, Optional, Sequence, Tuple

from data.utils.tokenization import SmilesTokenizer
from data.utils.preprocessing import SmilesDataPreprocessor
from models.seq2seq import RetrosynthesisSeq2SeqModel
from encoders.lstm_encoders import StackedBidirectionalLSTMEncoder
from decoders.lstm_decoders import StackedLSTMDecoder
from attention.attention import BahdanauAttention
from losses.losses import MaskedSparseCategoricalCrossentropy
from metrics.perplexity import Perplexity
from callbacks.checkpoints import BestValLossCallback
from callbacks.validation_metrics import ValidationMetricsCallback


class Seq2SeqExpansionStrategy(ExpansionStrategy):
    """
    A Seq2Seq-based expansion strategy using an ONNX model for retrosynthesis prediction.

    :param key: the key or label
    :param config: the configuration of the tree search
    :param model_path: path to the ONNX model file
    :param top_k: number of top predictions to consider
    """
    def __init__(self, key: str, config: Configuration, **kwargs: str) -> None:
        super().__init__(key, config, **kwargs)

        model_path = kwargs["model"]
        tokenizer_path = kwargs["tokenizer"]
        self.max_encoder_seq_length = int(kwargs.get("max_encoder_seq_length", 140))
        self.max_decoder_seq_length = int(kwargs.get("max_decoder_seq_length", 140))
        self.beam_width = int(kwargs.get("beam_width", 5))
        self.use_remote_models = bool(kwargs.get("use_remote_models", True))

        self.model = self.load_model(model_path)
        self.smiles_tokenizer = self.load_tokenizer(tokenizer_path)

        self._logger.info(f"Loaded Seq2Seq model and tokenizers for expansion policy {self.key}")

    def load_model(self, model_path: str):
        self._logger.info(f"Loading Seq2Seq model from {model_path}")

        custom_objects = {
            'RetrosynthesisSeq2SeqModel': RetrosynthesisSeq2SeqModel,
            'StackedBidirectionalLSTMEncoder': StackedBidirectionalLSTMEncoder,
            'StackedLSTMDecoder': StackedLSTMDecoder,
            'BahdanauAttention': BahdanauAttention,
            'MaskedSparseCategoricalCrossentropy': MaskedSparseCategoricalCrossentropy,
            'Perplexity': Perplexity,
            'BestValLossCallback': BestValLossCallback,
            'ValidationMetricsCallback': ValidationMetricsCallback
        }
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            self._logger.info("Model loaded successfully.")
            self._logger.info("Model Summary:")
            model.summary(print_fn=self._logger.info)
        except Exception as e:
            self._logger.error(f"Error loading model: {e}")
            raise
        return model

    def load_tokenizer(self, tokenizer_path: str):
        self._logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer: SmilesTokenizer = SmilesTokenizer.from_json(tokenizer_path)
        self._logger.info("Tokenizer loaded successfully.")

        return tokenizer

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Generate retrosynthetic actions using the Seq2Seq model.

        :param molecules: the molecules to consider
        :param cache_molecules: additional molecules to submit to the expansion
            policy but that only will be cached for later use
        :return: the actions and the priors of those actions
        """
        possible_actions = []
        priors = []

        smiles_list = [mol.smiles for mol in molecules]
        predicted_precursors_list, probabilities_list = self.predict_precursors(smiles_list)

        for mol, predicted_precursors, probs in zip(molecules, predicted_precursors_list, probabilities_list):
            for precursor_smiles, prob in zip(predicted_precursors, probs):
                metadata = {
                    "policy_probability": float(prob),
                    "policy_name": self.key,
                }
                new_action = SmilesBasedRetroReaction(
                    mol,
                    metadata=metadata,
                    reactants_str=precursor_smiles,
                )
                possible_actions.append(new_action)
                priors.append(prob)

        return possible_actions, priors

    def predict_precursors(self, smiles_list: List[str]) -> Tuple[List[List[str]], List[List[float]]]:
        tokenized_smiles_list = self.smiles_tokenizer.tokenize_list(
            smiles_list,
            is_input_sequence=True
        )

        encoder_data_preprocessor: SmilesDataPreprocessor = SmilesDataPreprocessor(
            smiles_tokenizer=self.smiles_tokenizer,
            max_seq_length=self.max_encoder_seq_length
        )
        preprocessed_smiles = encoder_data_preprocessor.preprocess_smiles(
            tokenized_smiles_list=tokenized_smiles_list
        )

        start_token = self.smiles_tokenizer.start_token
        start_token_id = self.smiles_tokenizer.word_index[start_token]
        end_token = self.smiles_tokenizer.end_token
        end_token_id = self.smiles_tokenizer.word_index[end_token]

        # Use beam search to get multiple predictions per molecule
        predicted_seqs_list = self.model.predict_sequence_beam_search(
            encoder_input=preprocessed_smiles,
            beam_width=self.beam_width,
            max_length=self.max_decoder_seq_length,
            start_token_id=start_token_id,
            end_token_id=end_token_id
        )

        all_predicted_smiles = []
        all_probabilities = []

        for predicted_seqs in predicted_seqs_list:
            # Convert token sequences to SMILES strings
            predicted_smiles = self.smiles_tokenizer.sequences_to_texts(
                sequences=[predicted_seqs],
                is_input_sequence=True
            )

            # For beam search, you can assign probabilities based on beam scores if available
            # Here, we assign equal probabilities for simplicity
            num_predictions = len(predicted_smiles)
            probabilities = [1.0 / num_predictions] * num_predictions

            # Validate and append
            for smiles_string in predicted_smiles:
                if self._is_valid_smiles(smiles_string):
                    all_predicted_smiles.append([smiles_string])
                    all_probabilities.append(probabilities)
                else:
                    self._logger.warning(f"Invalid SMILES generated: {predicted_smiles}")

        return all_predicted_smiles, all_probabilities

    @staticmethod
    def _is_valid_smiles(smiles: str) -> bool:
        return Chem.MolFromSmiles(smiles) is not None

    @staticmethod
    def _clean_sequence(sequence: str, start_token: str, end_token: str) -> str:
        # Remove start token
        if sequence.startswith(start_token):
            sequence = sequence[len(start_token):]
        # Remove everything after end token
        end_idx = sequence.find(end_token)
        if end_idx != -1:
            sequence = sequence[:end_idx]
        return sequence.strip()

    def reset_cache(self) -> None:
        pass  # Implement caching if necessary
