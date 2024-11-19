import numpy as np
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
        self.return_top_n = min(int(kwargs.get("return_top_n", 1)), self.beam_width)

        self.model = self.load_model(model_path)
        self.smiles_tokenizer = self.load_tokenizer(tokenizer_path)
        self.model.smiles_tokenizer = self.smiles_tokenizer

        self._logger.info(f"Loaded Seq2Seq model and tokenizers for expansion policy {self.key}")

    def load_model(self, model_path: str):
        self._logger.info(f"Loading Seq2Seq model from {model_path}")

        custom_objects = {
            'RetrosynthesisSeq2SeqModel': RetrosynthesisSeq2SeqModel,
            'StackedBidirectionalLSTMEncoder': StackedBidirectionalLSTMEncoder,
            'StackedLSTMDecoder': StackedLSTMDecoder,
            'BahdanauAttention': BahdanauAttention
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
        self._logger.debug(f"Target molecules: {smiles_list}")
        predicted_precursors_list, probabilities_list = self.predict_precursors(smiles_list)

        # Log the predictions for each molecule
        for idx, (mol, predicted_precursors, probs) in enumerate(
                zip(molecules, predicted_precursors_list, probabilities_list)):
            self._logger.debug(f"Predictions for molecule {mol.smiles}:")
            for precursor_smiles, prob in zip(predicted_precursors, probs):
                self._logger.debug(f"  Precursor: {precursor_smiles}, Probability: {prob}")

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

        encoder_data_preprocessor = SmilesDataPreprocessor(
            smiles_tokenizer=self.smiles_tokenizer,
            max_seq_length=self.max_encoder_seq_length
        )
        preprocessed_smiles = encoder_data_preprocessor.preprocess_smiles(
            tokenized_smiles_list=tokenized_smiles_list
        )

        start_token_id = self.smiles_tokenizer.word_index[self.smiles_tokenizer.start_token]
        end_token_id = self.smiles_tokenizer.word_index[self.smiles_tokenizer.end_token]

        # Use beam search to get multiple predictions per molecule
        predicted_seqs_list, seq_scores_list = self.model.predict_sequence_beam_search(
            encoder_input=preprocessed_smiles,
            beam_width=self.beam_width,
            max_length=self.max_decoder_seq_length,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            return_top_n=self.return_top_n
        )

        all_predicted_smiles = []
        all_probabilities = []

        for idx, (predicted_seqs, seq_scores) in enumerate(zip(predicted_seqs_list, seq_scores_list)):
            # Convert token sequences to SMILES strings
            predicted_smiles = self.smiles_tokenizer.sequences_to_texts(
                sequences=predicted_seqs,
                is_input_sequence=False
            )

            # Convert negative log probabilities to probabilities
            # First, convert seq_scores to a numpy array
            seq_scores = np.array(seq_scores)
            # Since seq_scores are negative log probabilities, we can compute probabilities as follows:
            # prob = exp(-score)
            exp_neg_scores = np.exp(-seq_scores)
            # Normalize probabilities
            probabilities = exp_neg_scores / np.sum(exp_neg_scores)

            # Validate and append
            valid_smiles = []
            valid_probs = []
            for smiles_string, prob in zip(predicted_smiles, probabilities):
                cleaned_smiles = self._clean_sequence(
                    smiles_string,
                    self.smiles_tokenizer.start_token,
                    self.smiles_tokenizer.end_token
                )
                # Split the SMILES string on '.' to handle multiple reactants
                reactant_smiles_list = cleaned_smiles.split('.')
                is_valid = all(self._is_valid_smiles(smi) for smi in reactant_smiles_list)
                if is_valid:
                    valid_smiles.append(cleaned_smiles)
                    valid_probs.append(float(prob))  # Ensure prob is a float, not numpy type
                else:
                    self._logger.warning(f"Invalid SMILES generated: {cleaned_smiles}")

            # Log the valid predicted SMILES
            self._logger.debug(f"Valid predicted SMILES for molecule {idx}: {valid_smiles}")

            all_predicted_smiles.append(valid_smiles)
            all_probabilities.append(valid_probs)

        return all_predicted_smiles, all_probabilities

    @staticmethod
    def _is_valid_smiles(smiles: str) -> bool:
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            # In case of any parsing exceptions
            return False

    @staticmethod
    def _clean_sequence(sequence: str, start_token: str, end_token: str) -> str:
        # Remove start token and the following space
        if sequence.startswith(start_token + ' '):
            sequence = sequence[len(start_token) + 1:]
        # Remove everything after end token and the preceding space
        end_idx = sequence.find(' ' + end_token)
        if end_idx != -1:
            sequence = sequence[:end_idx]
        # Remove any remaining start or end tokens
        sequence = sequence.replace(start_token, '').replace(end_token, '')
        # Remove all spaces
        sequence = sequence.replace(' ', '')
        return sequence.strip()

    def reset_cache(self) -> None:
        pass  # Implement caching if necessary
