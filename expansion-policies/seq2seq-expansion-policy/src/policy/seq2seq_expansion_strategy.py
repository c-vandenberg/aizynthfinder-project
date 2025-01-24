import numpy as np
import tensorflow as tf
from rdkit import Chem
from aizynthfinder.context.policy.expansion_strategies import ExpansionStrategy
from aizynthfinder.chem import SmilesBasedRetroReaction
from aizynthfinder.chem import TreeMolecule
from aizynthfinder.chem.reaction import RetroReaction
from aizynthfinder.context.config import Configuration
from aizynthfinder.utils.type_utils import List, Optional, Sequence, Tuple

from data.utils.tokenisation import SmilesTokeniser
from data.utils.preprocessing import TokenisedSmilesPreprocessor
from models.seq2seq import RetrosynthesisSeq2SeqModel
from encoders.lstm_encoders import StackedBidirectionalLSTMEncoder
from decoders.lstm_decoders import StackedLSTMDecoder
from attention.attention import BahdanauAttention


class Seq2SeqExpansionStrategy(ExpansionStrategy):
    """
    A Seq2Seq-based expansion strategy using a trained model for retrosynthesis prediction.

    This strategy generates retrosynthetic actions by predicting precursor molecules
    for a given target molecule using a Seq2Seq model. It integrates with the AiZynthFinder
    framework as an expansion policy.

    Parameters
    ----------
    key : str
        The key or label for this strategy.
    config : Configuration
        The configuration object of the tree search.
    **kwargs : dict
        Additional keyword arguments:
        - model (str): Path to the trained model file.
        - tokenizer (str): Path to the tokenizer file.
        - max_encoder_seq_length (int, optional): Maximum sequence length for the encoder. Default is 140.
        - max_decoder_seq_length (int, optional): Maximum sequence length for the decoder. Default is 140.
        - beam_width (int, optional): Beam width for beam search decoding. Default is 5.
        - use_remote_models (bool, optional): Whether to use remote models. Default is True.
        - return_top_n (int, optional): Number of top sequences to return. Default is 1.

    Attributes
    ----------
    _model : RetrosynthesisSeq2SeqModel
        The loaded Seq2Seq model for prediction.
    _smiles_tokenizer : SmilesTokeniser
        The tokenizer used for encoding and decoding SMILES strings.
    _max_encoder_seq_length : int
        Maximum sequence length for the encoder.
    _max_decoder_seq_length : int
        Maximum sequence length for the decoder.
    _beam_width : int
        Beam width for beam search decoding.
    _use_remote_models : bool
        Whether to use remote models.
    _return_top_n : int
        Number of top sequences to return.

    Methods
    -------
    get_actions(molecules, cache_molecules=None)
        Generate retrosynthetic actions using the Seq2Seq model.
    """
    def __init__(self, key: str, config: Configuration, **kwargs: str) -> None:
        super().__init__(key, config, **kwargs)

        model_path = kwargs["model"]
        tokenizer_path = kwargs["tokenizer"]
        self._max_encoder_seq_length = int(kwargs.get("max_encoder_seq_length", 140))
        self._max_decoder_seq_length = int(kwargs.get("max_decoder_seq_length", 140))
        self._beam_width = int(kwargs.get("beam_width", 5))
        self._use_remote_models = bool(kwargs.get("use_remote_models", True))
        self._return_top_n = min(int(kwargs.get("return_top_n", 1)), self._beam_width)

        self._model = self.load_model(model_path)
        self._smiles_tokenizer = self.load_tokenizer(tokenizer_path)
        self._model._smiles_tokenizer = self._smiles_tokenizer

        self._logger.info(f"Loaded Seq2Seq model and tokenizers for expansion policy {self.key}")

    def load_model(self, model_path: str) -> RetrosynthesisSeq2SeqModel:
        """
        Loads the Seq2Seq model from the specified path.

        Parameters
        ----------
        model_path : str
            Path to the trained model file.

        Returns
        -------
        RetrosynthesisSeq2SeqModel
            The loaded Seq2Seq model.

        Raises
        ------
        Exception
            If there is an error loading the model.
        """
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

    def load_tokenizer(self, tokenizer_path: str) -> SmilesTokeniser:
        """
        Loads the SMILES tokenizer from the specified path.

        Parameters
        ----------
        tokenizer_path : str
            Path to the tokenizer file.

        Returns
        -------
        SmilesTokeniser
            The loaded SMILES tokenizer.
        """
        self._logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer: SmilesTokeniser = SmilesTokeniser.from_json(tokenizer_path, self._logger)
        self._logger.info("Tokenizer loaded successfully.")

        return tokenizer

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Generate retrosynthetic actions using the Seq2Seq model.

        Parameters
        ----------
        molecules : sequence of TreeMolecule
            The target molecules to consider for expansion.
        cache_molecules : sequence of TreeMolecule, optional
            Additional molecules to submit to the expansion policy but that will only be cached
            for later use. Default is None.

        Returns
        -------
        possible_actions : List[RetroReaction]
            The list of generated retrosynthetic reactions.
        priors : List[float]
            The corresponding probabilities (priors) of the generated actions.
        """
        possible_actions = []
        priors = []

        smiles_list = [mol.smiles for mol in molecules]
        self._logger.debug(f"Target molecules: {smiles_list}")
        predicted_precursors_list, probabilities_list = self.predict_precursors(smiles_list)

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
        """
        Predicts precursor molecules for a list of target SMILES strings.

        Parameters
        ----------
        smiles_list : list of str
            The list of target SMILES strings.

        Returns
        -------
        all_predicted_smiles : List[List[str]]
            A list containing lists of predicted precursor SMILES strings for each target.
        all_probabilities : List[List[float]]
            A list containing lists of probabilities corresponding to each predicted precursor.

        Notes
        -----
        This method uses beam search decoding to generate multiple predictions per molecule.
        It also filters out invalid SMILES strings from the predictions.
        """
        tokenized_smiles_list = self._smiles_tokenizer.tokenise_list(
            smiles_list,
            is_input_sequence=True
        )

        encoder_data_preprocessor = TokenisedSmilesPreprocessor(
            smiles_tokenizer=self._smiles_tokenizer,
            max_seq_length=self._max_encoder_seq_length
        )
        preprocessed_smiles = encoder_data_preprocessor.preprocess_smiles(
            tokenized_smiles_list=tokenized_smiles_list
        )

        start_token_id = self._smiles_tokenizer.word_index[self._smiles_tokenizer.start_token]
        end_token_id = self._smiles_tokenizer.word_index[self._smiles_tokenizer.end_token]

        # Use beam search to get multiple predictions per molecule
        predicted_seqs_list, seq_scores_list = self._model.predict_sequence_beam_search(
            encoder_input=preprocessed_smiles,
            beam_width=self._beam_width,
            max_length=self._max_decoder_seq_length,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            return_top_n=self._return_top_n
        )

        all_predicted_smiles = []
        all_probabilities = []

        for idx, (predicted_seqs, seq_scores) in enumerate(zip(predicted_seqs_list, seq_scores_list)):
            # Convert token sequences to SMILES strings
            predicted_smiles = self._smiles_tokenizer.sequences_to_texts(
                sequences=predicted_seqs,
                is_input_sequence=False
            )

            # Convert negative log probabilities to probabilities
            # First, convert seq_scores to a numpy array
            seq_scores = np.array(seq_scores)

            # Since seq_scores are negative log probabilities, we can compute probabilities as `prob = exp(-score)`
            exp_neg_scores = np.exp(-seq_scores)
            # Normalize probabilities
            probabilities = exp_neg_scores / np.sum(exp_neg_scores)

            valid_smiles = []
            valid_probs = []
            for smiles_string, prob in zip(predicted_smiles, probabilities):
                cleaned_smiles = self._clean_sequence(
                    smiles_string,
                    self._smiles_tokenizer.start_token,
                    self._smiles_tokenizer.end_token
                )
                reactant_smiles_list = cleaned_smiles.split('.')
                is_valid = all(self._is_valid_smiles(smi) for smi in reactant_smiles_list)
                if is_valid:
                    valid_smiles.append(cleaned_smiles)
                    valid_probs.append(float(prob))
                else:
                    self._logger.warning(f"Invalid SMILES generated: {cleaned_smiles}")

            self._logger.debug(f"Valid predicted SMILES for molecule {idx}: {valid_smiles}")

            all_predicted_smiles.append(valid_smiles)
            all_probabilities.append(valid_probs)

        return all_predicted_smiles, all_probabilities

    @staticmethod
    def _is_valid_smiles(smiles: str) -> bool:
        """
        Checks if a SMILES string is valid.

        Parameters
        ----------
        smiles : str
            The SMILES string to validate.

        Returns
        -------
        bool
            True if the SMILES string is valid, False otherwise.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    @staticmethod
    def _clean_sequence(sequence: str, start_token: str, end_token: str) -> str:
        """
        Cleans a tokenized sequence by removing start and end tokens and whitespace.

        Parameters
        ----------
        sequence : str
            The tokenized sequence to clean.
        start_token : str
            The start token to remove.
        end_token : str
            The end token to remove.

        Returns
        -------
        str
            The cleaned sequence.
        """
        if sequence.startswith(start_token + ' '):
            sequence = sequence[len(start_token) + 1:]
        end_idx = sequence.find(' ' + end_token)
        if end_idx != -1:
            sequence = sequence[:end_idx]
        sequence = sequence.replace(start_token, '').replace(end_token, '')
        sequence = sequence.replace(' ', '')
        return sequence.strip()

    def reset_cache(self) -> None:
        """
        Resets any internal caches used by the expansion strategy.

        This method is a placeholder to comply with the interface and does nothing
        in this implementation.
        """
        pass
