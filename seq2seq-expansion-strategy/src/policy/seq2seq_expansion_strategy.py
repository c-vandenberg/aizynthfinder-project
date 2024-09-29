import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, Attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem
from aizynthfinder.context.policy.expansion_strategies import ExpansionStrategy
from aizynthfinder.chem import SmilesBasedRetroReaction, TemplatedRetroReaction
from aizynthfinder.context.policy.utils import _make_fingerprint
from aizynthfinder.utils.exceptions import PolicyException
from aizynthfinder.utils.logging import logger
from aizynthfinder.utils.models import load_model
from aizynthfinder.chem import TreeMolecule
from aizynthfinder.chem.reaction import RetroReaction
from aizynthfinder.context.config import Configuration
from aizynthfinder.utils.type_utils import Any, Dict, List, Optional, Sequence, StrDict, Tuple
from data.utils.tokenization import SmilesTokenizer


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

        model = kwargs["model"]
        self.use_remote_models: bool = bool(kwargs.get("use_remote_models", False))
        self._logger.info(
            f"Loading Seq2Seq expansion policy model from {model} to {self.key}"
        )

        # Load your Seq2Seq model
        self.model = load_model(model, self.key, self.use_remote_models)
        self.smiles_tokenizer = SmilesTokenizer()

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

        for mol in molecules:
            smiles_representation = mol.smiles
            tokenized_smiles = self.smiles_tokenizer.tokenize(smiles_representation)

            self._logger.debug(f"Generating retrosynthesis for molecules: {smiles_representation}")

            # Use the Seq2Seq model to predict the reactants
            predicted_reactants, predicted_probabilities = self.model.predict(tokenized_smiles)

            for reactants, prob in zip(predicted_reactants, predicted_probabilities):
                reactants_str = ".".join(reactants)
                metadata = {
                    "policy_probability": float(prob),
                    "policy_name": self.key,
                    "reactants": reactants_str,
                }
                possible_actions.append(
                    SmilesBasedRetroReaction(
                        mol,
                        metadata=metadata,
                        reactants_str=reactants_str,
                    )
                )
                priors.append(prob)

        return possible_actions, priors
