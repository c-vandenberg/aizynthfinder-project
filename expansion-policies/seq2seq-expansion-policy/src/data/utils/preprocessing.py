from typing import List, Union

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rdkit import Chem
from rdkit.Chem import AllChem

from data.utils.tokenization import SmilesTokenizer


class SmilesDataPreprocessor:
    """
    Preprocesses tokenized SMILES strings into padded sequences of integers.

    Parameters
    ----------
    tokenizer : Tokenizer
        An instance of a tokenizer that can convert text to sequences.
    max_seq_length : int
        The maximum sequence length for padding.

    Methods
    -------
    preprocess_smiles(tokenized_smiles_list)
        Converts tokenized SMILES strings into padded integer sequences.
    """
    def __init__(
        self,
        smiles_tokenizer: SmilesTokenizer,
        max_seq_length: int
    ) -> None:
        self.smiles_tokenizer = smiles_tokenizer
        self.max_seq_length = max_seq_length

    def preprocess_smiles(self, tokenized_smiles_list: List[str]) -> tf.Tensor:
        """
        Converts tokenized SMILES strings into padded sequences of integers.

        Parameters
        ----------
        tokenized_smiles_list : List[str]
            A list of tokenized SMILES strings.

        Returns
        -------
        tf.Tensor
            A tensor of padded sequences of shape (num_sequences, max_seq_length).
        """
        sequences = self.smiles_tokenizer.texts_to_sequences(tokenized_smiles_list)

        # Pad sequences
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_seq_length,
            padding='post',
            truncating='post'
        )
        return tf.constant(padded_sequences, dtype=tf.int32)

    @staticmethod
    def is_canonical(smiles: str) -> bool:
        """
        Checks if a given SMILES string is canonical.

        Parameters
        ----------
        smiles : str
            The SMILES string to check.

        Returns
        -------
        bool
            True if the SMILES is canonical, False otherwise.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES: {smiles}")
                return False

            # Generate canonical SMILES from the molecule
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            # Compare with the original SMILES
            return smiles == canonical_smiles
        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            return False

    @staticmethod
    def canonicalize_smiles(smiles: str) -> Union[str, None]:
        """
        Converts a SMILES string to its canonical form.

        Parameters
        ----------
        smiles : str
            The SMILES string to canonicalize.

        Returns
        -------
        str
            The canonical SMILES string, or None if invalid.
        """
        try:
            # Parse the SMILES string into a molecule object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES: {smiles}")
                return None

            # Generate the canonical SMILES
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            return canonical_smiles
        except Exception as e:
            print(f"Error canonicalizing SMILES '{smiles}': {e}")
            return None