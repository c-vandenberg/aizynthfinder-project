from typing import List

import Levenshtein
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

class SmilesStringMetrics:
    @staticmethod
    def smiles_exact_match(target_smiles: List[str], predicted_smiles: List[str]) -> float:
        """
       Compute the exact match accuracy between target and predicted SMILES.

       Parameters:
       ----------
       target_smiles : List[str]
           List of target SMILES strings.
       predicted_smiles : List[str]
           List of predicted SMILES strings.

       Returns:
       -------
       float
           Exact match accuracy as a proportion.
       """
        if not target_smiles or not predicted_smiles:
            return 0.0

        exact_matches = sum(1 for ref, hyp in zip(target_smiles, predicted_smiles) if ref == hyp)
        return exact_matches / len(target_smiles) if len(target_smiles) > 0 else 0.0

    @staticmethod
    def chemical_validity(predicted_smiles: List[str]) -> float:
        """
        Compute the chemical validity score for predicted SMILES.

        Parameters:
        ----------
        predicted_smiles : List[str]
            List of predicted SMILES strings.

        Returns:
        -------
        float
            Proportion of valid SMILES.
        """
        if not predicted_smiles:
            return 0.0

        # Suppress RDKit error messages. Invalid SMILES errors overload logs early in training.
        RDLogger.DisableLog('rdApp.error')

        try:
            valid_predictions = sum(
                1 for hyp in predicted_smiles if SmilesStringMetrics.is_valid_smiles(hyp)
            )
        finally:
            RDLogger.EnableLog('rdApp.error')

        return valid_predictions / len(predicted_smiles)

    @staticmethod
    def tanimoto_coefficient(smiles1: str, smiles2: str) -> float:
        """
        Compute the Tanimoto similarity between two SMILES strings.

        Parameters:
        ----------
        smiles1 : str
            First SMILES string.
        smiles2 : str
            Second SMILES string.

        Returns:
        -------
        float
            Tanimoto similarity score between 0.0 and 1.0. Returns 0.0 if either SMILES is invalid.
        """
        # Suppress RDKit error messages. Invalid SMILES errors overload logs early in training.
        RDLogger.DisableLog('rdApp.error')
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                return 0.0

            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        finally:
            RDLogger.EnableLog('rdApp.error')

        return DataStructs.TanimotoSimilarity(fp1, fp2)

    @staticmethod
    def average_tanimoto_similarity(target_smiles: List[str], predicted_smiles: List[str]) -> float:
        """
        Compute the average Tanimoto similarity across all SMILES pairs.

        Parameters:
        ----------
        target_smiles : List[str]
            List of target SMILES strings.
        predicted_smiles : List[str]
            List of predicted SMILES strings.

        Returns:
        -------
        float
            Average Tanimoto similarity percentage.
        """
        if not target_smiles or not predicted_smiles:
            return 0.0

        similarity_scores = []
        for ref_smiles, pred_smiles in zip(target_smiles, predicted_smiles):
            similarity = SmilesStringMetrics.tanimoto_coefficient(ref_smiles, pred_smiles)
            similarity_scores.append(similarity)

        average_similarity = (sum(similarity_scores) / len(similarity_scores)) if similarity_scores else 0.0
        return average_similarity

    @staticmethod
    def levenshtein_distance(target_smiles: List[str], predicted_smiles: List[str]) -> float:
        """
        Compute the average Levenshtein distance between target and predicted SMILES.

        Parameters:
        ----------
        target_smiles : List[str]
            List of target SMILES strings.
        predicted_smiles : List[str]
            List of predicted SMILES strings.

        Returns:
        -------
        float
            Average Levenshtein distance.
        """
        if not target_smiles or not predicted_smiles:
            return 0.0

        total_distance = sum(Levenshtein.distance(ref, hyp) for ref, hyp in zip(target_smiles, predicted_smiles))
        return total_distance / len(target_smiles)

    @staticmethod
    def is_valid_smiles(smiles: str) -> bool:
        """
        Check if a SMILES string is chemically valid.

        Parameters:
        ----------
        smiles : str
            SMILES string to validate.

        Returns:
        -------
        bool
            True if valid, False otherwise.
        """
        # Suppress RDKit error messages. Invalid SMILES errors overload logs early in training.
        RDLogger.DisableLog('rdApp.error')

        try:
            is_valid: bool = Chem.MolFromSmiles(smiles) is not None
        finally:
            RDLogger.EnableLog('rdApp.error')
        return is_valid