import Levenshtein
from rdkit import Chem, RDLogger


class SmilesStringMetrics:
    @staticmethod
    def smiles_exact_match(target_smiles, predicted_smiles) -> float:
        if not target_smiles:
            return 0.0

        exact_matches = sum(1 for ref, hyp in zip(target_smiles, predicted_smiles) if ref == hyp)
        return exact_matches / len(target_smiles)

    @staticmethod
    def chemical_validity(predicted_smiles) -> float:
        if not predicted_smiles:
            return 0.0

        # Suppress RDKit error messages. Invalid SMILES errors overload logs early in training.
        RDLogger.DisableLog('rdApp.error')

        try:
            valid_predictions = sum(1 for hyp in predicted_smiles if Chem.MolFromSmiles(hyp))
        finally:
            RDLogger.EnableLog('rdApp.error')

        return valid_predictions / len(predicted_smiles)

    @staticmethod
    def levenshtein_distance(target_smiles, predicted_smiles) -> float:
        if not target_smiles:
            return 0.0

        total_distance = sum(Levenshtein.distance(ref, hyp) for ref, hyp in zip(target_smiles, predicted_smiles))
        return total_distance / len(target_smiles)
