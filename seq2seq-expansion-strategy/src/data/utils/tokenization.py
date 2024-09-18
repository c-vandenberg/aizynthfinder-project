import re
from tensorflow.keras.preprocessing.text import Tokenizer


class SmilesTokenizer:
    def __init__(self, start_token='<START>', end_token='<END>'):
        self.start_token = start_token
        self.end_token = end_token
        self.tokenizer = None  # Will be initialized later if needed

    def tokenize(self, smiles):
        """
        Tokenize a single SMILES string.

        Args:
            smiles (str): A SMILES string.

        Returns:
            List[str]: A list of tokens.
        """
        pattern = r"(\[.*?\])|Cl?|Br?|Si|@@?|==?|[B-NOP-Zb-nop-z0-9]|\S"
        tokens = re.findall(pattern, smiles)
        tokens = [self.start_token] + tokens + [self.end_token]
        return tokens

    def tokenize_list(self, smiles_list):
        """
        Tokenize a list of SMILES strings.

        Args:
            smiles_list (List[str]): A list of SMILES strings.

        Returns:
            List[List[str]]: A list of token lists.
        """
        return [self.tokenize(smiles) for smiles in smiles_list]

    def create_tokenizer(self, tokenized_smiles_list):
        """
        Create and fit a Keras tokenizer on the tokenized SMILES list.

        Args:
            tokenized_smiles_list (List[List[str]]): A list of tokenized SMILES strings.
        """

        self.tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(tokenized_smiles_list)
        return self.tokenizer
