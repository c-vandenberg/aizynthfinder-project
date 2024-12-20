import os
from typing import List


def load_smiles_from_file(file_path: str) -> List[str]:
    """
    Loads SMILES strings from a specified file.

    Reads a file containing SMILES strings, ensuring that each line is
    properly stripped of whitespace and non-empty.

    Parameters
    ----------
    file_path : str
        The path to the file containing SMILES strings.

    Returns
    -------
    List[str]
        A list of SMILES strings.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r') as file:
        smiles_list = [line.strip() for line in file if line.strip()]

    return smiles_list