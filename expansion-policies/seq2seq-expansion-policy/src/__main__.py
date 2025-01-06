import os

from data.utils.file_utils import load_smiles_from_file

def main():
    dataset = '/home/chris-vdb/Computational-Chemistry/aizynthfinder-project/expansion-policies/seq2seq-expansion-policy/data/preprocessed/concatenated-smiles/products_smiles'

    products_smiles = load_smiles_from_file(dataset)

    print(f'{len(products_smiles)}')

if __name__ == "__main__":
    main()
