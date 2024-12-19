import os
import logging

from data.utils.preprocessing import SmilesDataPreprocessor
from data.utils.open_reaction_database_extractor import OpenReactionDatabaseExtractor

def main():
    logging.basicConfig(level=logging.INFO)

    smiles_preprocessor = SmilesDataPreprocessor()

    ord_extractor = OpenReactionDatabaseExtractor(
        ord_data_dir=os.environ.get('ORD_RAW_DATA_DIR'),
        smiles_preprocessor=smiles_preprocessor
    )

    logging.info("Extracting reactions from ORD dataset.")
    smiles_preprocessor.reactants_smiles = []
    smiles_preprocessor.products_smiles = []
    for reactant_line, product_line in ord_extractor.extract_all_reactions():
        reactant_smiles = reactant_line.split('.') if reactant_line else []
        product_smiles = product_line.split('.') if product_line else []

        smiles_preprocessor.reactants_smiles.append(reactant_smiles)
        smiles_preprocessor.products_smiles.append(product_smiles)

    logging.info(f"Total reactions extracted: {len(smiles_preprocessor.products_smiles)}")

    smiles_preprocessor.remove_duplicate_product_reactant_pairs(
        db_path=os.environ.get('SQLITE_DB_PATH'),
        batch_size=10000,
        log_interval=100000
    )

    logging.info(f"Total unique reactants after deduplication: {len(smiles_preprocessor.reactants_smiles)}")
    logging.info(f"Total unique products after deduplication: {len(smiles_preprocessor.products_smiles)}")

    # Write unique reactions to files
    smiles_preprocessor.write_reactions_to_files(
        reactants_smiles_path=os.environ.get('ORD_PROCESSED_REACTANTS_PATH'),
        products_smiles_path=os.environ.get('ORD_PROCESSED_PRODUCTS_PATH'),
        log_interval=100000
    )


    with open(os.environ.get('ORD_PROCESSED_REACTANTS_PATH'), 'r') as file:
        reactant_smiles_list = [line.strip() for line in file if line.strip()]

    with open(os.environ.get('ORD_PROCESSED_PRODUCTS_PATH'), 'r') as file:
        product_smiles_list = [line.strip() for line in file if line.strip()]

    print(f'Reactant SMILES Length: {len(reactant_smiles_list)}')
    print(f'Product SMILES Length: {len(product_smiles_list)}')

if __name__ == "__main__":
    main()
