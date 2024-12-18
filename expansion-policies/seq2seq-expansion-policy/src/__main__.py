import os
import logging

from data.utils.open_reaction_database_extractor import OpenReactionDatabaseExtractor

def main():
    logging.basicConfig(level=logging.INFO)

    ord_extractor = OpenReactionDatabaseExtractor(
        ord_data_dir=os.environ.get('ORD_RAW_DATA_DIR'),
        db_path=os.environ.get('SQLITE_DB_PATH')
    )

    ord_extractor.write_reactions_to_files(
        reactants_smiles_path=os.environ.get('ORD_PROCESSED_REACTANTS_PATH'),
        products_smiles_path=os.environ.get('ORD_PROCESSED_PRODUCTS_PATH')
    )

    with open(os.environ.get('ORD_PROCESSED_REACTANTS_PATH'), 'r') as file:
        reactant_smiles_list = [line.strip() for line in file if line.strip()]

    with open(os.environ.get('ORD_PROCESSED_PRODUCTS_PATH'), 'r') as file:
        product_smiles_list = [line.strip() for line in file if line.strip()]

    print(f'Reactant SMILES Length: {len(reactant_smiles_list)}')
    print(f'Product SMILES Length: {len(product_smiles_list)}')

if __name__ == "__main__":
    main()
