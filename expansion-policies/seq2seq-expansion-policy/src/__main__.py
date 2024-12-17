import os
import logging

from data.utils.open_reaction_database_extractor import OpenReactionDatabaseExtractor

def main():
    logging.basicConfig(level=logging.INFO)

    ord_extractor = OpenReactionDatabaseExtractor(ord_data_dir=os.environ.get('ORD_RAW_DATA_DIR'))

    extract_reactions_generator = ord_extractor.write_reactions_to_files(
        reactants_smiles_path=os.environ.get('ORD_PROCESSED_REACTANTS_PATH'),
        products_smiles_path=os.environ.get('ORD_PROCESSED_PRODUCTS_PATH')
    )

    for reaction in extract_reactions_generator:
        print(reaction)

if __name__ == "__main__":
    main()
