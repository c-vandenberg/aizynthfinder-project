import os
import logging
import sqlite3
import argparse

from data.utils.preprocessing import SmilesDataPreprocessor
from data.utils.open_reaction_database_extractor import OpenReactionDatabaseExtractor
from data.utils.database_utils import get_unique_count

def configure_logger():
    """
        Configures and returns a module-specific logger.

        Returns:
            logging.Logger: Configured logger.
        """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Prevent adding multiple handlers if the logger already has handlers
    if not logger.handlers:
        # Console handler for INFO and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # File handler for DEBUG and above
        ord_log_path: str = 'ord_database_extraction.log' if not os.environ.get('ORD_EXTRACTION_LOG_PATH') else os.environ.get('ORD_EXTRACTION_LOG_PATH')
        file_handler = logging.FileHandler(ord_log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Extract and deduplicate ORD reactions.")
    parser.add_argument('--ord_data_dir', type=str, required=True, help='Path to ORD raw data directory.')
    parser.add_argument('--db_path', type=str, default='data/database/ord/seen_pairs.db', help='Path to SQLite DB.')
    parser.add_argument('--reactants_output_path', type=str, required=True, help='Output path for reactant SMILES.')
    parser.add_argument('--products_output_path', type=str, required=True, help='Output path for product SMILES.')
    return parser.parse_args()

def main():
    # Configure logger and parse command-line arguments
    logger = configure_logger()
    args = parse_arguments()

    ord_data_dir = args.ord_data_dir
    db_path = args.db_path
    reactants_output_path = args.reactants_output_path
    products_output_path = args.products_output_path

    if not os.path.isdir(ord_data_dir):
        logger.error(f"ORD data directory does not exist: {ord_data_dir}")
        return

    # Initialise SMILES preprocessor
    try:
        smiles_preprocessor = SmilesDataPreprocessor()
        logger.debug("Initialized SmilesDataPreprocessor.")
    except Exception as e:
        logger.exception(f"Failed to initialize SmilesDataPreprocessor: {e}")
        return

    # Initialize ORD reaction extractor
    try:
        ord_extractor = OpenReactionDatabaseExtractor(
            ord_data_dir=ord_data_dir,
            smiles_preprocessor=smiles_preprocessor
        )
        logger.debug("Initialized OpenReactionDatabaseExtractor.")
    except Exception as e:
        logger.exception(f"Failed to initialize OpenReactionDatabaseExtractor: {e}")
        return

    # Extract reactions
    logger.info("Extracting reactions from ORD dataset.")
    try:
        for reactant_smiles, product_smiles in ord_extractor.extract_all_reactions():
            smiles_preprocessor.reactants_smiles.append(reactant_smiles)
            smiles_preprocessor.products_smiles.append(product_smiles)
    except Exception as e:
        logger.exception(f"An error occurred during reaction extraction: {e}")
        return

    logger.info(f"Total reactions extracted: {len(smiles_preprocessor.products_smiles)}")

    # Remove duplicate reactions
    try:
        smiles_preprocessor.remove_duplicate_product_reactant_pairs(
            db_path=db_path
        )
        logger.debug("Deduplication process completed.")
    except Exception as e:
        logger.exception(f"An error occurred during deduplication: {e}")
        return

    logger.info(f"Total unique reactants after deduplication: {len(smiles_preprocessor.reactants_smiles)}")
    logger.info(f"Total unique products after deduplication: {len(smiles_preprocessor.products_smiles)}")

    # Get number of unique reactions in database
    try:
        db_unique_count = get_unique_count(db_path)
        logger.info(f"SQLite3 Database Unique Count: {db_unique_count}")
    except Exception as e:
        logger.exception(f"An error occurred while fetching unique count from the database: {e}")
        return

    # Reactant and product dataset size and unique reaction database count verification
    print(f'Reactant SMILES Dataset Size: {len(smiles_preprocessor.reactants_smiles)}')
    print(f'Product SMILES Dataset Size: {len(smiles_preprocessor.products_smiles)}')
    print(f'Sqlite3 Database Unique Count : {db_unique_count}')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    smiles_preprocessor = SmilesDataPreprocessor()

    file_handler = logging.FileHandler(os.environ.get('ORD_EXTRACTION_LOG_PATH'))
    file_handler.setLevel(logging.DEBUG)

    ord_extractor = OpenReactionDatabaseExtractor(
        ord_data_dir=os.environ.get('ORD_RAW_DATA_DIR'),
        smiles_preprocessor=smiles_preprocessor
    )

    logging.info("Extracting reactions from ORD dataset.")
    for reactant_smiles, product_smiles in ord_extractor.extract_all_reactions():
        smiles_preprocessor.reactants_smiles.append(reactant_smiles)
        smiles_preprocessor.products_smiles.append(product_smiles)

    logging.info(f"Total reactions extracted: {len(smiles_preprocessor.products_smiles)}")

    smiles_preprocessor.remove_duplicate_product_reactant_pairs(
        db_path=os.environ.get('SQLITE_DB_PATH')
    )

    logging.info(f"Total unique reactants after deduplication: {len(smiles_preprocessor.reactants_smiles)}")
    logging.info(f"Total unique products after deduplication: {len(smiles_preprocessor.products_smiles)}")

    # Write unique reactions to files
    smiles_preprocessor.write_reactions_to_files(
        reactants_smiles_path=os.environ.get('ORD_PROCESSED_REACTANTS_PATH'),
        products_smiles_path=os.environ.get('ORD_PROCESSED_PRODUCTS_PATH')
    )


    with open(os.environ.get('ORD_PROCESSED_REACTANTS_PATH'), 'r') as file:
        reactant_smiles_list = [line.strip() for line in file if line.strip()]

    with open(os.environ.get('ORD_PROCESSED_PRODUCTS_PATH'), 'r') as file:
        product_smiles_list = [line.strip() for line in file if line.strip()]

    print(f'Reactant SMILES Length: {len(reactant_smiles_list)}')
    print(f'Product SMILES Length: {len(product_smiles_list)}')

    db_unique_count = get_unique_count(os.environ.get('SQLITE_DB_PATH'))

    print(f'Sqlite3 Database Unique Count : {db_unique_count}')

if __name__ == "__main__":
    main()
