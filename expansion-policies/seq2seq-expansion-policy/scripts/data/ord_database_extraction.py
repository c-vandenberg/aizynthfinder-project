#!/usr/bin/env python3

import os
import argparse

from data.utils.logging_utils import configure_logger
from data.utils.preprocessing import SmilesDataPreprocessor
from data.utils.open_reaction_database_extractor import OpenReactionDatabaseExtractor
from data.utils.database_utils import get_unique_count


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Extract and deduplicate ORD reactions.")
    parser.add_argument('--ord_data_dir', type=str, required=True, help='Path to ORD raw data directory.')
    parser.add_argument('--reactants_output_path', type=str, required=True, help='Output path for reactant SMILES.')
    parser.add_argument('--products_output_path', type=str, required=True, help='Output path for product SMILES.')
    parser.add_argument('--sqlite_db_path', type=str, default='data/database/ord/sqlite3/seen_pairs.db', help='Path to SQLite DB.')
    parser.add_argument('--ord_log_path', type=str, default='var/log/ord-reaction-extraction/ord_reaction_extraction.log', help='Path log file.')
    return parser.parse_args()


def main():
    # Parse command-line arguments, validate paths and configure logger
    args = parse_arguments()

    ord_data_dir: str = args.ord_data_dir
    reactants_output_path: str = args.reactants_output_path
    products_output_path: str = args.products_output_path
    sqlite_db_path: str = args.sqlite_db_path
    ord_log_path: str = args.ord_log_path

    os.makedirs(os.path.dirname(ord_log_path), exist_ok=True)
    logger = configure_logger(log_path=ord_log_path)

    if not os.path.isdir(ord_data_dir):
        logger.error(f"ORD data directory does not exist: {ord_data_dir}")
        return

    os.makedirs(os.path.dirname(reactants_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(products_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)

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
            db_path=sqlite_db_path
        )
        logger.debug("Deduplication process completed.")
    except Exception as e:
        logger.exception(f"An error occurred during deduplication: {e}")
        return

    logger.info(f"Total unique reactants after deduplication: {len(smiles_preprocessor.reactants_smiles)}")
    logger.info(f"Total unique products after deduplication: {len(smiles_preprocessor.products_smiles)}")

    # Write unique reactions to files
    try:
        smiles_preprocessor.write_reactions_to_files(
            reactants_smiles_path=reactants_output_path,
            products_smiles_path=products_output_path
        )
        logger.debug("Unique reactions written to files.")
    except Exception as e:
        logger.exception(f"An error occurred while writing reactions to files: {e}")
        return

    # Get number of unique reactions in database
    try:
        db_unique_count = get_unique_count(sqlite_db_path)
        logger.info(f"SQLite3 Database Unique Count: {db_unique_count}")
    except Exception as e:
        logger.exception(f"An error occurred while fetching unique count from the database: {e}")
        return

    # Reactant and product dataset size and unique reaction database count verification
    print(f'Reactant SMILES Dataset Size: {len(smiles_preprocessor.reactants_smiles)}')
    print(f'Product SMILES Dataset Size: {len(smiles_preprocessor.products_smiles)}')
    print(f'Sqlite3 Database Unique Count : {db_unique_count}')

if __name__ == "__main__":
    main()