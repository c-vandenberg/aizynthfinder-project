#!/usr/bin/env python3

import os
import argparse
from typing import List

from data.utils.logging_utils import configure_logger
from data.utils.file_utils import load_smiles_from_file
from data.utils.preprocessing import SmilesDataPreprocessor
from data.utils.database_utils import get_unique_count

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Combine two reactants SMILES datasets together, and two products SMILES datasets together."
    )
    parser.add_argument(
        '--reactants_a_filepath',
        type=str,
        required=True,
        help='File path for first reactants SMILES dataset.'
    )
    parser.add_argument(
        '--reactants_b_filepath',
        type=str,
        required=True,
        help='File path for second reactants SMILES dataset.'
    )
    parser.add_argument(
        '--products_a_filepath',
        type=str,
        required=True,
        help='File path for first products SMILES dataset.'
    )
    parser.add_argument(
        '--products_b_filepath',
        type=str,
        required=True,
        help='File path for second products SMILES dataset.'
    )
    parser.add_argument(
        '--concat_reactants_output_path',
        type=str,
        required=True,
        help='Output path for combined reactant SMILES.'
    )
    parser.add_argument(
        '--concat_products_output_path',
        type=str,
        required=True,
        help='Output path for combined product SMILES.'
    )
    parser.add_argument(
        '--sqlite3_db_path',
        type=str,
        default='data/database/concatenated-smiles-datasets/sqlite3/seen_pairs.db',
        help='Path to SQLite DB.'
    )
    parser.add_argument(
        '--script_log_path',
        type=str,
        default='var/log/concatenated-smiles-datasets/concatenate-smiles-datasets.log',
        help='Path log file.'
    )

    return parser.parse_args()

def main():
    # Parse command-line arguments, validate paths and configure logger
    args = parse_arguments()

    concat_reactants_output_path: str = args.concat_reactants_output_path
    concat_products_output_path: str = args.concat_products_output_path
    reactants_a_file_path: str = args.reactants_a_filepath
    reactants_b_file_path: str = args.reactants_b_filepath
    products_a_file_path: str = args.products_a_filepath
    products_b_file_path: str = args.products_b_filepath
    sqlite3_db_path: str = args.sqlite3_db_path
    log_path: str = args.script_log_path

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = configure_logger(log_path=log_path)

    file_paths: List = [
        reactants_a_file_path,
        reactants_b_file_path,
        products_a_file_path,
        products_b_file_path
    ]

    for file_path in file_paths:
        if not os.path.dirname(file_path):
            logger.error(f"SMILES dataset file directory does not exist: {file_path}")
            return

    os.makedirs(os.path.dirname(concat_reactants_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(concat_products_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(sqlite3_db_path), exist_ok=True)

    reactants_smiles_a: List[List[str]] = [
        reactant.split('.') for reactant in load_smiles_from_file(reactants_a_file_path)
    ]
    products_smiles_a: List[List[str]] = [
        product.split('.') for product in load_smiles_from_file(products_a_file_path)
    ]

    reactants_smiles_b: List[List[str]] = [
        reactant.split('.') for reactant in load_smiles_from_file(reactants_b_file_path)
    ]
    products_smiles_b: List[List[str]] = [
        product.split('.') for product in load_smiles_from_file(products_b_file_path)
    ]

    # Initialise SMILES preprocessor
    try:
        smiles_preprocessor = SmilesDataPreprocessor(
            products_smiles=products_smiles_a,
            reactants_smiles=reactants_smiles_a,
            logger=logger
        )
        logger.debug("Initialised SmilesDataPreprocessor.")
    except Exception as e:
        logger.exception(f"Failed to initialise SmilesDataPreprocessor: {e}")
        return

    # Concatenate datasets
    try:
        smiles_preprocessor.concatenate_datasets(
            products_smiles_lists=products_smiles_b,
            reactants_smiles_lists=reactants_smiles_b
        )
        logger.debug("Concatenation process completed.")
    except Exception as e:
        logger.exception(f"An error occurred during concatenation: {e}")
        return

    # Remove duplicate reactions
    try:
        smiles_preprocessor.remove_duplicate_product_reactant_pairs(
            db_path=sqlite3_db_path,
            batch_size=1000,
            log_interval=1000
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
            reactants_smiles_path=concat_reactants_output_path,
            products_smiles_path=concat_products_output_path
        )
        logger.debug("Unique reactions written to files.")
    except Exception as e:
        logger.exception(f"An error occurred while writing reactions to files: {e}")
        return

    # Get number of unique reactions in database
    try:
        db_unique_count = get_unique_count(sqlite3_db_path)
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
