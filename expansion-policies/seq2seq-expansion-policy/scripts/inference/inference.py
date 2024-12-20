#!/usr/bin/env python3

import logging
import argparse

from aizynthfinder.aizynthfinder import AiZynthFinder

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Perform AiZynthFinder retrosynthetic analysis on product SMILES.")
    parser.add_argument(
        '--product_smiles',
        type=str,
        required=True,
        help='SMILES string of product for retrosynthetic analysis.'
    )
    parser.add_argument(
        '--expansion_policy',
        type=str,
        required=True,
        help='Expansion policy to use.'
    )
    parser.add_argument(
        '--inference_config_file_path',
        type=str,
        required=True,
        help='Path to inference configuration file.'
    )

    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Configure AiZynthFinder logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("aizynthfinder")
    logger.setLevel(logging.DEBUG)

    finder = AiZynthFinder(configfile=args.inference_config_file_path)
    finder.expansion_policy.select(args.expansion_policy)
    finder.target_smiles = args.product_smiles

    # Prepare the search tree
    finder.prepare_tree()

    # Run the tree search
    finder.tree_search()

    # Build the synthesis routes
    finder.build_routes()

    # Extract statistics if needed
    stats = finder.extract_statistics()

    # Print the number of routes found
    print(f"Number of routes found: {len(finder.routes)}")

if __name__ == "__main__":
    main()