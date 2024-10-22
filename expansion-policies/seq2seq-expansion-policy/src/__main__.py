import logging

import pydevd_pycharm
from aizynthfinder.aizynthfinder import AiZynthFinder

from trainers.trainer import Trainer

pydevd_pycharm.settrace('localhost', port=63342, stdoutToServer=True, stderrToServer=True, suspend=False)

def main():
    # Configure logging to display debug information
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("aizynthfinder")
    logger.setLevel(logging.DEBUG)

    # Path to the configuration file
    config_file = 'src/config.yml'

    finder = AiZynthFinder(configfile=config_file)
    finder.expansion_policy.select("seq2seq_policy")
    finder.target_smiles = "O=C1NC(=O)CCC1N3C(=O)c2cccc(c2C3)N"  # Lenalidomide (Revlimid)

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