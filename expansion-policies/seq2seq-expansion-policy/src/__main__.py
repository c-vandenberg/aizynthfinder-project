import logging
import pydevd_pycharm

from aizynthfinder.aizynthfinder import AiZynthFinder

# Configure logging to display debug information
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("aizynthfinder")
logger.setLevel(logging.DEBUG)

pydevd_pycharm.settrace('localhost', port=63342, stdoutToServer=True, stderrToServer=True)

def main():
    # Path to the configuration file
    config_file = 'src/config.yml'

    finder = AiZynthFinder(configfile=config_file)
    finder.expansion_policy.select("seq2seq_policy")
    finder.target_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

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

if __name__ == '__main__':
    main()