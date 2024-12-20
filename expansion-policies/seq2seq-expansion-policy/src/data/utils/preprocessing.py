import os
import sqlite3
import logging
from typing import List, Tuple, Optional, Union

import tensorflow as tf
from rdkit import Chem
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data.utils.tokenization import SmilesTokenizer

logging.basicConfig(level=logging.INFO)

class SmilesDataPreprocessor:
    def __init__(
        self,
        products_smiles: Optional[List[List[str]]] = None,
        reactants_smiles: Optional[List[List[str]]] = None,
    ):
        """
        Initialize the SmilesDataPreprocessor with product and reactant SMILES lists.

        Parameters
        ----------
        products_smiles : List[List[str]], optional
            List containing lists of product SMILES strings for each reaction.
            If None, initializes as an empty list.
        reactants_smiles : List[List[str]], optional
            List containing lists of reactant SMILES strings for each reaction.
            If None, initializes as an empty list.
        """
        self.products_smiles: List[List[str]] = products_smiles if products_smiles is not None else []
        self.reactants_smiles: List[List[str]] = reactants_smiles if reactants_smiles is not None else []

        self._logger = logging.getLogger(__name__)

    def concatenate_datasets(
        self,
        products_smiles_lists: Optional[List[List[str]]] = None,
        reactants_smiles_lists: Optional[List[List[str]]] = None
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Concatenate additional product and reactant SMILES lists to the existing datasets.

        Parameters
        ----------
        products_smiles_lists : List[List[str]], optional
            Additional product SMILES lists to concatenate. Defaults to None.
        reactants_smiles_lists : List[List[str]], optional
            Additional reactant SMILES lists to concatenate. Defaults to None.

        Returns
        -------
        Tuple[List[List[str]], List[List[str]]]:
            The updated products and reactants SMILES lists after concatenation.
        """
        if products_smiles_lists is not None:
            self._logger.info("Concatenating product datasets.")
            self._logger.info(f"Total products before concatenation: {len(self.products_smiles)}")
            self.products_smiles.extend(products_smiles_lists)
            self._logger.info(f"Total products after concatenation: {len(self.products_smiles)}")

        if reactants_smiles_lists is not None:
            self._logger.info("Concatenating reactant datasets.")
            self._logger.info(f"Total reactants before concatenation: {len(self.reactants_smiles)}")
            self.reactants_smiles.extend(reactants_smiles_lists)
            self._logger.info(f"Total reactants after concatenation: {len(self.reactants_smiles)}")

        return self.products_smiles, self.reactants_smiles

    def write_reactions_to_files(
        self,
        reactants_smiles_path: str,
        products_smiles_path: str,
        log_interval: int = 100000
    )-> None:
        """
        Write reactant and product SMILES to specified files.

        Parameters
        ----------
        reactants_smiles_path : str
            Path to the output reactants SMILES file.
        products_smiles_path : str
            Path to the output products SMILES file.
        log_interval : int, optional
            Interval for logging progress. Defaults to 100000 reactions.

        Returns
        -------
        None
        """
        self._validate_smiles_datasets()

        self._logger.info("Starting writing reactions to files.")

        os.makedirs(os.path.dirname(reactants_smiles_path), exist_ok=True)
        os.makedirs(os.path.dirname(products_smiles_path), exist_ok=True)

        with open(reactants_smiles_path, 'w') as reactants_file, \
                open(products_smiles_path, 'w') as products_file:

            total = len(self.products_smiles)

            for idx, (product_smiles, reactant_smiles) in enumerate(zip(self.products_smiles, self.reactants_smiles), 1):
                reactant_line = '.'.join(reactant_smiles) if reactant_smiles else ''
                product_line = '.'.join(product_smiles) if product_smiles else ''
                cleaned_product_line = self.remove_non_product_fragment_smiles(product_line)

                reactants_file.write(reactant_line + '\n')
                products_file.write(cleaned_product_line + '\n')

                if idx % log_interval == 0:
                    self._logger.info(f"Written {idx}/{total} reactions to files.")

        self._logger.info("Writing reactions to files completed successfully.")

    def remove_duplicate_product_reactant_pairs(
        self,
        db_path: str = 'seen_pairs.db',
        batch_size: int = 10000,
        log_interval: int = 100000
    )-> None:
        """
        Remove duplicate (reactant, product) pairs using SQLite and return unique datasets.

        Parameters
        ----------
        db_path : str, optional
            Path to the SQLite database used for deduplication. Defaults to 'seen_pairs.db'.
        batch_size : int, optional
            Number of reactions to process in each batch. Defaults to 10000.
        log_interval : int, optional
            Interval for logging progress. Defaults to 100000 reactions.

        Returns
        -------
        None
        """
        self._validate_smiles_datasets()

        self._logger.info("Starting SQLite-based deduplication.")
        unique_products = []
        unique_reactants = []

        if os.path.exists(db_path):
            os.remove(db_path)
            self._logger.info(f"Existing database '{db_path}' removed for a fresh start.")

        conn = sqlite3.connect(db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pairs (
                    reactant TEXT,
                    product TEXT,
                    PRIMARY KEY (reactant, product)
                )
            """)
            cursor = conn.cursor()

            total = len(self.products_smiles)
            current_idx = 0

            while current_idx < total:
                batch_products = self.products_smiles[current_idx:current_idx + batch_size]
                batch_reactants = self.reactants_smiles[current_idx:current_idx + batch_size]

                temp_seen = set()
                sanitised_batch = []

                for product_smiles, reactant_smiles in zip(batch_products, batch_reactants):
                    try:
                        # Canonicalise/normalise SMILES
                        canonical_reactants = [self.canonicalise_smiles(smi) for smi in reactant_smiles]
                        canonical_products = [self.canonicalise_smiles(smi) for smi in product_smiles]
                    except ValueError as ve:
                        self._logger.error(str(ve))
                        continue

                    reactant_line = '.'.join(canonical_reactants).strip()
                    product_line = '.'.join(canonical_products).strip()
                    reaction_pair = (reactant_line, product_line)

                    if reaction_pair not in temp_seen:
                        temp_seen.add(reaction_pair)
                        sanitised_batch.append(reaction_pair)
                    else:
                        self._logger.debug(
                            f"Duplicate within batch skipped: Reactant={reactant_line}, Product={product_line}"
                        )

                # Process the sanitised batch
                self._process_batch(
                    batch=sanitised_batch,
                    cursor=cursor,
                    unique_products=unique_products,
                    unique_reactants=unique_reactants
                )

                current_idx += batch_size

                if current_idx % log_interval == 0:
                    self._logger.info(f"Processed {current_idx}/{total} reactions.")

            conn.commit()
            self._logger.info("SQLite-based deduplication completed successfully.")
        finally:
            conn.close()

        self.products_smiles = unique_products
        self.reactants_smiles = unique_reactants

    def deduplicate_in_memory(self) -> None:
        """
        Deduplicate reaction pairs using in-memory sets.

        Returns:
        -------
        None
        """
        self._validate_smiles_datasets()

        self._logger.info("Starting in-memory deduplication.")
        seen = set()
        unique_products = []
        unique_reactants = []

        for product_smiles, reactant_smiles in zip(self.products_smiles, self.reactants_smiles):
            try:
                # Canonicalise/normalise SMILES
                canonical_reactants = [self.canonicalise_smiles(smi) for smi in reactant_smiles]
                canonical_products = [self.canonicalise_smiles(smi) for smi in product_smiles]
            except ValueError as ve:
                self._logger.error(str(ve))
                continue

            reactant_line = '.'.join(canonical_reactants).strip()
            product_line = '.'.join(canonical_products).strip()
            reaction_pair = (reactant_line, product_line)

            if reaction_pair not in seen:
                seen.add(reaction_pair)
                unique_products.append(canonical_products)
                unique_reactants.append(canonical_reactants)
            else:
                self._logger.debug(
                    f"Duplicate within batch skipped: Reactant={reactant_line}, Product={product_line}")

        self._logger.info(f"Deduplication completed. Unique reactions: {len(unique_products)}")

        self.products_smiles = unique_products
        self.reactants_smiles = unique_reactants

    def _validate_smiles_datasets(self):
        if not self.products_smiles or not self.reactants_smiles:
            self._logger.error("Cannot remove duplicates from empty reaction dataset.")
            return None

        if not len(self.products_smiles) == len(self.reactants_smiles):
            self._logger.error("Product SMILES and reaction SMILES datasets must be the same length.")
            return

    def _process_batch(
        self,
        batch: List[Tuple[str, str]],
        cursor: sqlite3.Cursor,
        unique_products: List[List[str]],
        unique_reactants: List[List[str]]
    ) -> None:
        """
        Process a batch of reaction pairs: insert into DB and append to unique lists if unique.

        Parameters
        ----------
        batch : List[Tuple[str, str]]
            List of (reactant, product) pairs.
        cursor : sqlite3.Cursor
            SQLite cursor for executing database operations.
        unique_products : List[List[str]]
            List to append unique product SMILES.
        unique_reactants : List[List[str]]
            List to append unique reactant SMILES.

        Returns
        -------
        None
        """
        try:
            cursor.executemany(
                "INSERT INTO pairs (reactant, product) VALUES (?, ?)",
                batch
            )
            # All inserts were successful; append to unique lists
            for reactant_line, product_line in batch:
                unique_reactants.append(reactant_line.split('.'))
                unique_products.append(product_line.split('.'))
            self._logger.debug(f"Batch of {len(batch)} reactions inserted successfully.")
        except sqlite3.IntegrityError as e:
            self._logger.warning(f"IntegrityError encountered: {e}. Processing batch individually.")
            # Some inserts failed due to duplicates; handle individually
            for reactant_line, product_line in batch:
                try:
                    cursor.execute(
                        "INSERT INTO pairs (reactant, product) VALUES (?, ?)",
                        (reactant_line, product_line)
                    )
                    # Insert succeeded; append to unique lists
                    unique_reactants.append(reactant_line.split('.'))
                    unique_products.append(product_line.split('.'))
                    self._logger.debug(f"Inserted Reaction - Reactant: {reactant_line}, Product: {product_line}")
                except sqlite3.IntegrityError as e:
                    # Duplicate pair; skip appending
                    self._logger.warning(f"IntegrityError encountered: {e}. Duplicate pair skipped: Reactant={reactant_line}, Product={product_line}")
                    continue

    @staticmethod
    def canonicalise_smiles(smiles: str) -> str:
        """
        Canonicalises SMILES strings using `rdkit.Chem.MolFromSmiles()`.

        Correctly handles reactant SMILES separated by `.` by canonicalizing each separately
        and reassembling them in the same order with a `.` separator.

        Parameters
        ----------
        smiles : str
            The single SMILES string or multiple `.`-separated SMILES strings to canonicalise.

        Returns
        -------
        str
            The single canonicalised SMILES string or multiple `.`-separated canonicalised SMILES strings.

        Raises
        ------
        ValueError
            If any of the SMILES components are invalid.
        """
        smiles_components = smiles.split('.')
        canonical_components = []
        for smiles in smiles_components:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            canonical_components.append(canonical_smiles)

        # Reassemble the components in the same order, separated by '.'
        canonical_smiles = '.'.join(canonical_components)

        return canonical_smiles

    @staticmethod
    def remove_non_product_fragment_smiles(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return smiles

        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) == 1:
            return Chem.MolToSmiles(frags[0])

        main_frag = max(frags, key=lambda frag: frag.GetNumHeavyAtoms())

        return Chem.MolToSmiles(main_frag, canonical=True, isomericSmiles=True)


class TokenizedSmilesPreprocessor:
    """
    Preprocesses tokenized SMILES strings into padded sequences of integers for model training and evaluation.

    This class leverages a provided `SmilesTokenizer` to convert tokenized SMILES strings into
    integer sequences and applies padding to ensure uniform sequence lengths across the dataset.

    Attributes
    ----------
    smiles_tokenizer : SmilesTokenizer
        The tokenizer used for converting SMILES strings to integer sequences.
    max_seq_length : int
        The maximum sequence length for padding sequences.

    Methods
    -------
    preprocess_smiles(tokenized_smiles_list)
        Converts a list of tokenized SMILES strings into a tensor of padded integer sequences.
    """
    def __init__(
        self,
        smiles_tokenizer: SmilesTokenizer,
        max_seq_length: int
    ) -> None:
        if not isinstance(smiles_tokenizer, SmilesTokenizer):
            raise TypeError("smiles_tokenizer must be an instance of SmilesTokenizer.")
        if not isinstance(max_seq_length, int) or max_seq_length <= 0:
            raise ValueError("max_seq_length must be a positive integer.")

        self.smiles_tokenizer = smiles_tokenizer
        self.max_seq_length = max_seq_length

    def preprocess_smiles(self, tokenized_smiles_list: List[str]) -> tf.Tensor:
        """
        Converts a list of tokenized SMILES strings into a tensor of padded integer sequences.

        This method transforms each tokenized SMILES string into a sequence of integers using the
        provided tokenizer. It then pads or truncates these sequences to ensure they all have the
        same length (`max_seq_length`).

        Padding is applied post-sequence to maintain the start of the sequence intact, which is
        crucial for models such as seq2seq models that predict the next token in a sequence.

        Parameters
        ----------
        tokenized_smiles_list : List[str]
            A list of tokenized SMILES strings. Each string should be a space-separated sequence of tokens.

        Returns
        -------
        tf.Tensor
            A tensor of shape `(num_sequences, max_seq_length)` containing the padded integer sequences.
            Each row corresponds to a padded sequence of integers representing a SMILES string.

        Raises
        ------
        ValueError
            If `tokenized_smiles_list` is empty.
            If any of the SMILES strings in `tokenized_smiles_list` are not strings.
        """
        if not tokenized_smiles_list:
            raise ValueError("tokenized_smiles_list must contain at least one SMILES string.")
        if not all(isinstance(smiles, str) for smiles in tokenized_smiles_list):
            raise ValueError("All elements in tokenized_smiles_list must be strings.")

        sequences: tf.Tensor = self.smiles_tokenizer.texts_to_sequences(tokenized_smiles_list)

        # Pad sequences
        padded_sequences: tf.Tensor = pad_sequences(
            sequences,
            maxlen=self.max_seq_length,
            padding='post',
            truncating='post'
        )
        return tf.constant(padded_sequences, dtype=tf.int32)
