import os
import sqlite3
import logging
from typing import List, Tuple, Optional, Union

import tensorflow as tf
from rdkit import Chem
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data.utils.tokenization import SmilesTokenizer

class SmilesDataPreprocessor:
    def __init__(
        self,
        products_smiles: Optional[List[List[str]]] = None,
        reactants_smiles: Optional[List[List[str]]] = None,
    ):
        self.products_smiles = products_smiles
        self.reactants_smiles = reactants_smiles

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
            logging.info("Concatenating product datasets.")
            logging.info(f"Total products before concatenation: {len(self.products_smiles)}")
            self.products_smiles.extend(products_smiles_lists)
            logging.info(f"Total products after concatenation: {len(self.products_smiles)}")

        if reactants_smiles_lists is not None:
            logging.info("Concatenating reactant datasets.")
            logging.info(f"Total reactants before concatenation: {len(self.reactants_smiles)}")
            self.reactants_smiles.extend(reactants_smiles_lists)
            logging.info(f"Total reactants after concatenation: {len(self.reactants_smiles)}")

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
        if self.products_smiles is None or self.reactants_smiles is None:
            logging.error("Cannot write empty reaction dataset to files.")
            return

        logging.info("Starting writing reactions to files.")

        os.makedirs(os.path.dirname(reactants_smiles_path), exist_ok=True)
        os.makedirs(os.path.dirname(products_smiles_path), exist_ok=True)

        with open(reactants_smiles_path, 'w') as reactants_file, \
                open(products_smiles_path, 'w') as products_file:

            total = len(self.products_smiles)

            for idx, (product_smiles, reactant_smiles) in enumerate(zip(self.products_smiles, self.reactants_smiles), 1):
                reactant_line = '.'.join(reactant_smiles) if reactant_smiles else ''
                product_line = '.'.join(product_smiles) if product_smiles else ''
                cleaned_product_line = self.remove_smiles_inorganic_fragments(product_line)

                reactants_file.write(reactant_line + '\n')
                products_file.write(cleaned_product_line + '\n')

                if idx % log_interval == 0:
                    logging.info(f"Written {idx}/{total} reactions to files.")

        logging.info("Writing reactions to files completed successfully.")

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
        Union[Tuple[List[List[str]], List[List[str]]], None]
            Unique product and reactant SMILES lists after deduplication, or None.
        """
        if self.products_smiles is None or self.reactants_smiles is None:
            logging.error("Cannot remove duplicates from empty reaction dataset.")
            return None

        logging.info("Starting SQLite-based deduplication.")
        unique_x = []
        unique_y = []

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
            batch = []

            for idx, (x, y) in enumerate(zip(self.products_smiles, self.reactants_smiles), 1):
                reactant_line = '.'.join(y)
                product_line = '.'.join(x)
                batch.append((reactant_line, product_line))

                if len(batch) >= batch_size:
                    self._process_batch(
                        batch=batch,
                        cursor=cursor,
                        unique_x=unique_x,
                        unique_y=unique_y
                    )
                    batch = []

                    if idx % log_interval == 0:
                        logging.info(f"Processed {idx}/{total} reactions.")

            # Process any remaining pairs
            if batch:
                self._process_batch(
                    batch=batch,
                    cursor=cursor,
                    unique_x=unique_x,
                    unique_y=unique_y
                )

            conn.commit()
            logging.info("SQLite-based deduplication completed successfully.")
        finally:
            conn.close()

        self.products_smiles = unique_x
        self.reactants_smiles = unique_y

    def deduplicate_in_memory(self) -> Union[Tuple[List[List[str]], List[List[str]]], None]:
        """
        Deduplicate reaction pairs using in-memory sets.

        Returns:
        -------
        Union[Tuple[List[str], List[str]], None]
            Unique x_data (products) and y_data (reactants), or None.
        """
        if self.products_smiles is None or self.reactants_smiles is None:
            logging.error("Cannot remove duplicates from empty reaction dataset.")
            return None

        logging.info("Starting in-memory deduplication.")
        seen = set()
        unique_x = []
        unique_y = []

        for x, y in zip(self.products_smiles, self.reactants_smiles):
            pair = (tuple(y), tuple(x))  # Reactant, Product
            if pair not in seen:
                seen.add(pair)
                unique_x.append(x)
                unique_y.append(y)

        logging.info(f"Deduplication completed. Unique reactions: {len(unique_x)}")

        return unique_x, unique_y

    @staticmethod
    def remove_smiles_inorganic_fragments(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return smiles

        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) == 1:
            return Chem.MolToSmiles(frags[0])

        main_frag = max(frags, key=lambda frag: frag.GetNumHeavyAtoms())

        return Chem.MolToSmiles(main_frag, isomericSmiles=True)

    @staticmethod
    def _process_batch(
        batch: List[Tuple[str, str]],
        cursor: sqlite3.Cursor,
        unique_x: List[List[str]],
        unique_y: List[List[str]]
    ) -> None:
        """
        Process a batch of reaction pairs: insert into DB and append to unique lists if unique.

        Parameters
        ----------
        batch : List[Tuple[str, str]]
            List of (reactant, product) pairs.
        cursor : sqlite3.Cursor
            SQLite cursor for executing database operations.
        unique_x : List[List[str]]
            List to append unique product SMILES.
        unique_y : List[List[str]]
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
                unique_y.append(reactant_line.split('.'))
                unique_x.append(product_line.split('.'))
        except sqlite3.IntegrityError:
            # Some inserts failed due to duplicates; handle individually
            for reactant_line, product_line in batch:
                try:
                    cursor.execute(
                        "INSERT INTO pairs (reactant, product) VALUES (?, ?)",
                        (reactant_line, product_line)
                    )
                    # Insert succeeded; append to unique lists
                    unique_y.append(reactant_line.split('.'))
                    unique_x.append(product_line.split('.'))
                except sqlite3.IntegrityError:
                    # Duplicate pair; skip appending
                    logging.debug(f"Duplicate pair skipped: Reactant={reactant_line}, Product={product_line}")
                    continue


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
