import os
import json
import glob
import gzip
from typing import Generator, Tuple, List, Callable, Sequence

from ord_schema.message_helpers import load_message, write_message
from ord_schema.proto import dataset_pb2
from ord_schema.proto import reaction_pb2
from ord_schema import message_helpers
from rdkit import Chem
from rdkit.Chem import rdChemReactions, AllChem
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Contrib.RxnRoleAssignment import identifyReactants


class OpenReactionDatabaseExtractor:
    def __init__(self, ord_data_dir: str):
        self.ord_data_dir = ord_data_dir

    def write_reactions_to_files(
        self,
        reactants_smiles_path: str,
        products_smiles_path: str
    ):
        os.makedirs(os.path.dirname(reactants_smiles_path), exist_ok=True)
        os.makedirs(os.path.dirname(products_smiles_path), exist_ok=True)

        with open(reactants_smiles_path, 'w') as reactants_file, \
             open(products_smiles_path, 'w') as products_file:
            for reactant_line, cleaned_product_line in self.extract_all_reactions():
                reactants_file.write(reactant_line + '\n')
                products_file.write(cleaned_product_line + '\n')

    def extract_all_reactions(self)-> Generator[Tuple[str, str], None, None]:
        """
        Generator that yields all reactions contained in the open-reaction-database/ord-data repository dataset.

        Yields:
            reaction_pb2.Reaction: A parsed Reaction protocol buffer message.
        """
        pb_files = glob.glob(os.path.join(self.ord_data_dir, '**', '*.pb.gz'), recursive=True)

        for pb_file in pb_files:
            dataset = load_message(pb_file, dataset_pb2.Dataset)

            for rxn in dataset.reactions:
                rxn_smarts = None
                for identifier in rxn.identifiers:
                    if identifier.type == reaction_pb2.ReactionIdentifier.REACTION_CXSMILES:
                        rxn_smarts = identifier.value
                        break
                if rxn_smarts is None:
                    continue

                cleaned_rxn_smiles = identifyReactants.reassignReactionRoles(rxn_smarts)
                if not cleaned_rxn_smiles:
                    continue

                cleaned_rxn = AllChem.ReactionFromSmarts(cleaned_rxn_smiles, useSmiles=True)

                _, unmodified_reactants, unmodified_products = identifyReactants.identifyReactants(
                    cleaned_rxn
                )

                reactant_smiles = self._get_reactant_smiles_from_cleaned_rxn(cleaned_rxn, unmodified_reactants)
                product_smiles = self._get_product_smiles_from_cleaned_rxn(cleaned_rxn, unmodified_products)

                reactant_line = '.'.join(reactant_smiles) if reactant_smiles else ''
                product_line = '.'.join(product_smiles) if product_smiles else ''
                cleaned_product_line = self._remove_smiles_inorganic_fragments(product_line)

                yield reactant_line, cleaned_product_line

    def _get_smiles_from_templates(
        self,
        get_template_count: Callable[[], int],
        get_template: Callable[[int], Chem.Mol],
        unmodified_indices=None
    ):
        mols = [get_template(i) for i in range(get_template_count())]

        if unmodified_indices is not None:
            main_indices = [i for i in range(get_template_count()) if i not in unmodified_indices]
            mols = [mols[i] for i in main_indices]

        for mol in mols:
            self._remove_atom_mapping_from_mol(mol)

        return [Chem.MolToSmiles(mol) for mol in mols]

    def _get_reactant_smiles_from_cleaned_rxn(
        self,
        cleaned_rxn: ChemicalReaction,
        unmodified_reactants=None
    ):
        return self._get_smiles_from_templates(
            get_template_count=cleaned_rxn.GetNumReactantTemplates,
            get_template=cleaned_rxn.GetReactantTemplate,
            unmodified_indices=unmodified_reactants
        )

    def _get_product_smiles_from_cleaned_rxn(
        self,
        cleaned_rxn: ChemicalReaction,
        unmodified_products=None
    ):
        return self._get_smiles_from_templates(
            get_template_count=cleaned_rxn.GetNumProductTemplates,
            get_template=cleaned_rxn.GetProductTemplate,
            unmodified_indices=unmodified_products
        )

    def _extract_ord_reaction_smiles(
        self,
        rxn: reaction_pb2.Reaction,
        role_identifier: int
    ) -> List[str]:
        compound_smiles = []

        if role_identifier == reaction_pb2.ReactionRole.REACTANT:
            for rxn_input in rxn.inputs.values():
                for component in rxn_input.components:
                    if component.reaction_role == role_identifier:
                        self._extract_smiles_from_ord_identifiers(component.identifiers, compound_smiles)

            return compound_smiles

        elif role_identifier == reaction_pb2.ReactionRole.PRODUCT:
            for outcome in rxn.outcomes:
                for product in outcome.products:
                    if product.reaction_role == role_identifier:
                        self._extract_smiles_from_ord_identifiers(product.identifiers, compound_smiles)

            return compound_smiles

    @staticmethod
    def _extract_smiles_from_ord_identifiers(
        identifiers: Sequence[reaction_pb2.CompoundIdentifier],
        smiles_list: List
    ):
        for identifier in identifiers:
            if identifier.type == reaction_pb2.CompoundIdentifier.SMILES:
                mol = Chem.MolFromSmiles(identifier.value)
                if mol:
                    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    smiles_list.append(canonical_smiles)

        return smiles_list

    @staticmethod
    def _remove_atom_mapping_from_mol(mol: Chem.Mol):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    @staticmethod
    def _remove_smiles_inorganic_fragments(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return smiles

        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) == 1:
            return Chem.MolToSmiles(frags[0])

        main_frag = max(frags, key=lambda frag: frag.GetNumHeavyAtoms())

        return Chem.MolToSmiles(main_frag, isomericSmiles=True)
