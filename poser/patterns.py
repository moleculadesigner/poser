# -*- coding: utf-8 -*-
"""
medchem_pipelines.docking.pose_analyze
~~~~~~~~~~~~~~

Pose analyze module after docking
"""
from typing import NamedTuple

import numpy as np
import parmed
from IPython.core.display import Image
from pydantic import BaseModel
from rdkit import Chem

from . import utils


class LigandAtom(NamedTuple):
    """Wrapper of ligand atom map index"""

    map_index: int
    """Map number in the ligand pattern structure"""

    def __str__(self):
        return "L.{}".format(self.map_index)


class ProteinAtom(NamedTuple):
    """Wrapper of protein atom identifier"""

    chain_name: str
    residue_number: int
    atom_name: str

    def __str__(self):
        return "P.{}@{}".format(self.atom_name, self.residue_number)


Atom = LigandAtom | ProteinAtom
"""Type alias for the atom mapping"""


class DistanceMapping(NamedTuple):
    """
    Representation of desirable distance between ligand or protein atoms
    """

    atom_1: Atom
    atom_2: Atom
    allowed_distances: tuple[float, float]
    """(min, max) distance range between query atoms"""


class AngleMapping(NamedTuple):
    """
    Representation of desirable angle formed by three ligand or protein atoms
    """

    atom_1: Atom
    atom_2: Atom
    atom_3: Atom
    allowed_angles: tuple[float, float]
    """Minimal and maximal allowed angles in degrees from 0 to 180"""


class DihedralMapping(NamedTuple):
    """
    Representation of desirable dihedral angle
    formed by four ligand or protein atoms
    """

    atom_1: Atom
    atom_2: Atom
    atom_3: Atom
    atom_4: Atom
    allowed_dihedrals: tuple[float, float]
    """
    Minimal and maximal allowed dihedrals in degrees from -180 to 180

    if first value is more than the second, allowed angles are counted through
    180 degrees, eg (160, -160) means, that 168 and -179 are within the interval
    and 150 and -30 are not.
    """


class StructuralFingerprintPattern(BaseModel):
    """
    Representation of tagged fingerprint pattern
    """

    core_name: str
    """Name of the pattern"""

    core_smarts: str
    """Tagged smarts of the pattern"""

    dist_mappings: list[DistanceMapping] = []
    """Distances to measure in the pose analysis"""

    angle_mappings: list[AngleMapping] = []
    """Angles to measure in the pose analysis"""

    dihedral_mappings: list[DihedralMapping] = []
    """Dihedrals to measure in the pose analysis"""


def draw_patterns(patterns: list[StructuralFingerprintPattern]) -> Image:
    """
    Draw StructuralFingerprintPattern strucrures in a grid

    :param patterns: list of StructuralFingerprintPattern's to draw
    :return: IPython.core.display.Image object with drawing
    """
    return Chem.Draw.MolsToGridImage(
        [Chem.MolFromSmarts(pattern.core_smarts) for pattern in patterns],
        molsPerRow=4,
        legends=[pattern.core_name for pattern in patterns],
        subImgSize=(300, 300),
        returnPNG=True,
    )


def find_matching_pattern(
        ligand_rdmol: Chem.Mol,
        patterns: list[StructuralFingerprintPattern],
) -> StructuralFingerprintPattern | None:
    matched_pattern = None
    for pattern in patterns:
        substructure = Chem.MolFromSmarts(pattern.core_smarts)
        matched_atoms = ligand_rdmol.GetSubstructMatch(substructure)
        if matched_atoms:
            ligand_rdmol.SetBoolProp(
                "matched",
                True,
            )
            ligand_rdmol.SetProp(
                "match_core_name",
                pattern.core_name,
            )
            matched_pattern = pattern
            break
    if not matched_pattern:
        ligand_rdmol.SetBoolProp(
            "matched",
            False,
        )
        ligand_rdmol.SetProp(
            "match_core_name",
            "_unmatched",
        )
    return matched_pattern


def atom_mappings(
    ligand_rdmol: Chem.Mol,
    pattern: StructuralFingerprintPattern,
) -> dict[int, int]:
    """
        Match ligand with the patterns and assign atoms with the first matched pattern tags

        :param ligand_rdmol: RDkit instance of the ligand
        :param pattern: pattern to match

        :return: mapping of pattern tags to atom number in matched molecule

    """
    mapping = {}
    substructure = Chem.MolFromSmarts(pattern.core_smarts)
    matched_atoms = ligand_rdmol.GetSubstructMatch(substructure)
    if not matched_atoms:
        raise ValueError(
            f"Provided pattern {pattern.core_name} ({pattern.core_smarts}) does not match the ligand {Chem.MolToSmiles(ligand_rdmol)}"
        )
    for core_idx, mol_idx in enumerate(matched_atoms):
        core_atom_label = substructure.GetAtomWithIdx(core_idx).GetAtomMapNum()
        if core_atom_label:
            mapping[core_atom_label] = mol_idx
            mol_atom = ligand_rdmol.GetAtomWithIdx(mol_idx)
            mol_atom.SetAtomMapNum(core_atom_label)
    return mapping


def pose_atom_mapping(
    ligand_structure: parmed.Structure,
    pattern: StructuralFingerprintPattern,
) -> dict[int, int]:
    """
    Find indices of pose atoms which correspond
    to pattern mapping
    """
    ligand_rdmol = ligand_structure.rdkit_mol
    mol_to_pose_map = {
        rdmol_atom.GetIdx(): pose_atom.idx
        for pose_atom, rdmol_atom in zip(ligand_structure.atoms, ligand_rdmol.GetAtoms())
    }

    tag_index = atom_mappings(ligand_rdmol, pattern)
    if tag_index:
        tags_to_pose_map = {tag: mol_to_pose_map[mol_idx] for tag, mol_idx in tag_index.items()}
    else:
        # FIXME: return after feature/chain_id merge
        raise ValueError(
            f"Ligand {Chem.MolToSmiles(ligand_rdmol)} does not match pattern {pattern.core_name} ({pattern.core_smarts})."
        )

    return tags_to_pose_map


def calculate_pose_fingerprints(
    ligand_structure: parmed.Structure,
    receptor_structure: parmed.Structure,
    pattern: StructuralFingerprintPattern,
) -> dict:
    """
    Check all criteria from pattern for particular pose
    """
    tags_to_pose_map = pose_atom_mapping(ligand_structure, pattern)

    def _extract_atom_coords(atom: Atom) -> np.ndarray:
        match atom:
            case LigandAtom(map_index):
                index = tags_to_pose_map[map_index]
                at_coords = ligand_structure.coordinates[index]
            case ProteinAtom(chain_name, residue_number, atom_name):
                atoms = []
                for res in receptor_structure.residues:
                    if (res.chain == chain_name) and (res.number == residue_number):
                        atoms.extend(res.atoms)
                found_atom = [i for i in atoms if i.name == atom_name][0]
                at_coords = receptor_structure.coordinates[found_atom.idx]
        return at_coords

    record = {}
    good_pose = True
    for atom_1, atom_2, (min_dist, max_dist) in pattern.dist_mappings:
        at_1_coords = _extract_atom_coords(atom_1)
        at_2_coords = _extract_atom_coords(atom_2)
        distance = np.linalg.norm(
            at_1_coords - at_2_coords
        )
        good_pose = good_pose and (min_dist <= distance <= max_dist)
        record[f"{atom_1} -> {atom_2} distance"] = distance
        record[f"{atom_1} -> {atom_2} minimum"] = min_dist
        record[f"{atom_1} -> {atom_2} maximum"] = max_dist

    for atom_1, atom_2, atom_3, angles in pattern.angle_mappings:
        atom_1_coords = _extract_atom_coords(atom_1)
        atom_2_coords = _extract_atom_coords(atom_2)
        atom_3_coords = _extract_atom_coords(atom_3)
        low, high = min(angles), max(angles)

        angle_val = utils.angle(
            atom_1_coords,
            atom_2_coords,
            atom_3_coords,
        )
        good_pose = good_pose and (low <= angle_val <= high)
        record[f"{atom_1} -> {atom_2} -> {atom_3} angle"] = angle_val
        record[f"{atom_1} -> {atom_2} -> {atom_3} low"] = low
        record[f"{atom_1} -> {atom_2} -> {atom_3} high"] = high

    for (
        atom_1,
        atom_2,
        atom_3,
        atom_4,
        (dihedrals_low, dihedrals_high),
    ) in pattern.dihedral_mappings:
        if dihedrals_low <= dihedrals_high:

            def in_interval(val):
                return dihedrals_low <= val <= dihedrals_high

        else:

            def in_interval(val):
                in_plus = dihedrals_low <= val <= 180
                in_minus = -180 <= val <= dihedrals_high
                return in_plus or in_minus

        atom_1_coords = _extract_atom_coords(atom_1)
        atom_2_coords = _extract_atom_coords(atom_2)
        atom_3_coords = _extract_atom_coords(atom_3)
        atom_4_coords = _extract_atom_coords(atom_4)
        dihedral_val = utils.dihedral(
            atom_1_coords,
            atom_2_coords,
            atom_3_coords,
            atom_4_coords,
        )
        good_pose = good_pose and in_interval(dihedral_val)
        record[f"{atom_1} -> {atom_2} -> {atom_3} -> {atom_4} dihedral"] = dihedral_val
        record[f"{atom_1} -> {atom_2} -> {atom_3} -> {atom_4} low"] = dihedrals_low
        record[f"{atom_1} -> {atom_2} -> {atom_3} -> {atom_4} high"] = dihedrals_high

    record["good_pose"] = good_pose
    return record
