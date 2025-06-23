
# -*- coding: utf-8 -*-
"""
poser.ligand.
~~~~~~~~~~~~~~

A ligand data container definition
"""

import warnings
from pathlib import Path

import pandas as pd
import parmed
from pydantic import BaseModel, ConfigDict
from rdkit import Chem

from .patterns import StructuralFingerprintPattern, calculate_pose_fingerprints


class Ligand(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    rdmol: Chem.Mol
    scores: Path | None = None
    vina_poses: Path | None = None
    receptor: Path | None = None
    docking_data: pd.DataFrame | None = None

    @property
    def smiles(self) -> str:
        return Chem.MolToSmiles(self.rdmol)

    @property
    def poses(self) -> dict[int, tuple[parmed.Structure, parmed.Structure]] | None:
        """
        Read multi-model pdb or many single-structure pdbs and return structures

        This attribute was made dynamic property for memory efficiency,
        as it is needed only once

        :return: Map of pose number -> Parmed pose structure (ligand, protein)
        """

        if self.vina_poses is None or self.receptor is None:
            warnings.warn(
                Warning(ValueError(f"There was no docking performed for the ligand {self.name}"))
            )
            return None

        structures = {}
        receptor_structure = parmed.read_pdb(self.receptor.resolve().as_posix())
        for pose_number, ligand_structure in enumerate(parmed.load_file(self.vina_poses.resolve().as_posix())):
            structures[pose_number] = ligand_structure, receptor_structure

        return structures

    def apply_pattern(self, pattern: StructuralFingerprintPattern) -> pd.DataFrame | None:
        """
        Match pattern to ligand

        :param ligand: Ligand to match pattern
        :param pattern: Pattern to match
        :return: Dataframe with match metricas or None if no match
        """
        substructure = Chem.MolFromSmarts(pattern.core_smarts)
        if not self.rdmol.GetSubstructMatch(substructure):
            return None
        poses = self.poses
        if poses is None:
            return None
        records = []
        for pose_n, pose in poses.items():
            record = calculate_pose_fingerprints(*pose, pattern)
            record["pose"] = pose_n
            records.append(record)
        return pd.DataFrame(records)