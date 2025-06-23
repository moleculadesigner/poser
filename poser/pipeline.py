# -*- coding: utf-8 -*-
"""
medchem_pipelines.docking.ifd_mmgbsa_pipeline
~~~~~~~~~~~~~~

Functions to run combined IFD -> MM/GBSA pipeline with pose analysis
"""

import asyncio
import io
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Optional
import datetime

import oneq as oq
import pandas as pd
from IPython.core.display import Image
from pydantic import BaseModel, ConfigDict
from rdkit import Chem
from rdkit.Chem import Draw



from .ligand import Ligand
from .patterns import StructuralFingerprintPattern, find_matching_pattern
from . import vina


def analyze_good_poses(merged_df: pd.DataFrame) -> dict:
    """
    Analyse good poses and aggregate docking data for best pose of 1 ligand

    :param merged_df: Docking results DataFrame with `good_pose` column
    :return: Dict of best pose data
    """
    
    best_pose_i = int(merged_df["total"].idxmin())
    good_poses_df = merged_df[merged_df["good_pose"]]

    result = dict(
        min_score=merged_df["total"].min(),
        best_pose=best_pose_i,
        good_poses=list(good_poses_df["pose"]),
        pose_ratio=len(good_poses_df) / len(merged_df),
        avg_score=(
            merged_df["total"].mean()
            if merged_df.get("total") is not None
            else None
        ),
        avg_good_score=good_poses_df["total"].mean(),
    )
    return result


class DockPipeline(BaseModel):

    receptor_path: Path
    smi: Path | dict[str, str]
    work_dir: Path

    box_center: tuple[float, float, float]
    box_size: tuple[float, float, float]

    patterns: list[StructuralFingerprintPattern]

    ligands: list[Ligand] = []
    docking_results: pd.DataFrame | None = None

    exhaustiveness: int = 32
    n_poses: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def dump(self):
        """Dump pipeline state to workdir"""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        dump_file = self.work_dir / "ifd_pipeline.pkl"
        dump_file.write_bytes(pickle.dumps(self))
        return dump_file

    @classmethod
    def load(cls, dump_file: Path):
        """
        Load pipeline state from dump

        :param dump_file: Dump file dump
        :return: Pipeline object
        """
        load_ = pickle.loads(dump_file.read_bytes())
        if not isinstance(load_, cls):
            raise ValueError(f"Inappropriate dump at {dump_file.as_posix()}")
        return load_

    def read_ligands(self):
        """Read ligands from .smi file or dict {lig_name: SMILES}"""

        print(f"Pipeline state is stored at {self.dump().absolute().as_posix()}")
        if isinstance(self.smi, Path):
            self.ligands = [
                Ligand(
                    name=line.split()[1],
                    rdmol=Chem.AddHs(Chem.MolFromSmiles(line)),
                )
                for line in self.smi.read_text().splitlines()
                if line
            ]
        elif isinstance(self.smi, dict):
            self.ligands = [
                Ligand(
                    name=name,
                    rdmol=Chem.AddHs(Chem.MolFromSmiles(smiles)),
                )
                for name, smiles in self.smi.items()
            ]
        else:
            raise ValueError(f"Incorrect smi list: {self.smi}")

    def run(self):
        """
        Run docking and aggregate results

        :param oneq_session: OneQ session instance
        """
        self.read_ligands()
        self.dump()
        self.run_vina()
        self.dump()
        self.aggregate_results()
        self.dump()
        return self.screen_poses(self.patterns)

    def highlight_patterns(
        self, patterns: Optional[list[StructuralFingerprintPattern]] = None
    ) -> Image:
        """
        Draw image with ligands highlited by their matched patterns

        :param patterns: List of patterns to highlight matched atoms in ligand molecules
        :return: Grid image on highlighted molecules
        """
        if not patterns:
            patterns = self.patterns
        matches = []
        for ligand in self.ligands:
            pattern = find_matching_pattern(
                ligand.rdmol,
                patterns,
            )
            if pattern is not None:
                matched_atoms = ligand.rdmol.GetSubstructMatch(
                    Chem.MolFromSmarts(pattern.core_smarts)
                )
                matches.append((matched_atoms, pattern))
            else:
                matches.append(((), pattern))

        img = Draw.MolsToGridImage(
            [ligand.rdmol for ligand in self.ligands],
            molsPerRow=4,
            legends=[
                f"{ligand.name} : {pattern.core_name if pattern else 'Unmatched'}"
                for ligand, (_, pattern) in zip(self.ligands, matches)
            ],
            highlightAtomLists=[matched_atoms for (matched_atoms, _) in matches],
            subImgSize=(300, 300),
            returnPNG=True,
        )
        return img

    def screen_poses(self, patterns: list[StructuralFingerprintPattern]) -> pd.DataFrame:
        """
        Screen poses of matching patterns and find metricas and best pose data

        :param patterns: List of patterns
        :param save_dir: Dir to save screen results
        :return: Screening DataFrame
        """
        save_dir = self.work_dir / f"screening-{datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')}"
        save_dir.mkdir(parents=True, exist_ok=True)

        screening_results = []
        for ligand in self.ligands:
            pattern = find_matching_pattern(ligand.rdmol, patterns)
            if pattern is None:
                continue
            fingerprints = ligand.apply_pattern(pattern)
            if fingerprints is None:
                continue
            fingerprints.to_csv(save_dir / f"{ligand.name}-fp.csv")

            if ligand.docking_data is None:
                continue
            df = ligand.docking_data.copy()
            df["good_pose"] = fingerprints["good_pose"]
            screening_data = analyze_good_poses(merged_df=df)
            screening_results.append(screening_data)


        screening_df = pd.DataFrame.from_records(screening_results)
        screening_df.to_csv(save_dir / "result.csv")
        return screening_df

    def run_vina(
        self,
    ):
        write_dir = self.work_dir / "docking"
        write_dir.mkdir(exist_ok=True, parents=True)
        grid = vina.prepare_vina_grid(
            receptor_path=self.receptor_path,
            box_center=self.box_center,
            box_size=self.box_size,
            save_dir=write_dir,
        )
        results = vina.vina_dock(
                grid_path=grid,
                ligands=self.ligands,
                write_dir=write_dir,
                exhaustiveness=self.exhaustiveness,
                n_poses=self.n_poses,
            )
        for ligand in self.ligands:
            ligand.vina_poses, ligand.scores = results[ligand.name]
            ligand.receptor = self.receptor_path


    def aggregate_results(self):
        """Aggregate docking run results"""
        for ligand in self.ligands:
            print("Extracting", ligand.name)
            ligand.name
            ligand.smiles
            if ligand.scores is None:
                warnings.warn(
                    Warning(
                        ValueError(
                            f"There was no docking scores performed for the ligand {ligand.name}"
                        )
                    )
                )
                continue
            df = pd.read_csv(ligand.scores)  # type: ignore
            df["pose"] = list(range(len(df)))
            df["ligand"] = ligand.name
            df["smiles"] = ligand.smiles
            ligand.docking_data = df
            df.to_csv(ligand.scores.parent / f"{ligand.name}-docking-data.csv")  # type: ignore

