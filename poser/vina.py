# -*- coding: utf-8 -*-
"""
medchem_pipelines.docking.vina
~~~~~~~~~~~~~~

A wrapper around Autodock Vina
"""
import json
from pathlib import Path
from zipfile import ZipFile
from subprocess import run
from tempfile import TemporaryDirectory

import pandas as pd
from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina

from .ligand import Ligand

class PDBQTError(ValueError):
    pass


def check_obabel_installation():
    """Check if OpenBabel can be invoked, otherwise throw an exception"""

    try:
        run(["obabel", "-V"], check=False)
    except FileNotFoundError:
        # TODO: place correct exception type
        raise Exception(
            "OpenBabel executable not found. Please install obabel: "
            "http://openbabel.org/docs/Installation/install.html"
        )


def prepare_ligand(ligand: Ligand) -> str:
    """Convert SMILES to Vina-readable ligand format PDBQT"""

    ligand_mol = Chem.AddHs(ligand.rdmol)
    AllChem.EmbedMultipleConfs(ligand_mol, numConfs=1, params=AllChem.ETKDG())

    preparator = MoleculePreparation(min_ring_size=5)
    mol_setups = preparator.prepare(ligand_mol)

    for setup in mol_setups:
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
        if is_ok:
            return f"REMARK NAME {ligand.name}\n{pdbqt_string}"
        else:
            raise PDBQTError(error_msg)


def vina_poses_to_sdf(poses_pdbqt: str, poses_sdf: Path, mol_name: str = "lig"):
    """
    Convert Vina's output PDBQT to sdf

    :param poses_pdbqt: Vina output as PDBQT string
    :param poses_sdf: Path to save converted output
    :param mol_name: Ligand name to store in the converted file
    """
    pdbqt_mol = PDBQTMolecule(poses_pdbqt, skip_typing=True)
    rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)

    with Chem.SDWriter(poses_sdf.as_posix()) as sd_writer:
        sd_writer.SetProps(["_Name"])
        for mol in rdkitmol_list:
            mol.SetProp("_Name", mol_name)
            for conformer in mol.GetConformers():
                conf_id = conformer.GetId()
                sd_writer.write(mol, confId=conf_id)


def prepare_vina_grid(
    receptor_path: Path,
    box_center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    save_dir: Path,
) -> Path:
    """
    Make Vina grid with ligandless receptor PDB and docking box parameters.

    :param receptor_path: Path to receptor PDB
    :param box: Docking box center and dimensions
    :param save_dir: directory to write resulting grid
    :return: Path to grid zip file
    """
    output_suffix = "_vina_grid"
    out_fname = f"{receptor_path.stem}{output_suffix}.zip"

    check_obabel_installation()
    save_dir.mkdir(exist_ok=True, parents=True)

    with TemporaryDirectory() as tmp_wd:
        pdbqt_path = Path(tmp_wd) / f"{receptor_path.stem}.pdbqt"
        grid_params_path = Path(tmp_wd) / "grid.json"
        grid_path = save_dir / out_fname

        obabel_command = [
            "obabel",
            receptor_path.as_posix(),
            "--partialcharge",
            "gasteiger",
            "-xr",
            "-O",
            pdbqt_path.as_posix(),
        ]

        process = run(obabel_command, capture_output=True, check=False)
        print(process)

        with grid_params_path.open("w") as grid_json:
            json.dump(
                {
                    "center": box_center,
                    "size": box_size,
                },
                grid_json
            )

        with ZipFile(grid_path.as_posix(), "w") as grid_zip:
            grid_zip.write(pdbqt_path, pdbqt_path.name)
            grid_zip.write(grid_params_path, grid_params_path.name)

    return grid_path


def vina_dock(
    grid_path: Path,
    ligands: list[Ligand],
    write_dir: Path,
    exhaustiveness: int = 128,
    n_poses: int = 5,
) -> dict[str, tuple[Path, Path]]:
    """
    Perform Autodock Vina docking

    :param grid_path: Path to grid zip archive
    :param ligands: list of ligands
    :param write_dir: Path to save results
    :param store_pdbqt: keep resulting PDBQT pose
    :param exhaustiveness: how thoroughly to make a docking. The more exhaustiveness,
    the more computation time and accuracy
    :param n_poses: how many poses to keep in the report
    :return: Lists of resulting paths (one file per ligand) always in format:
        *.sdf - list ligand poses,
        *.csv - list of scores,
        *.pdbqt - path to pdbqt if `store_pdbqt`.
    """
    output_suffix = "_vina_result"

    write_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(grid_path.as_posix()) as grid:
        receptor_fname, grid_fname = grid.namelist()
        with grid.open(grid_fname) as grid_json:
            box = json.load(grid_json)
        with grid.open(receptor_fname) as recep:
            receptor_pdbqt = write_dir / receptor_fname
            receptor_pdbqt.write_bytes(recep.read())

    results = {}

    engine = Vina(sf_name="vina")
    engine.set_receptor(receptor_pdbqt.as_posix())
    engine.compute_vina_maps(center=box["center"], box_size=box["size"])

    for ligand in ligands:

        lig_pdbqt = prepare_ligand(ligand)
        engine.set_ligand_from_string(lig_pdbqt)
        engine.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

        score = pd.DataFrame(
            engine.energies(n_poses=n_poses)[:, :4], columns=["total", "inter", "intra", "torsions"]
        )
        score_path = write_dir / f"{receptor_pdbqt.stem}_{ligand.name}{output_suffix}.csv"
        score.to_csv(score_path, index=False)

        poses_pdbqt = engine.poses(n_poses=n_poses)
        poses_sdf_path = write_dir / f"{receptor_pdbqt.stem}_{ligand.name}{output_suffix}.sdf"
        vina_poses_to_sdf(poses_pdbqt, poses_sdf_path, mol_name=ligand.name)

        results[ligand.name] = (poses_sdf_path, score_path)
    return results
