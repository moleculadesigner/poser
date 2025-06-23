# Poser
A pipeline to do docking in Vina followed by ligand fingerprint screening.

## Usage
Minimalistic example:

``` python
import poser
from pathlib import Path

patterns = [
    poser.StructuralFingerprintPattern(
        core_name="bic",
        core_smarts="[#6]~1~[#6]~[#6]~2~[#6]~[#7:2]~[#6]~[#7]~[#6]~2~[#7:1]~1",
        dist_mappings=[
            poser.DistanceMapping(
                poser.LigandAtom(1),
                poser.LigandAtom(2),
                (0, 1000),
            ),
        ]
    )
]

pipeline = poser.DockPipeline(
    receptor_path=Path("5s18-clean_Prep_Recep.pdb"),
    smi={
        "woy": "c1ncc2c(n1)NCC2",
        "lig2": "c1nc(F)c2c(n1)NCC2",
    },
    work_dir="result",
    box_center=(-2.398, 14.862, -0.02),
    box_size=(6.0, 2.0, 4.0),
    patterns=patterns,
    exhaustiveness= 2,
    n_poses= 3,
)

pipeline.run()
```

## Authors
Danila Iakovlev, Biocad
Nikita Efimenko, Biocad
Ilya Krainov, Biocad