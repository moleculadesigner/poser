import numpy as np


def angle(
        coord_a: np.ndarray,
        coord_b: np.ndarray,
        coord_c: np.ndarray
    ) -> float:
    """
    Calculate angle between vectors BA and BC

    :param coord_*: (1,3) array with corresponding point coordinates
    :return: angle ABC in degrees
    """
    vec_1 = coord_a - coord_b
    vec_1 = vec_1 / np.linalg.norm(vec_1)
    vec_2 = coord_c - coord_b
    vec_2 = vec_2 / np.linalg.norm(vec_2)

    return float(180 * np.arccos(np.dot(vec_1, vec_2)) / np.pi)


def dihedral(
        coord_i: np.ndarray,
        coord_j: np.ndarray,
        coord_k: np.ndarray,
        coord_l: np.ndarray,
    ) -> float:
    """
    Calculate dihedral angle between plains IJK and JKL

    Implemented from
    https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/MolTransforms/MolTransforms.cpp#L578

    :param coord_*: (1,3) array with corresponding point coordinates
    """
    coords = [coord_i, coord_j, coord_k, coord_l]

    ij, jk, kl = (coords[i + 1] - coords[i] for i in range(3))

    nijk = np.cross(ij, jk)
    nijk = nijk / np.linalg.norm(nijk)

    njkl = np.cross(jk, kl)
    njkl = njkl / np.linalg.norm(njkl)

    m = np.cross(nijk, jk)
    m = m / np.linalg.norm(m)
    return float(
        -180
        * np.arctan2(
            np.dot(m, njkl),
            np.dot(nijk, njkl),
        )
        / np.pi
    )

