import os
import numpy as np

import warnings


REFERENCE_PATH = "../reference/"
DELIMITER = ","
PRECISION = 1e-5


def flatten_density(density: np.ndarray):
    density_expand = np.empty(
        (density.shape[0], density.shape[1], density.shape[2], 2),
        dtype=np.float64
    )
    density_expand[:, :, :, 0] = density.real
    density_expand[:, :, :, 1] = density.imag
    density_flatten = density_expand.flatten()
    density_flatten = density_flatten[::29]

    return density_flatten


def save_density(name: str, density: np.ndarray):
    os.makedirs(REFERENCE_PATH, exist_ok=True)
    density_flatten = flatten_density(density)
    path = REFERENCE_PATH + name + ".csv"
    if not os.path.exists(path):
        np.savetxt(path, density_flatten, delimiter=DELIMITER)


def load_density(name: str):
    if not os.path.exists(REFERENCE_PATH):
        return

    path = REFERENCE_PATH + name + ".csv"
    density = np.loadtxt(path, dtype=np.float64, delimiter=DELIMITER)
    return density


def compare_density(name: str, density: np.ndarray):
    reference = load_density(name)
    if reference is None:
        raise FileNotFoundError(name)

    this = flatten_density(density)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            difference = np.sqrt(np.sum((this - reference)**2))/this.size
        except RuntimeWarning:
            difference = 2*PRECISION
            print("LMAO")

    if difference > PRECISION:
        raise Exception("Test failed")
