import os
import h5py
import numpy as np

from comparisons.lindbladians import contrast


def main(lindbladian="contrast"):
    if lindbladian == "contrast":
        generate_return, generate_return_comparison = contrast()

    coefficients, generators_coherent, generators_jump, density_initial, \
        time_step, time_end = generate_return_comparison

    with h5py.File("to_julia.h5", "w") as h5_file:
        h5_file.attrs["time_step"] = time_step
        h5_file.attrs["time_end"] = time_end

    os.system("julia --project=julia_env comparisons/comp_quantum_toolbox.jl")

    with h5py.File("from_julia.h5", "r") as h5_file:
        print(np.asarray(h5_file["times"].shape))
    os.remove("to_julia.h5")
    os.remove("from_julia.h5")
