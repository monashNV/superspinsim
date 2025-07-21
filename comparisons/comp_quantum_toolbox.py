import os
import h5py


def main():
    os.system("julia --project=julia_env comparisons/comp_quantum_toolbox.jl")
    with h5py.File("julia_save.h5", "r") as h5_file:
        print(h5_file.attrs["hello world"].decode("utf-8"))
    os.remove("julia_save.h5")
