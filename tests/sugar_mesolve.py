import math
import numpy as np

from superspinsim import mesolve

def main():
    spin_z = np.array(
        [[1, 0],
        [0, -1]]
    )/2
    spin_x = np.array(
        [[0, 1],
        [1, 0]]
    )/2
    density_init = np.array(
        [[1, 0],
        [0, 0]]
    )

    frequency = 2.8e9

    def coef_x(time):
        return np.sin(math.tau*frequency*time)

    hamiltonian = [spin_z, [spin_x, coef_x]]
    mesolve(hamiltonian, density_init, 0, 1e-6, 1e-9, density_init)
    print("done")
