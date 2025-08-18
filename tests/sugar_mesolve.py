import math
import numpy as np
import qutip as qt

from superspinsim import mesolve

def main():
    density_init = np.array(
        [[1, 0],
        [0, 0]]
    )

    frequency = 2.8e9

    def coef_x(time):
        return np.sin(math.tau*frequency*time)

    hamiltonian = [qt.sigmaz(), [qt.sigmax(), coef_x]]
    mesolve(hamiltonian, density_init, 0, 1e-6, 1e-9, density_init)
    print("done")
