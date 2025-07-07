import qutip as qt
import numpy as np
import math
from superspinsim.util import colour_complex_matrix
from matplotlib import pyplot as plt


def main():
    spin_x = 1/math.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    spin_x = qt.Qobj(spin_x)
    spin_z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    spin_z = qt.Qobj(spin_z)
    hamiltonian = [
        [spin_x, lambda t: 1 + np.sin(20*math.tau*t)],
        [spin_z, lambda t: 1]
    ]

    density_initial = np.zeros((3, 3))
    density_initial[0, 0] = 1
    density_initial = qt.Qobj(density_initial)
    time = np.linspace(0, 1)

    results = qt.mesolve(
        H=hamiltonian,
        rho0=density_initial,
        tlist=time
    )
    density = results.states
    density_final = density[-1].data.to_array()

    plt.figure(label="figure_test")
    plt.imshow(
        colour_complex_matrix(density_final/np.max(np.abs(density_final))))
    plt.draw()
