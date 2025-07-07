import qutip as qt
import numpy as np
import math
from superspinsim.util import colour_complex_matrix
from comparisons.lindbladians import contrast
from matplotlib import pyplot as plt


def main():
    # spin_x = 1/math.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    # spin_x = qt.Qobj(spin_x)
    # spin_z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    # spin_z = qt.Qobj(spin_z)
    # hamiltonian = [
    #     [spin_x, lambda t: 1 + np.sin(20*math.tau*t)],
    #     [spin_z, lambda t: 1]
    # ]

    # results = qt.mesolve(
    #     H=hamiltonian,
    #     rho0=density_initial,
    #     tlist=time
    # )

    # plt.figure(label="figure_test")
    # plt.imshow(
    #     colour_complex_matrix(density_final/np.max(np.abs(density_final))))
    # plt.draw()

    coefficients, generators_coherent, generators_jump, time_end = contrast()

    hamiltonian = [
        qt.Qobj(generators_coherent[3]),
        [qt.Qobj(generators_coherent[0]), coefficients[0]],
        [qt.Qobj(generators_coherent[1]), coefficients[1]],
        [qt.Qobj(generators_coherent[2]), coefficients[2]]
    ]

    jumps = []
    jump_coefficient = lambda t: math.sqrt(coefficients[3](t))
    for jump in generators_jump[1]:
        jumps.append(qt.Qobj(jump))
    for jump in generators_jump[0]:
        jumps.append([qt.Qobj(jump), jump_coefficient])

    density_initial = np.zeros((7, 7))
    density_initial[0, 0] = 1/3
    density_initial[1, 1] = 1/3
    density_initial[2, 2] = 1/3
    density_initial = qt.Qobj(density_initial)

    time_step = 2e-9
    time = np.arange(0, time_end, time_step)

    results = qt.mesolve(
        H=hamiltonian,
        c_ops=jumps,
        rho0=density_initial,
        tlist=time
    )
    density = np.array([state.data.to_array() for state in results.states])
    fluorescence = density[:, 3, 3] + density[:, 4, 4] + density[:, 5, 5]

    plt.figure(label="fluorescence")
    plt.plot(time, fluorescence, "k-")
    plt.draw()
