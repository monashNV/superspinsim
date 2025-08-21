import math
import numpy as np

from superspinsim import simspins
from superspinsim.models import nv_7

from matplotlib import pyplot as plt


def main():
    time_start = 1e-6
    time_end = 1000e-6
    time_step = 15e-9
    frequency_start = 2.82e9
    frequency_width = 100e6

    density_initial = np.zeros([7, 7], np.complex128)
    density_initial[0, 0] = 1/3
    density_initial[1, 1] = 1/3
    density_initial[2, 2] = 1/3

    quiescent_magnetic_field = np.array([0, 0, 1])*1e-3

    def coefficient_x(time):
        if time > time_start:
            phase = math.tau*(
                frequency_start*(time - time_start)
                + frequency_width*((time - time_start)**2)
                / (time_end - time_start)/2
            )
            return 100e-6*math.sin(phase)
        else:
            return 0

    def coefficient_y(time):
        return 0

    def coefficient_z(time):
        return 0

    def coefficient_r(time):
        return 0.01

    coefficients = [coefficient_x, coefficient_y, coefficient_z, coefficient_r]

    time, density = simspins(
        coefficients, 0, time_end, time_step, *nv_7(quiescent_magnetic_field),
        density_initial, number_of_fine_divisions=30, use_rotating=False
    )
    fluorescence = density[:, 3, 3] + density[:, 4, 4] + density[:, 5, 5]
    fluorescence = np.real(fluorescence)

    plt.figure()
    plt.plot(time/1e-6, 100*fluorescence, "k-")
    plt.xlabel("Time (us)")
    plt.ylabel("Fluorescence (%)")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.show()
