import math
import numpy as np
from matplotlib import pyplot as plt

from superspinsim import generate_simulator, params as s3p
from superspinsim.generate_generators import generate_7, generate_21

from pogger import Pogger


def main():
    hyperfine = False

    time_start = 1e-6
    time_end = 1000e-6
    frequency_start = 2.72e9
    frequency_width = 400e6

    quiescent_magnetic_field = np.array([1, 0, 0])*1e-3

    def coefficient_z(time):
        if time > time_start:
            phase = math.tau*(
                frequency_start*(time - time_start)
                + frequency_width*((time - time_start)**2)
                / (time_end - time_start)/2
            )
            return 500e-6*math.sin(phase)
        else:
            return 0

    def coefficient_y(time):
        return 0

    def coefficient_x(time):
        return 0

    def coefficient_r(time):
        return 0.01

    coefficients = [coefficient_x, coefficient_y, coefficient_z, coefficient_r]
    orientations = [
        s3p.PointGroups.T.T111.a1p,
        s3p.PointGroups.T.T111.a3p,
        s3p.PointGroups.T.T111.b2p,
        s3p.PointGroups.T.T111.b4p,
    ]

    fluorescence_total = None
    for orientation in orientations:
        if hyperfine:
            lindbladian, generators, vectorisation_map = generate_21(
                coefficients, quiescent_magnetic_field)
        else:
            lindbladian, generators, vectorisation_map, vectors_real, \
                inv_vectors_real, doubles, singles = generate_7(
                    coefficients, quiescent_magnetic_field, use_rotating=True,
                    rotation_magnetic_gets_atom=orientation
                )

        fine_step = 200e-12
        if hyperfine:
            coarse_step = 150e-9
        else:
            coarse_step = 150e-9
        number_of_divisions = int(round(coarse_step/fine_step))
        coarse_step = number_of_divisions*fine_step

        simulator = generate_simulator(
            lindbladian, np.array(generators), vectorisation_map,
            number_of_fine_divisions=number_of_divisions,
            number_of_exponentials=5, use_rotating=True,
            vectors_real=vectors_real, inv_vectors_real=inv_vectors_real,
            doubles=doubles, singles=singles
        )
        if hyperfine:
            density_operator_initial = np.zeros((21, 21))
            density_operator_initial[:9, :9] = 1/9*np.eye(9)
        else:
            density_operator_initial = np.zeros((7, 7))
            density_operator_initial[:3, :3] = 1/3*np.eye(3)

        # density_operator_initial[1, 1, 0] = 1/2
        # density_operator_initial[0, 0, 0] = 1/4
        # density_operator_initial[2, 2, 0] = 1/4
        # density_operator_initial[0, 2, 0] = 1/4
        # density_operator_initial[0, 2, 0] = 1/4
        # density_operator_initial[0, 1, 0] = 1/(2*math.sqrt(2))
        # density_operator_initial[1, 0, 0] = 1/(2*math.sqrt(2))
        # density_operator_initial[1, 2, 0] = 1/(2*math.sqrt(2))
        # density_operator_initial[2, 1, 0] = 1/(2*math.sqrt(2))

        time, density = simulator(
            density_operator_initial, 0, time_end, coarse_step)

        if hyperfine:
            fluorescense = \
                density[:, 9, 9] + density[:, 10, 10] + density[:, 11, 11] \
                + density[:, 12, 12] + density[:, 13, 13] \
                + density[:, 14, 14] + density[:, 15, 15] \
                + density[:, 16, 16] + density[:, 17, 17]
        else:
            fluorescense = \
                density[:, 3, 3] + density[:, 4, 4] + density[:, 5, 5]

        if fluorescence_total is None:
            fluorescence_total = fluorescense
        else:
            fluorescence_total += fluorescense

    with Pogger("superspinsim-generate") as logger:
        # population = np.abs(density[:, 0, 0, 0] - 1/3)
        # log_population = np.log(population)
        # fit = np.polyfit(time, log_population, 1)
        # thermalisation_time = -1/fit[0]
        # print(thermalisation_time)

        # coherence = density[:, 0, 1, 0]

        # def sine_template(t, f, a, h, p):
        #     return a*np.cos(math.tau*f*t - p) + h

        # fit_parameters, fit_cov = spo.curve_fit(
        #     sine_template, time, coherence, (2.9e9, 1, 0, 0))
        # oscillation = fit_parameters[0]
        # field = oscillation/28e9
        # print(f"Frequency: {oscillation}")
        # print(f"Field: {field}")

        # log_coherence = np.log(coherence)
        # fit = np.polyfit(time, log_coherence, 1)
        # dephasing_time = -1/fit[0]
        # print(dephasing_time)

        @logger.record(("time", "fluorescense", "density"), ("s", None, None))
        def plot():
            plt.figure("odmr")
            plt.plot(
                time/1e-6, 100*fluorescence_total/np.max(fluorescence_total),
                "k-"
            )
            plt.xlabel("Time (us)")
            plt.ylabel("Fluorescence (%)")
            plt.draw()

            plt.figure("odmr-sweep")
            plt.plot(
                frequency_start/1e9 + frequency_width/1e9*(
                    time[time > time_start*15] - time_start)
                    / (time_end - time_start),
                100*fluorescence_total[time > time_start*15] \
                    / np.max(fluorescence_total),
                "k-"
            )
            plt.xlabel("Microwave frequency (GHz)")
            plt.ylabel("Fluorescence (%)")
            plt.gca().spines[["top", "right"]].set_visible(False)
            plt.draw()

            # labels = ["+", "0", "-"]
            # plt.figure("states")
            # for magnetic_index in range(3):
            #     plt.plot(
            #         time/1e-6,
            #         100*density[:, magnetic_index, magnetic_index, 0], "-",
            #         color=cm.hawaii(magnetic_index/3),
            #         label=f"g{labels[magnetic_index]}"
            #     )

            #     plt.plot(
            #         time/1e-6,
            #         100*density[:, 3 + magnetic_index, 3 + magnetic_index, 0],
            #         "--", color=cm.hawaii(magnetic_index/3),
            #         label=f"e{labels[magnetic_index]}"
            #     )
            # plt.plot(
            #     time/1e-6, 100*density[:, 6, 6, 0], "--",
            #     color=cm.hawaii(0.99), label="s")
            # plt.legend()
            # plt.xlabel("Time (us)")
            # plt.ylabel("Population (%)")
            # plt.draw()

            # plt.figure("coherences")
            # # plt.plot(time/1e-6, population, "k-", label="Population")
            # plt.plot(time/1e-6, coherence, "k-", label="Coherence")
            # plt.plot(
            #     time/1e-6, sine_template(time, *fit_parameters), "k--",
            #     label="Fit")
            # plt.xlabel("Time (us)")
            # plt.ylabel("Value")
            # plt.legend()
            # plt.draw()

            return time, fluorescense, density

        if hyperfine:
            logger.set_context("21-level")
        else:
            logger.set_context("7-level")
        plot()

    plt.show()
