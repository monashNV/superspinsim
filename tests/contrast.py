import superspinsim as s3
from superspinsim.generate_generators import generate_7, generate_21

import math
import numpy as np
from matplotlib import pyplot as plt
from cmcrameri import cm


def main():
    from pogger import Pogger as Logger
    with Logger("superspinsim-generate") as logger:
        absolutes = []
        relatives = []
        amplitudes = []
        execute_record = logger.record(
            ("amplitude", "absolute", "relative", "density", "fluorescence")
        )(execute)
        for simulation_index, excitation_amplitude in \
                enumerate(np.geomspace(0.0001, 300, 400)):
            logger.set_context(f"simulations/{simulation_index}")
            excitation_amplitude, absolute_contrast, relative_contrast, _, _ =\
                execute_record(excitation_amplitude)
            absolutes.append(absolute_contrast)
            relatives.append(relative_contrast)
            amplitudes.append(excitation_amplitude)

        absolutes = np.array(absolutes)
        relatives = np.array(relatives)
        amplitudes = np.array(amplitudes)

        logger.set_context("contrast_vs_amplitude")

        @logger.record(("amplitudes", "absolutes", "relatives"))
        def plot():
            plt.figure(label="amplitudes")
            plt.loglog(
                amplitudes, np.abs(absolutes)/1e-9, "-", color=cm.hawaii(0/2))
            plt.xlabel("Excitation amplitude")
            plt.ylabel(
                "Contrast (excited population nanoseconds)",
                color=cm.hawaii(0/2)
            )
            plt.gca().spines["left"].set_edgecolor(cm.hawaii(0/2))
            plt.gca().tick_params(
                axis="y", which="both", colors=cm.hawaii(0/2),
                labelcolor=cm.hawaii(0/2)
            )
            plt.gca().spines[["top", "right"]].set_visible(False)

            right_axis = plt.twinx()
            right_axis.loglog(
                amplitudes, np.abs(relatives)/1e-9, "-", color=cm.hawaii(1/2))
            plt.xlabel("Excitation rate/relaxation rate")
            right_axis.set_ylabel(
                "Contrast (ppb peak fluorescence)", color=cm.hawaii(1/2))
            right_axis.spines["right"].set_edgecolor(cm.hawaii(1/2))
            right_axis.spines[["top", "left"]].set_visible(False)
            right_axis.tick_params(
                axis="y", which="both", colors=cm.hawaii(1/2),
                labelcolor=cm.hawaii(1/2)
            )
            plt.draw()

            return amplitudes, absolutes, relatives

        plot()
        plt.show()


def execute(excitation_amplitude):
    hyperfine = False
    duration_excitation = 3e-6
    duration_relax_wait = 250e-9
    duration_mw = 1/(2*10e6)
    frequency_mw = 2.87e9
    amplitude_mw = 2*10e6/28e9

    time_thermal_start = 0
    time_thermal_end = time_thermal_start + duration_excitation
    time_zero_start = duration_excitation + duration_relax_wait
    time_zero_end = time_zero_start + duration_excitation
    time_one_start = 2*duration_excitation + 2*duration_relax_wait \
        + duration_mw
    time_one_end = time_one_start + duration_excitation
    time_end = 10e-6

    quiescent_magnetic_field = np.array([0, 0, 1])*1e-12

    def coefficient_x(time):
        if time < 2*duration_excitation + 2*duration_relax_wait:
            return 0
        if time < 2*duration_excitation + 2*duration_relax_wait + duration_mw:
            return amplitude_mw*math.sin(math.tau*frequency_mw*time)
        return 0

    def coefficient_y(time):
        return 0

    def coefficient_z(time):
        return 0

    def coefficient_r(time):
        if time < duration_excitation:
            return excitation_amplitude
        if time < duration_excitation + duration_relax_wait:
            return 0
        if time < 2*duration_excitation + duration_relax_wait:
            return excitation_amplitude
        if time < 2*duration_excitation + 2*duration_relax_wait + duration_mw:
            return 0
        return excitation_amplitude

    coefficients = [coefficient_x, coefficient_y, coefficient_z, coefficient_r]
    if hyperfine:
        lindbladian, generators, vectorisation_map = generate_21(
            coefficients, quiescent_magnetic_field)
    else:
        lindbladian, generators, vectorisation_map = generate_7(
            coefficients, quiescent_magnetic_field)

    fine_step = 1e-9
    if hyperfine:
        coarse_step = 10e-9
    else:
        coarse_step = 2e-9
    number_of_divisions = int(round(coarse_step/fine_step))
    coarse_step = number_of_divisions*fine_step

    simulator = s3.generate_simulator(
        lindbladian, np.array(generators), vectorisation_map,
        number_of_fine_divisions=number_of_divisions, number_of_exponentials=2
    )
    if hyperfine:
        density_operator_initial = np.zeros((21, 21, 2))
        density_operator_initial[:9, :9, 0] = 1/9*np.eye(9)
    else:
        density_operator_initial = np.zeros((7, 7, 2))
        density_operator_initial[:3, :3, 0] = 1/3*np.eye(3)

    time, density = simulator(
        density_operator_initial, 0, time_end, coarse_step)

    if hyperfine:
        fluorescense = \
            density[:, 9, 9, 0] + density[:, 10, 10, 0] \
            + density[:, 11, 11, 0] + density[:, 12, 12, 0] \
            + density[:, 13, 13, 0] + density[:, 14, 14, 0] \
            + density[:, 15, 15, 0] + density[:, 16, 16, 0] \
            + density[:, 17, 17, 0]
    else:
        fluorescense = \
            density[:, 3, 3, 0] + density[:, 4, 4, 0] + density[:, 5, 5, 0]

    interval_zero = np.logical_and(
        time > time_zero_start, time < time_zero_end)
    interval_one = np.logical_and(
        time > time_one_start, time < time_one_end)
    absolute_contrast = (np.sum(fluorescense[interval_zero])
        - np.sum(fluorescense[interval_one]))*coarse_step
    relative_contrast = absolute_contrast/np.max(fluorescense)

    return (
        excitation_amplitude, absolute_contrast, relative_contrast, density, fluorescense
    )
    # print("Absolute contrast:", absolute_contrast)
    # print("Relative contrast:", relative_contrast)


def plot_odmr(fluorescence: np.ndarray):
    with Logger("superspinsim-generate") as logger:
        @logger.record(("time", "fluorescense", "density"), ("s", None, None))
        def plot():
            plt.figure("contrast")
            interval_thermal = np.logical_and(
                time > time_thermal_start, time < time_thermal_end)
            plt.plot(
                (time[interval_thermal] - time_thermal_start)/1e-6,
                100*fluorescense[interval_thermal]/np.max(fluorescense),
                "-", color=cm.hawaii(0/3), label="Thermal"
            )

            interval_zero = np.logical_and(
                time > time_zero_start, time < time_zero_end)
            plt.plot(
                (time[interval_zero] - time_zero_start)/1e-6,
                100*fluorescense[interval_zero]/np.max(fluorescense),
                "-", color=cm.hawaii(1/3), label="Zero-polarised"
            )

            interval_one = np.logical_and(
                time > time_one_start, time < time_one_end)
            plt.plot(
                (time[interval_one] - time_one_start)/1e-6,
                100*fluorescense[interval_one]/np.max(fluorescense),
                "-", color=cm.hawaii(2/3), label="One-polarised"
            )
            plt.ylim(50, 102)

            plt.xlabel("Time (us)")
            plt.ylabel("Fluorescence (%)")
            plt.legend()

            plt.draw()

            plt.figure("odmr")
            plt.plot(time/1e-6, 100*fluorescense/np.max(fluorescense), "k-")
            plt.xlabel("Time (us)")
            plt.ylabel("Fluorescence (%)")
            plt.draw()

            plt.figure("states")
            state_labels = [
                "(g), +", "(g), 0", "(g), -",
                "(e), +", "(e), 0", "(e), -", "(s)"]
            for state_index in range(density.shape[1]):
                plt.plot(
                    time/1e-6, 100*density[:, state_index, state_index, 0],
                    "-", color=cm.hawaii(0.999*state_index/density.shape[1]),
                    # label=state_labels[state_index]
                )
            plt.xlabel("Time (us)")
            plt.ylabel("Population (%)")
            plt.legend()
            plt.draw()
            return time, fluorescense, density

        if hyperfine:
            logger.set_context("21-level")
        else:
            logger.set_context("7-level")
        plot()

    plt.show()
