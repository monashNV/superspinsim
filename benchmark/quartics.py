from superspinsim import generate_simulator
from superspinsim.nv import lindbladians as nvl

import numpy as np
import math

from matplotlib import pyplot as plt

from pogger import Pogger


def main():
    with Pogger("superspinsim-benchmarks") as pogger:
        @pogger.record(("number_of_fine_divisions", "density_operators"))
        def simulate(
                simulation_index,
                number_of_quartics,
                sampler,
                density_operator_initial,
                time_start,
                time_end,
                time_step):
            number_of_quartics = int(number_of_quartics)
            print(f"| {simulation_index:8d} | {number_of_quartics:8d} |")
            simulate = generate_simulator(
                sampler,
                number_of_fine_divisions=2,
                number_of_quartic_repeats=number_of_quartics,
                number_of_exponentials=1,
                use_cayley=True
            )

            time, density_operators = simulate(
                density_operator_initial, time_start, time_end, time_step)

            return number_of_quartics, density_operators

        @pogger.record(("ground_truth", "squares", "errors"))
        def calculate_error(density_operators_list, squares):
            print("Comparing")
            ground_truth = density_operators_list.pop()
            errors = []
            for density_operators in density_operators_list:
                difference = density_operators - ground_truth
                error = np.sqrt(np.mean(difference**2))
                errors.append(error)
            print("Done!")

            errors = np.array(errors)
            return ground_truth, squares[:-1], errors

        @pogger.record(("decay_fit", "scaling_fit"))
        def plot_errors(squares, errors):
            # Fit
            log10_errors = np.log10(errors)
            linear_fit = np.polyfit(squares, log10_errors, 1)
            decay_fit = -linear_fit[0]
            scaling_fit = math.pow(10, linear_fit[1])

            squares_fine = np.linspace(squares[0], squares[-1], 200)
            errors_fit = scaling_fit*np.power(10, -decay_fit*squares_fine)

            # Plot
            plt.figure(label="errors")
            plt.plot(
                squares, errors, "k.",
                label="Calculated"
            )
            plt.plot(
                squares_fine, errors_fit, "k--",
                label=f"Fit: Decay = {decay_fit:.2f}"
            )
            plt.yscale("log")
            plt.xlabel("Number of squares")
            plt.ylabel("Error (RMS)")
            plt.legend()
            plt.draw()

            return decay_fit, scaling_fit

        datatype = np.float64
        wavefunction_size = 7

        sampler = nvl.rabi_excited
        time_start: datatype = 0.0
        time_step: datatype = 200e-12
        time_end: datatype = 12e-6

        density_operator_initial = np.zeros(
            (wavefunction_size, wavefunction_size, 2),
            dtype=datatype
        )
        density_operator_initial[6, 6, 0] = 1

        # Simulate with different fidelity
        print("Simulating")
        density_operators_list = []
        number_of_quartics = np.arange(5, 30, 1)  # np.geomspace(1, 10, 10)
        # number_of_quartics = np.arange(25, 42, 1)  # np.geomspace(1, 10, 10)
        # number_of_quartics = np.arange(20, 60, 5)  # np.geomspace(1, 10, 10)
        for simulation_index, number_of_quartics_use in \
                enumerate(number_of_quartics):
            pogger.set_context(f"density_matrices/{simulation_index}")
            _, density_operators = simulate(
                simulation_index,
                number_of_quartics_use,
                sampler,
                density_operator_initial,
                time_start,
                time_end,
                time_step
            )
            density_operators_list.append(density_operators)
        print("Done!")

        pogger.set_context("quartics")

        ground_truth, squares, errors = \
            calculate_error(density_operators_list, number_of_quartics*2)

        plot_errors(squares, errors)

        with open("profile/datetime", "w") as file_datetime:
            file_datetime.write(pogger.get_datetime())
