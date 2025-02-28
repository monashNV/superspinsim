from superspinsim import generate_simulator
from superspinsim.nv import lindbladians as nvl

import numpy as np
import math

from matplotlib import pyplot as plt
from cmcrameri import cm

from pogger import Pogger


def main():
    with Pogger("superspinsim-benchmarks") as pogger:
        @pogger.record(("number_of_fine_divisions", "density_operators"))
        def simulate(
                simulation_index,
                number_of_fine_divisions,
                sampler,
                density_operator_initial,
                time_start,
                time_end,
                time_step):
            number_of_fine_divisions = int(number_of_fine_divisions)
            print(f"| {simulation_index:8d} | {number_of_fine_divisions:8d} |")
            simulate = generate_simulator(
                sampler,
                number_of_fine_divisions=number_of_fine_divisions,
                number_of_quartic_repeats=int(math.ceil(
                    (math.log2(500e-12/number_of_fine_divisions) + 85.2)/2)),
                number_of_exponentials=2
            )

            time, density_operators = simulate(
                density_operator_initial, time_start, time_end, time_step)

            return number_of_fine_divisions, density_operators

        @pogger.record(("ground_truth", "divisions", "sample_rate", "errors"))
        def calculate_error(density_operators_list, divisions, time_step):
            print("Comparing")
            ground_truth = density_operators_list.pop()
            sample_rate = divisions[:-1]/time_step
            divisions = divisions[:-1]*ground_truth.shape[0]
            errors = []
            for density_operators in density_operators_list:
                difference = density_operators - ground_truth
                error = np.sqrt(np.mean(difference**2))
                errors.append(error)
            print("Done!")

            errors = np.array(errors)
            return ground_truth, divisions, sample_rate, errors

        @pogger.record()
        def plot_errors(divisions, errors):
            # Plot
            plt.figure()
            plt.loglog(divisions, errors, ".--", color=cm.hawaii(0))
            plt.xlabel("Number of time steps")
            plt.ylabel("Error (RMS)")
            plt.draw()

        datatype = np.float64
        wavefunction_size = 7

        # # Define sampler
        # resonance = 20
        # rabi = 2

        # def sampler(time, coefficient):
        #     coefficient[0] = 2*math.tau*rabi*math.cos(math.tau*resonance*time)
        #     coefficient[2] = math.tau*resonance

        # number_of_samples = int(2**16)
        # time_start: datatype = 0.0
        # time_step: datatype = 1/number_of_samples
        # time_end: datatype = 1.0

        # # Define initial condition
        # density_operator_initial = np.zeros(
        #     (wavefunction_size, wavefunction_size, 2),
        #     dtype=datatype
        # )
        # density_operator_initial[0, 0, 0] = 1
        # density_operator_initial[1, 1, 0] = 0
        # density_operator_initial[2, 2, 0] = 0

        sampler = nvl.rabi_excited
        time_start: datatype = 0.0
        time_step: datatype = 500e-12
        time_end: datatype = 12e-6
        # time_start: datatype = 0e-6
        # time_end: datatype = 6e-6
        # time_step: datatype = 500e-12

        density_operator_initial = np.zeros(
            (wavefunction_size, wavefunction_size, 2),
            dtype=datatype
        )
        density_operator_initial[6, 6, 0] = 1

        # Simulate with different fidelity
        print("Simulating")
        density_operators_list = []
        # divisions = np.geomspace(1, 10000, 6)  # np.geomspace(1, 10, 10)
        divisions = np.geomspace(1, 1024, 11)
        divisions = np.round(divisions)
        for simulation_index, number_of_fine_divisions in enumerate(divisions):
            pogger.set_context(f"density_matrices/{simulation_index}")
            _, density_operators = simulate(
                simulation_index,
                number_of_fine_divisions,
                sampler,
                density_operator_initial,
                time_start,
                time_end,
                time_step
            )
            density_operators_list.append(density_operators)
        print("Done!")

        pogger.set_context("error_analysis")
        ground_truth, divisions, sample_rate, errors = \
            calculate_error(density_operators_list, divisions, time_step)
        plot_errors(divisions, errors)

        with open("profile/datetime", "w") as file_datetime:
            file_datetime.write(pogger.get_datetime())
