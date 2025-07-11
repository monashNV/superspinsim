import numpy as np
import math
import time as tm
from matplotlib import pyplot as plt

from superspinsim import generate_simulator
from comparisons.analysis import calculate_errors_diff
from comparisons.lindbladians import contrast


def main():
    # Setup
    generate_return, generate_return_comparison = contrast(False)
    # lindbladian, generators_list, vectorisation_map, vectors_real, \
    #     inv_vectors_real, doubles, singles, _ = generate_return
    lindbladian, generators_list, vectorisation_map, _ = generate_return
    time_step = generate_return_comparison[-2]
    time_end = generate_return_comparison[-1]

    density_operator_initial = np.zeros((7, 7, 2))
    density_operator_initial[0, 0, 0] = 1/3
    density_operator_initial[1, 1, 0] = 1/3
    density_operator_initial[2, 2, 0] = 1/3

    densities = []
    wall_durations = []
    fine_divisions = []

    fine_division = 16
    fine_division_multiple = 1.3

    error_min = math.inf
    strikes = 0
    strikes_max = 3
    strike_aim = 3/4
    index = 0
    while True:
        wall_time_start = tm.perf_counter()

        simulator = generate_simulator(
            lindbladian, generators=np.array(generators_list),
            vectorisation_map=vectorisation_map, number_of_exponentials=5,
            number_of_fine_divisions=fine_division,
            use_rotating=True,
            # vectors_real=vectors_real, inv_vectors_real=inv_vectors_real,
            # doubles=doubles, singles=singles
        )
        _, density = simulator(
            density_operator_initial, 0, time_end, time_step)

        wall_duration = tm.perf_counter() - wall_time_start

        fine_divisions.append(fine_division)
        densities.append(density)
        wall_durations.append(wall_duration)

        if index > 1:
            errors = calculate_errors_diff(
                np.asarray(densities)[:, :, 0]
                + 1j*np.asarray(densities)[:, :, 1]
            )
            error_min_current = np.min(errors)
            if error_min_current < strike_aim*error_min:
                error_min = error_min_current
                strikes = 0
            else:
                strikes += 1
                if strikes >= strikes_max:
                    break
            print(index, wall_duration, error_min_current, error_min, strikes)
        else:
            print(index, wall_duration)

        index += 1
        fine_division_previous = fine_division
        fine_division *= fine_division_multiple
        fine_division = int(fine_division)
        if fine_division == fine_division_previous:
            fine_division += 1

    fine_divisions = np.array(fine_divisions)
    densities = np.array(densities)
    wall_durations = np.array(wall_durations)

    plt.figure(label="errors_diff")
    plt.loglog(wall_durations, errors, "k.-")
    plt.draw()

    return fine_divisions, densities, wall_durations, errors
