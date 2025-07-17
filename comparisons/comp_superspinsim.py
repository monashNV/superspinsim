import numpy as np
import math
import time as tm
from matplotlib import pyplot as plt

from pogger import Pogger

from superspinsim import generate_simulator
from comparisons.analysis import calculate_errors_diff
from comparisons.lindbladians import contrast


def main(use_rotating=False, number_of_exponentials=2):
    # Setup
    generate_return, generate_return_comparison = contrast(use_rotating)
    if use_rotating:
        lindbladian, generators_list, vectorisation_map, vectors_real, \
            inv_vectors_real, doubles, singles, _ = generate_return
    else:
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

    strikes_dict = {}
    strikes_dict["strikes"] = strikes
    strikes_dict["strike_aim"] = strike_aim
    strikes_dict["error_min"] = error_min
    strikes_dict["strikes_max"] = strikes_max
    strikes_dict["fine_division_multiple"] = fine_division_multiple

    with Pogger("superspinsim-comparisons") as logger:
        run_wrap = logger.record(
            (
                "fine_division", "wall_duration", "density", "strikes",
                "break", "index"
            ),
            (None, "s", None, None, None, None)
        )(run)
        final_wrap = logger.record(
            ("fine_division", "density", "wall_duration", "error"),
            (None, None, "s", None)
        )(final)

        while True:
            try:
                logger.set_context(f"superspinsim/trials/{index}")
                fine_division, wall_duration, density, strikes_dict, \
                    do_break, index = run_wrap(
                        index=index,
                        fine_division=fine_division,
                        densities=densities,
                        wall_durations=wall_durations,
                        fine_divisions=fine_divisions,
                        time_step=time_step,
                        time_end=time_end,
                        density_operator_initial=density_operator_initial,
                        generate_return=generate_return,
                        strikes_dict=strikes_dict,
                        use_rotating=use_rotating,
                        number_of_exponentials=number_of_exponentials
                    )
                if do_break:
                    break
            except Exception as e:
                print(e)
                break

        logger.set_context("superspinsim")
        final_wrap(fine_divisions, densities, wall_durations)


def run(
        index: int, fine_division: int, densities: list, wall_durations: list,
        fine_divisions: list, time_step: float, time_end: float,
        density_operator_initial: np.ndarray, generate_return: dict,
        strikes_dict: dict, use_rotating=False, number_of_exponentials=2):
    wall_timeout = 1  # 30
    # wall_timeout = 60
    wall_timeout_std = 0.1
    wall_duration_list = []

    while True:
        wall_time_start = tm.perf_counter()
        if use_rotating:
            lindbladian, generators_list, vectorisation_map, vectors_real, \
                inv_vectors_real, doubles, singles, _ = generate_return
            simulator = generate_simulator(
                lindbladian, generators=np.array(generators_list),
                vectorisation_map=vectorisation_map,
                number_of_exponentials=number_of_exponentials,
                number_of_fine_divisions=fine_division,
                use_rotating=True,
                vectors_real=vectors_real, inv_vectors_real=inv_vectors_real,
                doubles=doubles, singles=singles
            )
        else:
            lindbladian, generators_list, vectorisation_map, _ = \
                generate_return
            simulator = generate_simulator(
                lindbladian, generators=np.array(generators_list),
                vectorisation_map=vectorisation_map,
                number_of_exponentials=number_of_exponentials,
                number_of_fine_divisions=fine_division,
            )

        time, density = simulator(
            density_operator_initial, 0, time_end, time_step)

        wall_duration = tm.perf_counter() - wall_time_start
        wall_duration_list.append(wall_duration)
        wall_duration_mean = np.mean(wall_duration_list)
        wall_duration_std = np.std(wall_duration_list)
        if np.sum(wall_duration_list) > wall_timeout:
            break
        if len(wall_duration_list) > 1:
            if wall_duration_std/wall_duration_mean < wall_timeout_std:
                break

    fine_divisions.append(fine_division)
    densities.append(density)
    wall_durations.append(wall_duration_mean)

    strikes = strikes_dict["strikes"]
    strike_aim = strikes_dict["strike_aim"]
    error_min = strikes_dict["error_min"]
    strikes_max = strikes_dict["strikes_max"]
    fine_division_multiple = strikes_dict["fine_division_multiple"]

    do_break = False
    if index > 1:
        errors = calculate_errors_diff(
            np.asarray(densities)[:, :, 0]
            + 1j*np.asarray(densities)[:, :, 1]
        )
        error_min_current = np.min(errors)
        if error_min_current < strike_aim*error_min:
            error_min = error_min_current
            strikes = 0
        elif error_min < 1e-8:
            strikes += 1
            if strikes >= strikes_max:
                do_break = True
        print(
            index, len(wall_duration_list), wall_duration_mean,
            error_min_current, error_min, strikes
        )
    else:
        print(index, len(wall_duration_list), wall_duration_mean)

    index += 1
    fine_division_previous = fine_division
    fine_division *= fine_division_multiple
    fine_division = int(fine_division)
    if fine_division == fine_division_previous:
        fine_division += 1

    fluorescence = \
        density[:, 3, 3, 0] + density[:, 4, 4, 0] + density[:, 5, 5, 0]
    fluorescence /= np.max(fluorescence)

    plt.figure(label="fluorescence")
    plt.plot(time/1e-6, fluorescence/0.01, "k-")
    plt.xlabel("Time (us)")
    plt.ylabel("Fluorescence (%)")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.draw()

    strikes_dict["strikes"] = strikes
    strikes_dict["strike_aim"] = strike_aim
    strikes_dict["error_min"] = error_min
    strikes_dict["strikes_max"] = strikes_max
    strikes_dict["fine_division_multiple"] = fine_division_multiple

    return fine_division, wall_duration, density, strikes_dict, do_break, index


def final(fine_divisions: list, densities: list, wall_durations: list):
    errors = calculate_errors_diff(
        np.asarray(densities)[:, :, 0] + 1j*np.asarray(densities)[:, :, 1])

    fine_divisions = np.array(fine_divisions)
    densities = np.array(densities)
    wall_durations = np.array(wall_durations)

    plt.figure(label="errors_diff")
    plt.loglog(wall_durations, errors, "k.-")
    plt.xlabel("Simulation time (s)")
    plt.ylabel("Error from adjacent")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.draw()

    return fine_divisions, densities, wall_durations, errors
