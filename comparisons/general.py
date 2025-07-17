import numpy as np
from matplotlib import pyplot as plt
import time as tm

from pogger import Pogger


def error_function(left: np.ndarray, right: np.ndarray):
    left = np.array(left, dtype=np.complex256)
    right = np.array(right, dtype=np.complex256)
    error = np.sqrt(np.average(
        (left.real - right.real)**2 + (left.imag - right.imag)**2))
    return error


def calculate_errors_diff(densities: np.ndarray):
    errors_diff = []
    for trial_index in range(densities.shape[0]):
        if trial_index == 0:
            error = error_function(
                densities[trial_index, :, :, :],
                densities[trial_index + 1, :, :, :]
            )
        elif trial_index == densities.shape[0] - 1:
            error = error_function(
                densities[trial_index - 1, :, :, :],
                densities[trial_index, :, :, :]
            )
        else:
            error = error_function(
                densities[trial_index - 1, :, :, :],
                densities[trial_index, :, :, :]
            ) + error_function(
                densities[trial_index, :, :, :],
                densities[trial_index + 1, :, :, :]
            )
            error /= 2
        errors_diff.append(error)
    errors_diff = np.array(errors_diff)
    return errors_diff


def loop(init_function: callable, init_arguments: dict):
    trial_name, run_arguments, index, sweep_parameter = \
        init_function(**init_arguments)
    with Pogger("superspinsim-comparisons") as logger:
        run_wrap = logger.record(
            (
                run_arguments["sweep_parameter_code"], "wall_duration",
                "density", "strikes", "break", "index"
            ), (
                run_arguments["sweep_parameter_units"], "s", None, None, None,
                None
            )
        )(run)
        final_wrap = logger.record(
            (
                run_arguments["sweep_display_code"], "density",
                "wall_duration", "error"
            ), (run_arguments["sweep_display_units"], None, "s", None)
        )(final)

        while True:
            try:
                run_arguments["index"] = index
                run_arguments[run_arguments["sweep_parameter_code"]] = \
                    sweep_parameter

                logger.set_context(f"{trial_name}/trials/{index}")
                sweep_parameter, wall_duration, density, strikes_dict, \
                    do_break, index = run_wrap(**run_arguments)
                if do_break:
                    break
            except Exception as e:
                print(e)
                break

        logger.set_context(trial_name)
        final_wrap(
            run_arguments[run_arguments["sweep_display_code"]],
            run_arguments["sweep_display_label"], run_arguments["densities"],
            run_arguments["wall_durations"]
        )


def run(**run_arguments):
    wall_timeout = 1  # 30
    # wall_timeout = 60
    wall_timeout_std = 0.1
    wall_duration_list = []

    time_step = run_arguments["time_step"]
    densities = run_arguments["densities"]
    wall_durations = run_arguments["wall_durations"]
    strikes_dict = run_arguments["strikes_dict"]
    index = run_arguments["index"]
    run_raw = run_arguments["run_raw"]

    while True:
        wall_time_start = tm.perf_counter()

        time, density = run_raw(**run_arguments)

        wall_duration = tm.perf_counter() - wall_time_start
        wall_duration_list.append(wall_duration)
        wall_duration_mean = np.mean(wall_duration_list)
        wall_duration_std = np.std(wall_duration_list)
        if np.sum(wall_duration_list) > wall_timeout:
            break
        if len(wall_duration_list) > 1:
            if wall_duration_std/wall_duration_mean < wall_timeout_std:
                break

    if "fine_division" in run_arguments:
        fine_division = run_arguments["fine_division"]
        fine_steps = run_arguments["fine_steps"]
        fine_steps.append(time_step/fine_division)
        fine_division_multiple = strikes_dict["fine_division_multiple"]

    densities.append(density)
    wall_durations.append(wall_duration_mean)

    strikes = strikes_dict["strikes"]
    strike_aim = strikes_dict["strike_aim"]
    error_min = strikes_dict["error_min"]
    strikes_max = strikes_dict["strikes_max"]

    do_break = False
    if index > 1:
        errors = calculate_errors_diff(np.array(densities))
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

    strikes_dict["strikes"] = strikes
    strikes_dict["strike_aim"] = strike_aim
    strikes_dict["error_min"] = error_min
    strikes_dict["strikes_max"] = strikes_max

    index += 1
    if "fine_division" in run_arguments:
        fine_division_previous = fine_division
        fine_division *= fine_division_multiple
        fine_division = int(fine_division)
        if fine_division == fine_division_previous:
            fine_division += 1
        strikes_dict["fine_division_multiple"] = fine_division_multiple

    fluorescence = \
        np.real(density[:, 3, 3] + density[:, 4, 4] + density[:, 5, 5])
    fluorescence /= np.max(fluorescence)

    plt.figure(label="fluorescence")
    plt.plot(time/1e-6, fluorescence/0.01, "k-")
    plt.xlabel("Time (us)")
    plt.ylabel("Fluorescence (%)")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.draw()

    return fine_division, wall_duration, density, strikes_dict, do_break, index


def final(
        sweep_parameter: list, sweep_parameter_label: str, densities: list,
        wall_durations: list):
    sweep_parameter = np.array(sweep_parameter)
    densities = np.array(densities)
    wall_durations = np.array(wall_durations)

    errors = calculate_errors_diff(densities)

    plt.figure(label="errors_diff_wall_duration")
    plt.loglog(wall_durations, errors, "k.-")
    plt.xlabel("Simulation time (s)")
    plt.ylabel("Error from adjacent")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.draw()

    plt.figure(label="errors_diff_parameter")
    plt.loglog(sweep_parameter, errors, "k.-")
    plt.xlabel(sweep_parameter_label)
    plt.ylabel("Error from adjacent")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.draw()

    return sweep_parameter, densities, wall_durations, errors
