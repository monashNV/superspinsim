import numpy as np
from matplotlib import pyplot as plt


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
