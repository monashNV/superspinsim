import numpy as np
from matplotlib import pyplot as plt
from pogger import Read


def error_function(left: np.ndarray, right: np.ndarray):
    return np.average(
        (left.real - right.real)**2 + (left.imag - right.imag)**2)


def weight_function(error):
    return (1/error**6)


def main():
    project_name = "superspinsim-comparisons"
    timestamp_qutip = "2025-07-08T16-10-49"

    read = Read(project_name, timestamp_qutip)
    densities = read.read_array("density", "qutip")
    wall_durations, _ = read.read_array("wall_duration", "qutip")

    order = np.argsort(wall_durations)
    densities = densities[order]
    wall_durations = wall_durations[order]

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

    plt.figure(label="differential_errors")
    plt.loglog(wall_durations, errors_diff, "k.-")
    plt.xlabel("Time spent simulating (s)")
    plt.ylabel("Error between adjacent points")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.draw()

    centre = np.zeros_like(densities[0, :, :, :])
    for density, error in zip(densities, errors_diff):
        centre += density*weight_function(error)
    centre /= np.sum(weight_function(error))

    errors_centre = []
    for density in densities:
        error = error_function(density, centre)
        errors_centre.append(error)
    errors_centre = np.array(errors_centre)

    plt.figure(label="error_from_centre")
    plt.loglog(wall_durations, errors_centre, "k.-")
    plt.xlabel("Time spent simulating (s)")
    plt.ylabel("Error from centre")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.draw()
