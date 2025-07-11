import numpy as np


def error_function(left: np.ndarray, right: np.ndarray):
    return np.sqrt(np.average(
        (left.real - right.real)**2 + (left.imag - right.imag)**2))


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
