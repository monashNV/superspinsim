import math
import numpy as np
from matplotlib import pyplot as plt
from cmcrameri import cm
from pogger import Read


def error_function(left: np.ndarray, right: np.ndarray):
    return np.average(
        (left.real - right.real)**2 + (left.imag - right.imag)**2)


def weight_function(error):
    return (1/error**20)


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

    centre_real = np.empty(
        (centre.shape[0], centre.shape[1], centre.shape[2], 2))
    centre_real[:, :, :, 0] = centre.real
    centre_real[:, :, :, 1] = centre.imag
    centre_flat = centre_real.flatten()

    densities_flat = []
    for density in densities:
        density_real = np.empty(
            (density.shape[0], density.shape[1], density.shape[2], 2))
        density_real[:, :, :, 0] = density.real
        density_real[:, :, :, 1] = density.imag
        density_flat = density_real.flatten()
        density_flat -= centre_flat
        densities_flat.append(density_flat)
    densities_flat = np.array(densities_flat)

    densities_flat = densities_flat[:, ::1000]

    pca_covariance = densities_flat.T@densities_flat
    pca_vals, pca_vecs = np.linalg.eigh(pca_covariance)
    densities_pca = pca_vecs[-2:, :]@densities_flat.T
    densities_pca = densities_pca.T

    densities_pca_scale = np.empty_like(densities_pca)
    for index, (density_pca, error) in \
            enumerate(zip(densities_pca, errors_centre)):
        norm = math.sqrt(density_pca[0]**2 + density_pca[1]**2)
        densities_pca_scale[index, :] = density_pca*error/norm

    densities_pca = densities_pca_scale

    plt.figure(label="pca")
    # plt.plot(densities_pca[:, 0], densities_pca[:, 1], "k-")
    for index, density_pca in enumerate(densities_pca):
        if index < densities_pca.shape[0] - 1:
            plt.plot(
                [density_pca[0], densities_pca[index + 1, 0]],
                [density_pca[1], densities_pca[index + 1, 1]], "-",
                color=cm.batlow(index/densities_pca.shape[0])
            )
        plt.plot(
            [density_pca[0]], [density_pca[1]], ".",
            color=cm.batlow(index/densities_pca.shape[0])
        )
    plt.plot([0], [0], "xr")
    plt.xlim([-np.max(errors_centre), np.max(errors_centre)])
    plt.ylim([-np.max(errors_centre), np.max(errors_centre)])
    plt.xlabel("PCA direction 1 from centre")
    plt.ylabel("PCA direction 2 from centre")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.draw()
