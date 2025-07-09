import math
import numpy as np
from matplotlib import pyplot as plt
from cmcrameri import cm
from pogger import Read


def error_function(left: np.ndarray, right: np.ndarray):
    return np.sqrt(np.average(
        (left.real - right.real)**2 + (left.imag - right.imag)**2))


def weight_function(error):
    return (1/error**32)


def weight_function_pca(error):
    return (1/error**2)
    # return 1


def _circular_log(density_pca, smallest_error=-16):
    norm = math.sqrt(density_pca[0]**2 + density_pca[1]**2)
    error = np.log10(norm)
    print(error)
    error -= smallest_error
    error = max(0, error)
    return density_pca*error/norm


def circular_log(density_pca, smallest_error=-14):
    print(density_pca.shape)
    if len(density_pca.shape) == 1:
        return _circular_log(density_pca)
    else:
        density_cl = np.empty_like(density_pca)
        for index, density in enumerate(density_pca):
            density_cl[index, :] = _circular_log(density, smallest_error)
        return density_cl


def main():
    project_name = "superspinsim-comparisons"
    # timestamp_qutip = "2025-07-08T16-10-49"
    timestamp_qutip = "2025-07-08T16-10-49"

    timestamps = {
        "qutip": timestamp_qutip
    }

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
    for density, error in zip(densities, errors_centre):
        density_real = np.empty(
            (density.shape[0], density.shape[1], density.shape[2], 2))
        density_real[:, :, :, 0] = density.real
        density_real[:, :, :, 1] = density.imag
        density_flat = density_real.flatten()
        density_flat -= centre_flat
        density_flat *= weight_function_pca(error)
        densities_flat.append(density_flat)
    densities_flat = np.array(densities_flat)
    densities_flat /= np.max(np.abs(densities_flat))

    support = np.not_equal(densities_flat, 0)
    support = np.sum(support, axis=0)
    support = np.not_equal(support, 0)
    print("l0:", np.sum(support), "/", support.size)
    sample = np.arange(densities_flat.shape[1])
    sample = sample[support]

    np.random.seed(20250709**2 % 2**32)
    np.random.shuffle(sample)
    sample = sample[:densities_flat.shape[1]//15]
    densities_flat = densities_flat[:, sample]

    # sample = np.random.choice(
    #     np.arange(densities_flat.shape[1]),
    #     densities_flat.shape[1]//100, False
    # )
    # sample = np.arange(densities_flat.shape[1])
    # densities_flat = densities_flat[:, :densities_flat.shape[1]//15]
    # densities_flat = densities_flat[:, ::100]
    # print(densities_flat)

    pca_covariance = densities_flat.T@densities_flat
    pca_vals, pca_vecs = np.linalg.eigh(pca_covariance)
    # print(pca_vals)
    # print(pca_vecs@densities_flat.T)
    densities_pca = pca_vecs[-2:, :]@densities_flat.T
    densities_pca = densities_pca.T
    print(densities_pca)

    densities_pca_scale = np.empty_like(densities_pca)
    for index, (density_pca, error) in \
            enumerate(zip(densities_pca, errors_centre)):
        norm = math.sqrt(density_pca[0]**2 + density_pca[1]**2)
        densities_pca_scale[index, :] = density_pca*error/norm

    densities_pca = densities_pca_scale
    smallest_error = -16
    largest_error = -4
    densities_pca = circular_log(densities_pca, smallest_error)

    plt.figure(label="pca")
    plt.rcParams['text.usetex'] = True
    angle = np.linspace(0, math.tau)
    circle_x = np.cos(angle)
    circle_y = np.sin(angle)
    for radius in range(0, largest_error - smallest_error + 4, 4):
        plt.text(
            0, radius, r"$10^{" + f"{radius + smallest_error}" + r"}$",
            alpha=0.5, va="bottom"
        )
        plt.plot(radius*circle_x, radius*circle_y, "k-", alpha=0.5)
    plt.plot([0], [0], "kx")

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
    # plt.plot([0], [0], "xr")
    # plt.xlim([-np.max(errors_centre), np.max(errors_centre)])
    # plt.ylim([-np.max(errors_centre), np.max(errors_centre)])
    plt.xlabel("PCA direction 1 from centre")
    plt.ylabel("PCA direction 2 from centre")
    # plt.gca().spines[["top", "right"]].set_visible(False)
    plt.axis("off")
    plt.draw()

    return (timestamps,)
