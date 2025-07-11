import math
import numpy as np
from matplotlib import pyplot as plt
from cmcrameri import cm
from pogger import Read

from comparisons.analysis import calculate_errors_diff, error_function

weight_power = 48
weight_power = 0
weight_power_pca = 2


def weight_function(error):
    # inverse_weight = (error/1e-14)**weight_power
    # weight = 1/inverse_weight
    # print(weight)
    # weight[weight == np.nan] == 0
    # return weight
    return 1


def weight_function_pca(error):
    # return (1/error**weight_power_pca)
    return 1


def _circular_log(density_pca, smallest_error=-16):
    norm = math.sqrt(density_pca[0]**2 + density_pca[1]**2)
    error = np.log10(norm)
    # print(error)
    error -= smallest_error
    error = max(0, error)
    return density_pca*error/norm


def circular_log(density_pca, smallest_error=-14):
    # print(density_pca.shape)
    if len(density_pca.shape) == 1:
        return _circular_log(density_pca)
    else:
        density_cl = np.empty_like(density_pca)
        for index, density in enumerate(density_pca):
            density_cl[index, :] = _circular_log(density, smallest_error)
        return density_cl


def read_archives(protocols: dict[str, dict]):
    project_name = "superspinsim-comparisons"
    for protocol, protocol_data in protocols.items():
        read = Read(project_name, protocol_data["timestamp"])
        if "data_label" in protocol_data:
            protocol = protocol_data["data_label"]
        densities = read.read_array("density", protocol)
        wall_durations, _ = read.read_array("wall_duration", protocol)

        order = np.argsort(wall_durations)
        densities = densities[order]
        wall_durations = wall_durations[order]

        if len(densities.shape) == 5:
            densities = densities[:, :, :, :, 0] + 1j*densities[:, :, :, :, 1]

        densities = densities[:, -1000:, :, :]

        protocol_data["densities"] = densities
        protocol_data["wall_durations"] = wall_durations


def find_centre(protocol: dict):
    densities = protocol["densities"]
    wall_durations = protocol["wall_durations"]

    errors_diff = calculate_errors_diff(densities)

    centre_index = np.argmin(errors_diff)
    centre = densities[centre_index, :, :, :]
    # centre = np.zeros_like(densities[0, :, :, :])
    # for density, error in zip(densities, errors_diff):
    #     centre += density*weight_function(error)
    # centre /= np.sum(weight_function(error))

    protocol["centre"] = centre
    protocol["centre_weights"] = weight_function(errors_diff)
    protocol["centre_weight_power"] = weight_power
    protocol["errors_diff"] = errors_diff

    plt.figure(label="differential_errors")
    plt.loglog(wall_durations, errors_diff, "k.-")
    plt.xlabel("Time spent simulating (s)")
    plt.ylabel("Error between adjacent points")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.draw()


def find_errors_from_centre(protocols: dict[str, dict], centre: np.ndarray):
    for protocol, protocol_data in protocols.items():
        densities = protocol_data["densities"]

        errors_centre = []
        for density in densities:
            error = error_function(density, centre)
            errors_centre.append(error)
        errors_centre = np.array(errors_centre)

        protocol_data["errors_centre"] = errors_centre

    plt.figure(label="error_from_centre")
    for index, (protocol, protocol_data) in enumerate(protocols.items()):
        errors_centre = protocol_data["errors_centre"]
        wall_durations = protocol_data["wall_durations"]
        plot_marker = protocol_data["plot_marker"]
        plt.loglog(
            wall_durations, errors_centre, plot_marker + "-",
            color=cm.hawaii(index/len(protocols)), label=protocol
        )
    plt.xlabel("Time spent simulating (s)")
    plt.ylabel("Error from centre")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.legend()
    plt.draw()


def flatten(density: np.ndarray):
    density_real = np.empty(
        (density.shape[0], density.shape[1], density.shape[2], 2))
    density_real[:, :, :, 0] = density.real
    density_real[:, :, :, 1] = density.imag
    density_flat = density_real.flatten()
    return density_flat


def pca(protocols: dict[str, dict], centre: np.ndarray):
    # Flatten
    centre_flat = flatten(centre)

    for protocol, protocol_data in protocols.items():
        densities = protocol_data["densities"]
        errors_centre = protocol_data["errors_centre"]

        densities_flat = []
        for density, error in zip(densities, errors_centre):
            density_flat = flatten(density)
            density_flat -= centre_flat
            density_flat *= weight_function_pca(error)
            densities_flat.append(density_flat)
        densities_flat = np.array(densities_flat)
        densities_flat /= np.max(np.abs(densities_flat))

        protocol_data["densities_flat"] = densities_flat

    # Compile all densities
    densities_flat = []
    for protocol, protocol_data in protocols.items():
        densities_flat_partial = protocol_data["densities_flat"]
        for density_flat in densities_flat_partial:
            densities_flat.append(density_flat)
    densities_flat = np.array(densities_flat)

    # Subsample meaningful entries (entire vector is huge)
    support = np.not_equal(densities_flat, 0)
    support = np.sum(support, axis=0)
    support = np.not_equal(support, 0)
    # 2i# print("l0:", np.sum(support), "2i/", support.size)
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

    # Find principle components
    pca_data = {}
    pca_covariance = densities_flat.T@densities_flat
    pca_vals, pca_vecs = np.linalg.eigh(pca_covariance)
    pca_data["vals"] = pca_vals
    pca_data["vecs"] = pca_vecs
    # print(pca_vals)
    # print(pca_vecs@densities_flat.T)

    # Get principle components of each protocol
    for protocol, protocol_data in protocols.items():
        densities_flat = protocol_data["densities_flat"]
        densities_pca = pca_vecs[-2:, :]@densities_flat[:, sample].T
        densities_pca = densities_pca.T
        # print(densities_pca)
        protocol_data["densities_pca"] = densities_pca

    return pca_data


def plot_circular_log(protocols: dict[str, dict]):
    smallest_error = -17
    largest_error = -4

    # Calculate circular log
    for protocol, protocol_data in protocols.items():
        densities_pca = protocol_data["densities_pca"]
        errors_centre = protocol_data["errors_centre"]

        densities_pca_scale = np.empty_like(densities_pca)
        for index, (density_pca, error) in \
                enumerate(zip(densities_pca, errors_centre)):
            norm = math.sqrt(density_pca[0]**2 + density_pca[1]**2)
            densities_pca_scale[index, :] = density_pca*error/norm

        densities_pca = densities_pca_scale
        densities_pca = circular_log(densities_pca, smallest_error)
        protocol_data["densities_pca_scale"] = densities_pca

    plt.figure(label="pca")
    plt.rcParams['text.usetex'] = True

    # Plot grid
    angle = np.linspace(0, math.tau)
    circle_x = np.cos(angle)
    circle_y = np.sin(angle)
    grid_alpha = 0.1
    circle_resolution = 1
    circle_resolution_large = 4
    text_offset = (largest_error - smallest_error)/50
    for radius in range(
            0, largest_error - smallest_error + circle_resolution,
            circle_resolution):
        if radius % circle_resolution_large == 0:
            text = r"$10^{" + f"{radius + smallest_error}" + r"}$"
            if radius == 0:
                text = r"$\le$" + text + r" (64b prec.)"
            elif radius + circle_resolution > largest_error - smallest_error:
                text += " error from QuTip GT"
            plt.text(text_offset, text_offset + radius, text, va="bottom")
        if radius > 0:
            if radius % circle_resolution_large == 0:
                plt.plot(
                    radius*circle_x, radius*circle_y, "k-", alpha=grid_alpha)
            else:
                plt.plot(
                    radius*circle_x, radius*circle_y, "k--", alpha=grid_alpha)

    radius = largest_error - smallest_error
    angle_resolution = 12
    angle_resolution_large = 3
    for angle in np.arange(0, angle_resolution):
        if angle % angle_resolution_large == 0:
            format = "k-"
        else:
            format = "k--"

        angle *= math.pi/angle_resolution
        plt.plot(
            [radius*math.cos(angle), -radius*math.cos(angle)],
            [radius*math.sin(angle), -radius*math.sin(angle)],
            format, alpha=grid_alpha
        )

    plt.plot([0], [0], "kx")
    plt.gca().set_aspect("equal")

    # Plot convergence
    for protocol, protocol_data in protocols.items():
        for index, density_pca in enumerate(densities_pca):
            densities_pca = protocol_data["densities_pca_scale"]
            plot_marker = protocol_data["plot_marker"]

            if index < densities_pca.shape[0] - 1:
                plt.plot(
                    [density_pca[0], densities_pca[index + 1, 0]],
                    [density_pca[1], densities_pca[index + 1, 1]], "-",
                    color=cm.batlow(index/densities_pca.shape[0])
                )
            if index == 0:
                plt.plot(
                    [density_pca[0]], [density_pca[1]], plot_marker,
                    color=cm.batlow(index/densities_pca.shape[0]),
                    label=protocol
                )
            else:
                plt.plot(
                    [density_pca[0]], [density_pca[1]], plot_marker,
                    color=cm.batlow(index/densities_pca.shape[0])
                )
    # plt.plot([0], [0], "xr")
    # plt.xlim([-np.max(errors_centre), np.max(errors_centre)])
    # plt.ylim([-np.max(errors_centre), np.max(errors_centre)])
    # plt.xlabel("PCA direction 1 from centre")
    # plt.ylabel("PCA direction 2 from centre")
    # plt.gca().spines[["top", "right"]].set_visible(False)
    plt.axis("off")
    plt.legend()
    plt.draw()


def main():
    protocols = {
        "qutip": {
            # "timestamp": "2025-07-08T16-10-49",
            "timestamp": "2025-07-08T19-01-21",
            "plot_marker": "^"
        },

        "s3_pc": {
            "timestamp": "2025-07-11T16-14-55",
            "plot_marker": ".",
            "data_label": "superspinsim"
        },

        "s3_metaoptics": {
            "timestamp": "2025-07-11T18-31-56",
            "plot_marker": "o",
            "data_label": "superspinsim"
        }
    }
    protocol_ground_truth = "s3_metaoptics"

    read_archives(protocols)
    find_centre(protocols[protocol_ground_truth])
    centre = protocols[protocol_ground_truth]["centre"]
    find_errors_from_centre(protocols, centre)
    # pca_data = 0
    pca_data = pca(protocols, centre)
    plot_circular_log(protocols)

    return (protocols, pca_data)
