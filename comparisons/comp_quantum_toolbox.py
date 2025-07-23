import os
import h5py
import numpy as np

from matplotlib import pylab as plt
from cmcrameri import cm

from comparisons.lindbladians import contrast


def main(lindbladian="contrast"):
    if lindbladian == "contrast":
        generate_return, generate_return_comparison = contrast()

    coefficients, generators_coherent, generators_jump, density_initial, \
        time_step, time_end = generate_return_comparison

    with h5py.File("to_julia.h5", "w") as h5_file:
        h5_file.attrs["time_step"] = time_step
        h5_file.attrs["time_end"] = time_end
        h5_file["density_operator_initial"] = density_initial
        group_coherent = h5_file.create_group("generators_coherent")
        for index, generator_coherent in enumerate(generators_coherent):
            group_coherent[str(index + 1)] = generator_coherent
        group_jump_static = h5_file.create_group("generators_jump_static")
        for index, generator_jump in enumerate(generators_jump[1]):
            group_jump_static[str(index + 1)] = generator_jump
        group_jump_dynamic = h5_file.create_group("generators_jump_dynamic")
        for index, generator_jump in enumerate(generators_jump[0]):
            group_jump_dynamic[str(index + 1)] = generator_jump

        print(len(generators_jump[0]), len(generators_jump[1]))

    os.system("julia --project=julia_env comparisons/comp_quantum_toolbox.jl")

    with h5py.File("from_julia.h5", "r") as h5_file:
        time = np.asarray(h5_file["time"])
        density = np.asarray(h5_file["density"])

        sampled_jl_x = np.asarray(h5_file["sampled/x"])
        sampled_jl_l = np.asarray(h5_file["sampled/l"])

    os.remove("to_julia.h5")
    os.remove("from_julia.h5")

    print(density.shape)

    plt.figure(label="coefficients")

    plt.subplot(2, 2, 1)
    plt.plot(time/1e-6, sampled_jl_x/1e-3, "k-", label="Julia")
    plt.ylabel("Transverse magnetic field (mT)")
    plt.legend()

    plt.subplot(2, 2, 2)
    sampled_py_x = np.empty_like(time)
    for index, time_element in enumerate(time):
        sampled_py_x[index] = coefficients[0](time_element)
    plt.plot(time/1e-6, sampled_py_x/1e-3, "k-", label="Python")
    # plt.ylabel("Transverse magnetic field (mT)")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(time/1e-6, sampled_jl_l/1e-2, "k-", label="Julia")
    plt.ylabel("Excitation (% of decay rate)")
    plt.xlabel("Time (us)")
    plt.legend()

    plt.subplot(2, 2, 4)
    sampled_py_l = np.empty_like(time)
    for index, time_element in enumerate(time):
        sampled_py_l[index] = np.sqrt(coefficients[3](time_element))
    plt.plot(time/1e-6, sampled_py_l/1e-2, "k-", label="Python")
    # plt.ylabel("Excitation (% of decay rate)")
    plt.xlabel("Time (us)")
    plt.legend()

    plt.draw()

    fluorescence = \
        np.real(density[:, 3, 3] + density[:, 4, 4] + density[:, 5, 5])
    fluorescence /= np.max(fluorescence)

    plt.figure(label="fluorescence")
    plt.plot(time/1e-6, fluorescence/0.01, "k-", label="Fluoro")
    plt.plot(
        time/1e-6, density[:, 0, 0]/0.01, "-", color=cm.hawaii(0/3),
        label="(g) mS=+1"
    )
    plt.plot(
        time/1e-6, density[:, 1, 1]/0.01, "-", color=cm.hawaii(1/3),
        label="(g) mS=0"
    )
    plt.plot(
        time/1e-6, density[:, 2, 2]/0.01, "-", color=cm.hawaii(2/3),
        label="(g) mS=-1"
    )
    plt.plot(
        time/1e-6, density[:, 3, 3]/0.01, "--", color=cm.hawaii(0/3),
        label="(e) mS=+1"
    )
    plt.plot(
        time/1e-6, density[:, 4, 4]/0.01, "--", color=cm.hawaii(1/3),
        label="(e) mS=0"
    )
    plt.plot(
        time/1e-6, density[:, 5, 5]/0.01, "--", color=cm.hawaii(2/3),
        label="(e) mS=-1"
    )
    plt.plot(
        time/1e-6, density[:, 6, 6]/0.01, "-", color=cm.hawaii(0.99),
        label="(s)"
    )
    plt.xlabel("Time (us)")
    plt.ylabel("Fluorescence (%), State population (%)")
    plt.legend()
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.show()
