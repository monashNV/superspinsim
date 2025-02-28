from superspinsim.util import colour_complex_matrix as _colour_complex_matrix
from superspinsim import generate_simulator

import math
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image as pli
from cmcrameri import cm

# import scipy.linalg as sl


def plot_populations(time, density_operators):
    print("Plotting populations...")
    state_labels = ["+g", "0g", "-g", "+e", "0e", "-e", "s"]
    plt.figure()
    for state_index in range(density_operators.shape[1]):
        plt.plot(
            time,
            100*density_operators[:, state_index, state_index, 0],
            "-",
            color=cm.hawaii(state_index/density_operators.shape[1]),
            label=state_labels[state_index]
        )
    plt.xlabel("Time")
    plt.ylabel("Population (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.draw()
    print("Done!")


def visualise_time_evolution(density_operators=None, time_evolution=None):
    print("Creating animations...")

    if density_operators is not None:
        frames = []

        # plt.figure()
        for density_operator_index in range(density_operators.shape[0]):
            # plt.subplot(6, 10, density_operator_index + 1)
            coloured = _colour_complex_matrix(
                density_operators[density_operator_index, :, :, :])
            # plt.imshow(coloured)
            # plt.axis("off")

            scale = 20
            frame = np.empty(
                (scale*density_operators.shape[1],
                 scale*density_operators.shape[1], 3),
                dtype=np.uint8
            )
            for x_index in range(scale):
                for y_index in range(scale):
                    frame[y_index::scale, x_index::scale] = coloured*255
                    progress = int(frame.shape[1]*(density_operator_index + 1)
                                   / density_operators.shape[0])
                    frame[-4:, :progress] = np.array([255, 255, 255])
            frame = pli.fromarray(frame)
            frames.append(frame)

        frames[0].save(
            "density_operator.gif",
            save_all=True,
            append_images=frames[1:],
            duration=1e3/density_operators.shape[0],
            loop=0
        )

    if time_evolution is not None:
        transforms_true = time_evolution.copy()
        transforms_true[:, :, :] += np.eye(transforms_true.shape[1])

        frames = []

        # plt.figure()
        for transform_index in range(transforms_true.shape[0]):
            # plt.subplot(6, 10, transform_index + 1)
            coloured = _colour_complex_matrix(
                transforms_true[transform_index, :, :])
            # plt.imshow(coloured)
            # plt.axis("off")

            scale = 10
            frame = np.empty(
                (scale*transforms_true.shape[1],
                 scale*transforms_true.shape[1], 3),
                dtype=np.uint8
            )
            for x_index in range(scale):
                for y_index in range(scale):
                    frame[y_index::scale, x_index::scale] = coloured*255
                    progress = int(frame.shape[1]*(transform_index + 1)
                                   / transforms_true.shape[0])
                    frame[-4:, :progress] = np.array([255, 255, 255])
            frame = pli.fromarray(frame)
            frames.append(frame)

        frames[0].save(
            "time_evolution.gif",
            save_all=True,
            append_images=frames[1:],
            duration=1e3/15,
            loop=0
        )


def test_time_sample_quadrature():
    print("Testing time sampling...")

    # User input --------------------------------------------------------------

    datatype = np.float64
    wavefunction_size = 7

    # Define sampler
    resonance = 20
    rabi = 2

    def sampler(time, coefficient):
        coefficient[0] = 2*math.tau*rabi*math.cos(math.tau*resonance*time)
        coefficient[2] = math.tau*resonance

    number_of_samples = 256
    time_start: datatype = 0.0
    time_step: datatype = 1/number_of_samples
    time_end: datatype = 1.0

    # Define initial condition
    density_operator_initial = np.zeros(
        (wavefunction_size, wavefunction_size, 2),
        dtype=datatype
    )
    density_operator_initial[0, 0, 0] = 1
    density_operator_initial[1, 1, 0] = 0
    density_operator_initial[2, 2, 0] = 0

    simulate = generate_simulator(sampler)

    # User input end ----------------------------------------------------------

    time, density_operators = simulate(
        density_operator_initial, time_start, time_end, time_step)

    # Visualise
    visualise_time_evolution(density_operators[::8, :, :, :], None)
    plot_populations(time, density_operators)

    print("Done!")


# During development, broken now
# def test_generators():
#     print("Testing generators...")
#     superoperators = np.array(
#         list(superoperator_basis_dict.values()), dtype=datatype
#     )
# 
#     superoperators *= 100
# 
#     if use_cuda:
#         superoperators_device = nc.to_device(
#             superoperators/(2**(2*number_of_quartic_repeats))
#         )
#     else:
#         superoperators_device = superoperators \
#             / (2**(2*number_of_quartic_repeats))
# 
#     _repeated_quartic_superoperator_run(superoperators_device)
#     if use_cuda:
#         transforms = superoperators_device.copy_to_host()
# 
#     transforms_true = transforms.copy()
#     transforms_true[:, :, :, 0] += np.eye(operator_size)
# 
#     # Visualise
#     plt.figure()
#     for transform_index in range(transforms_true.shape[0]):
#         plt.subplot(3, 5, transform_index + 1)
#         plt.imshow(_colour_complex_matrix(
#             transforms_true[transform_index, :, :, :])
#         )
#         plt.title(list(superoperator_basis_dict.keys())[transform_index])
#         plt.axis("off")
# 
#     print("Done!")
# 
# 
# def test_squaring():
#     print("Testing squaring")
# 
#     number_of_superoperators = int(1e2)
#     superoperators = np.random.normal(
#         size=(
#             number_of_superoperators,
#             operator_size,
#             operator_size,
#             2
#         ),
#     )
#     superoperators = np.array(superoperators, dtype=datatype)
#     superoperators /= 1e6*wavefunction_size
# 
#     superoperators_visualise = _colour_complex_matrix(
#         superoperators[0, :, :, :]
#     )
# 
#     plt.figure()
#     plt.imshow(superoperators_visualise)
#     plt.title("Before")
#     plt.draw()
# 
#     if use_cuda:
#         superoperators_device = nc.to_device(
#             superoperators/(2**(2*number_of_quartic_repeats))
#         )
#     else:
#         superoperators_device = superoperators \
#             / (2**(2*number_of_quartic_repeats))
# 
#     _repeated_quartic_superoperator_run(superoperators_device)
#     if use_cuda:
#         transforms = superoperators_device.copy_to_host()
# 
#     transforms_true = transforms.copy()
#     transforms_true[:, :, :, 0] += np.eye(operator_size)
# 
#     # Comparison with scipy.linalg
#     do_compare_sl = True
#     if do_compare_sl:
#         superoperator_true_complex = superoperators[0, :, :, 0] \
#             + 1j*superoperators[0, :, :, 1]
#         transform_true_complex_sl = sl.expm(superoperator_true_complex)
#         transform_true_sl = np.empty(
#             (operator_size, operator_size, 2), dtype=datatype
#         )
#         transform_true_sl[:, :, 0] = np.real(transform_true_complex_sl)
#         transform_true_sl[:, :, 1] = np.imag(transform_true_complex_sl)
# 
#         transform_true_error_sl = transforms_true[0, :, :, :] \
#             - transform_true_sl
# 
#     # Visualise
#     transform_true_visualise = _colour_complex_matrix(
#         transforms_true[0, :, :, :]
#     )
#     if do_compare_sl:
#         transform_true_sl_visualise = _colour_complex_matrix(
#             transform_true_sl[:, :, :]
#         )
#         transform_true_error_sl_visualise = _colour_complex_matrix(
#             transform_true_error_sl[:, :, :]
#         )
# 
#         transform_true_error_sl_f = np.sqrt(
#             np.sum(transform_true_error_sl**2)
#         )/(operator_size**2)
#         print(f"Error (Frobenius): {transform_true_error_sl_f}")
# 
#     plt.figure()
#     if do_compare_sl:
#         plt.subplot(1, 3, 1)
#     plt.imshow(transform_true_visualise)
#     plt.title("After")
# 
#     if do_compare_sl:
#         plt.subplot(1, 3, 2)
#         plt.imshow(transform_true_sl_visualise)
#         plt.title("Scipy\nground truth")
# 
#         plt.subplot(1, 3, 3)
#         plt.imshow(transform_true_error_sl_visualise)
#         plt.title("Error")
#         plt.draw()
# 
#     print("Done!")
# 
# 
# def test_combination():
#     print("Testing basic combination...")
#     superoperators = np.empty(
#         (60, operator_size, operator_size, 2),
#         dtype=datatype
#     )
#     superoperators[:, :, :, :] = (math.tau/60)*(
#         2*generators[0, :, :, :] + 2*generators[2, :, :, :]
#         + 5*generators[5, :, :, :] + 5*generators[6, :, :, :]
#         + 10*generators[3, :, :, :] + 10*generators[7, :, :, :]
#         + generators[8, :, :, :] + generators[10, :, :, :]
#         + 2*generators[9, :, :, :]
#         + 2*generators[13, :, :, :]
#     )
# 
#     density_operator_initial = np.zeros(
#         (wavefunction_size, wavefunction_size, 2),
#         dtype=datatype
#     )
#     density_operator_initial[0, 0, 0] = 1/3
#     density_operator_initial[1, 1, 0] = 1/3
#     density_operator_initial[2, 2, 0] = 1/3
# 
#     density_operator_initial_flat = density_operator_initial.reshape(
#         (operator_size, 2))
# 
#     if use_cuda:
#         superoperators_device = nc.to_device(
#             superoperators/(2**(2*number_of_quartic_repeats)))
#         density_operator_initial_device = nc.to_device(
#                               density_operator_initial_flat)
#         density_operators_device = nc.device_array(
#             (superoperators_device.shape[0],
#              operator_size, 2),
#             dtype=datatype)
#     else:
#         superoperators_device = superoperators \
#             / (2**(2*number_of_quartic_repeats))
# 
#     _repeated_quartic_superoperator_run(superoperators_device)
# 
#     _basic_combine_run(superoperators_device)
# 
#     _apply_time_evolution_run(superoperators_device,
#                               density_operator_initial_device,
#                               density_operators_device)
# 
#     if use_cuda:
#         transforms = superoperators_device.copy_to_host()
#         density_operators = density_operators_device.copy_to_host()
# 
#     density_operators = density_operators.reshape((density_operators.shape[0],
#                                                    wavefunction_size,
#                                                    wavefunction_size, 2))
# 
#     visualise_time_evolution(density_operators, transforms)
# 
#     print("Done!")
# def test_time_sample():
#     print("Testing time sampling...")
# 
#     number_of_samples = 512
#     if use_cuda:
#         time_device = nc.device_array(number_of_samples, dtype=datatype)
#         time_sample_device = nc.device_array(
#             number_of_samples, dtype=datatype)
# 
#     time_start: datatype = 0.0
#     time_step: datatype = 1/512
#     _calculate_time_basic_run(
#         time_device, time_sample_device, time_start, time_step)
# 
#     if use_cuda:
#         time = time_device.copy_to_host()
#         time_device = None
# 
#     # Sample coefficients
#     generators = np.array(
#         list(superperator_basis_dict.values()), dtype=datatype
#     )
# 
#     decay_amp = 50
# 
#     pulse_amp = 100
#     pulse_time_0 = 0.1
# 
#     def sampler(time, coefficient):
#         coefficient[8] = decay_amp
#         coefficient[9] = 1.5*decay_amp
#         coefficient[10] = decay_amp/3
#         coefficient[11] = decay_amp/2
# 
#         coefficient[3] = 80
# 
#         if time < 0.1:
#             pass
#         elif time < 0.15:
#             coefficient[13] = pulse_amp
#         elif time < 0.2:
#             pass
#         elif time < 0.5:
#             coefficient[0] = 20
#         elif time < 0.85:
#             pass
#         elif time < 0.9:
#             coefficient[13] = pulse_amp
#         else:
#             pass
# 
#     sample_run = _generate_sampler(sampler)
# 
#     if use_cuda:
#         coefficients_device = nc.device_array(
#             (time_sample_device.size, generators.shape[0]),
#             dtype=datatype
#         )
# 
#     sample_run(time_sample_device, coefficients_device)
# 
#     # Scale generators
#     if use_cuda:
#         generators_device = nc.to_device(generators)
#         superoperators_device = nc.device_array(
#             (
#                 time_sample_device.shape[0], operator_size,
#                 operator_size, 2
#             ), dtype=datatype)
# 
#     _calculate_differential_run(
#         time_step, generators_device,
#         coefficients_device, superoperators_device
#     )
# 
#     if use_cuda:
#         # coefficients = coefficients_device.copy_to_host()
#         # print(coefficients)
#         coefficients_device = None
#         generators_device = None
# 
#     # Exponentiate
#     _scale_differential_basic_run(superoperators_device)
#     _repeated_quartic_superoperator_run(superoperators_device)
# 
#     # Integrate
#     _basic_combine_run(superoperators_device)
# 
#     # Apply
#     density_operator_initial = np.zeros(
#         (wavefunction_size, wavefunction_size, 2),
#         dtype=datatype
#     )
#     density_operator_initial[0, 0, 0] = 1/3
#     density_operator_initial[1, 1, 0] = 1/3
#     density_operator_initial[2, 2, 0] = 1/3
# 
#     density_operator_initial_flat = density_operator_initial.reshape(
#         (operator_size, 2))
# 
#     if use_cuda:
#         density_operator_initial_device = nc.to_device(
#                               density_operator_initial_flat)
#         density_operators_device = nc.device_array(
#             (superoperators_device.shape[0],
#              operator_size, 2),
#             dtype=datatype)
# 
#     _apply_time_evolution_run(
#         superoperators_device,
#         density_operator_initial_device,
#         density_operators_device
#     )
# 
#     if use_cuda:
#         time_evolution = superoperators_device.copy_to_host()
#         superoperators_device = None
#         density_operators = density_operators_device.copy_to_host()
#         density_operators_device = None
# 
#     density_operators = density_operators.reshape((density_operators.shape[0],
#                                                    wavefunction_size,
#                                                    wavefunction_size, 2))
# 
#     visualise_time_evolution(density_operators, time_evolution)
# 
#     print("Done")

def scipy_exponentiate():
    from pogger import Pogger as Logger

    with Logger("superspinsim-benchmarks") as logger:

        @logger.record(("superoperators"))
        def run_scipy_exponentiate():
            import scipy.linalg as sl
            from superspinsim.generators import superoperators \
                as superoperators_dict

            superoperator = list(superoperators_dict.values())[0]
            superoperator *= math.tau
            times = np.arange(100)/100

            superoperators = []
            for time in times:
                superoperators.append(sl.expm(superoperator*time))

            superoperators = np.array(superoperators)

            plt.figure("scipy_exponenentiate")
            plt.plot(
                times, 100*superoperators[:, 0, 0], "-",
                color=cm.hawaii(0/3), label="+"
            )
            plt.plot(
                times, 100*superoperators[:, 0, 1], "-",
                color=cm.hawaii(1/3), label="0"
            )
            plt.plot(
                times, 100*superoperators[:, 0, 2], "-",
                color=cm.hawaii(2/3), label="-"
            )

            plt.xlabel("Time (arb.)")
            plt.ylabel("Population (%)")
            plt.legend()
            plt.draw()

            # superoperators -= np.eye(superoperator.shape[0])

            # visualise_time_evolution(time_evolution=superoperators)
            return superoperators

        run_scipy_exponentiate()


if __name__ == "__main__":
    # test_generators()
    # test_squaring()
    # test_combination()
    # test_time_sample()
    test_time_sample_quadrature()
    plt.show()
