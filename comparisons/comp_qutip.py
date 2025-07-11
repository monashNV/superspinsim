import qutip as qt
import numpy as np
import math
import time as tm
from matplotlib import pyplot as plt
from cmcrameri import cm

from comparisons.lindbladians import contrast
from comparisons.analysis import calculate_errors_diff


def main():
    # spin_x = 1/math.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    # spin_x = qt.Qobj(spin_x)
    # spin_z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    # spin_z = qt.Qobj(spin_z)
    # hamiltonian = [
    #     [spin_x, lambda t: 1 + np.sin(20*math.tau*t)],
    #     [spin_z, lambda t: 1]
    # ]

    # results = qt.mesolve(
    #     H=hamiltonian,
    #     rho0=density_initial,
    #     tlist=time
    # )

    # plt.figure(label="figure_test")
    # plt.imshow(
    #     colour_complex_matrix(density_final/np.max(np.abs(density_final))))
    # plt.draw()

    coefficients, generators_coherent, generators_jump, time_step, time_end = \
        contrast()[1]

    hamiltonian = [
        qt.Qobj(generators_coherent[3]),
        [qt.Qobj(generators_coherent[0]), coefficients[0]],
        [qt.Qobj(generators_coherent[1]), coefficients[1]],
        [qt.Qobj(generators_coherent[2]), coefficients[2]]
    ]

    jumps = []
    jump_coefficient = lambda t: math.sqrt(coefficients[3](t))
    for jump in generators_jump[1]:
        jumps.append(qt.Qobj(jump))
    for jump in generators_jump[0]:
        jumps.append([qt.Qobj(jump), jump_coefficient])

    density_initial = np.zeros((7, 7))
    density_initial[0, 0] = 1/3
    density_initial[1, 1] = 1/3
    density_initial[2, 2] = 1/3
    density_initial = qt.Qobj(density_initial)

    time = np.arange(0, time_end, time_step)

    # densities = []
    # # tolerances = np.geomspace(1e-5, 1e-16, 6)
    # # max_steps = np.geomspace(1e-15, 10e-9, 21)
    # wall_durations = []
    # # for index, max_step in enumerate(max_steps):

    densities = []
    wall_durations = []
    max_steps = []

    max_step = 10e-9
    max_step_multiple = 1/1.3

    error_min = math.inf
    strikes = 0
    strikes_max = 3
    strike_aim = 3/4
    index = 0
    while True:
        wall_time_start = tm.perf_counter()
        results = qt.mesolve(
            H=hamiltonian,
            c_ops=jumps,
            rho0=density_initial,
            tlist=time,
            options=qt.Options(
                first_step=max_step, max_step=max_step, nsteps=100e6,
                atol=max_step, rtol=max_step
            )
        )
        densities.append(
            np.array([state.data.to_array() for state in results.states]))
        wall_duration = tm.perf_counter() - wall_time_start
        wall_durations.append(wall_duration)
        max_steps.append(max_step)

        if index > 1:
            errors = calculate_errors_diff(
                np.asarray(densities)[:, :, 0]
                + 1j*np.asarray(densities)[:, :, 1]
            )
            error_min_current = np.min(errors)
            if error_min_current < strike_aim*error_min:
                error_min = error_min_current
                strikes = 0
            else:
                strikes += 1
                if strikes >= strikes_max:
                    break
            print(index, wall_duration, error_min_current, error_min, strikes)
        else:
            print(index, wall_duration)

        index += 1
        max_step *= max_step_multiple

    max_steps = np.array(max_steps)
    densities = np.array(densities)
    wall_durations = np.array(wall_durations)

    plt.figure(label="errors_diff")
    plt.loglog(wall_durations, errors, "k.-")
    plt.draw()

    return max_steps, densities, wall_durations, errors
    # trials = []
    # for max_step, density, wall_duration in zip(
    #         max_steps, densities, wall_durations):
    #     trial = {
    #         "max_step": max_step,
    #         "density": density,
    #         "wall_duration": wall_duration
    #     }
    #     trials.append(trial)

    # errors = []
    # for density_compare in densities:
    #     errors_compare = []
    #     for density in densities:
    #         errors_compare.append(
    #             math.sqrt(np.average(np.abs((density - density_compare)**2)))
    #         )
    #     errors.append(errors_compare)

    # errors = np.array(errors)
    # wall_durations = np.array(wall_durations)
    # densities = np.array(densities)

    # plt.figure(label="max_steps")
    # for index, error in enumerate(errors):
    #     if index == 0:
    #         alpha = 1
    #     else:
    #         alpha = 0.2
    #     indices = np.arange(len(error))
    #     indices = indices[indices != index]
    #     plt.loglog(
    #         max_steps[indices]/1e-9, error[indices], ".-",
    #         color=cm.hawaii(index/len(errors)), label=f"Sample {index}",
    #         alpha=alpha
    #     )
    # # plt.legend()
    # plt.xlabel("Min step (ns)")
    # plt.ylabel("Error")
    # plt.draw()

    # plt.figure(label="wall_duration")
    # for index, error in enumerate(errors):
    #     if index == 0:
    #         alpha = 1
    #     else:
    #         alpha = 0.2
    #     indices = np.arange(len(error))
    #     indices = indices[indices != index]
    #     plt.loglog(
    #         wall_durations[indices]/1e-3, error[indices], ".-",
    #         color=cm.hawaii(index/len(errors)), label=f"Sample {index}",
    #         alpha=alpha
    #     )
    # # plt.legend()
    # plt.xlabel("Wall duration (ms)")
    # plt.ylabel("Error")
    # plt.draw()

    # return max_steps, densities, wall_durations, errors

    # density = np.array([state.data.to_array() for state in results.states])
    # fluorescence = density[:, 3, 3] + density[:, 4, 4] + density[:, 5, 5]

    # plt.figure(label="fluorescence")
    # plt.plot(time, fluorescence, "k-")
    # plt.draw()
