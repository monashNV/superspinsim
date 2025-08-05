import math
import numpy as np
from superspinsim.generate_generators import generate_7


def _general(
        coefficients: list[callable], quiescent_magnetic_field: np.ndarray,
        time_step: float, time_end: float, use_rotating: bool):
    generate_return = generate_7(
        coefficients, quiescent_magnetic_field, return_full=True,
        use_rotating=use_rotating
    )
    generators_full = generate_return[-1]

    # print(generators_full["jump"].keys())
    generators_jump = [[], []]
    for key, generator in generators_full["jump"].items():
        if "Lr" in key:
            generators_jump[0].append(
                generator[:, :, 0] + 1j*generator[:, :, 1])
        else:
            generators_jump[1].append(
                generator[:, :, 0] + 1j*generator[:, :, 1])

    generators_coherent = [
        generators_full["coherent"]["Gx"][:, :, 0]
        + 1j*generators_full["coherent"]["Gx"][:, :, 1],
        generators_full["coherent"]["Gy"][:, :, 0]
        + 1j*generators_full["coherent"]["Gy"][:, :, 1],
        generators_full["coherent"]["Gz"][:, :, 0]
        + 1j*generators_full["coherent"]["Gz"][:, :, 1],
        generators_full["coherent"]["H0"][:, :, 0]
        + 1j*generators_full["coherent"]["H0"][:, :, 1],
    ]

    density_operator_initial = np.zeros((7, 7), dtype=np.complex128)
    density_operator_initial[0, 0] = 1/3
    density_operator_initial[1, 1] = 1/3
    density_operator_initial[2, 2] = 1/3

    generate_return_comparison = (
        coefficients, generators_coherent, generators_jump,
        density_operator_initial, time_step, time_end
    )

    return generate_return, generate_return_comparison


def contrast(use_rotating=False, use_jax=False):
    duration_excitation = 3e-6
    duration_relax_wait = 250e-9
    duration_mw = 1/(2*10e6)
    frequency_mw = 2.87e9 - 280e6
    amplitude_mw = math.sqrt(1/2)*2*10e6/28e9
    excitation_amplitude = 0.1

    time_thermal_start = 0
    time_thermal_end = time_thermal_start + duration_excitation
    time_zero_start = duration_excitation + duration_relax_wait
    time_zero_end = time_zero_start + duration_excitation
    time_one_start = 2*duration_excitation + 2*duration_relax_wait \
        + duration_mw
    time_one_end = time_one_start + duration_excitation
    time_end = 10e-6

    time_step = 10e-9

    quiescent_magnetic_field = np.array([0, 0, 1])*10e-3

    if use_jax:
        from jax import numpy as npj

        def coefficient_x(time):
            return_value = \
                (time >= time_one_start - duration_mw) \
                * (time < time_one_start) \
                * amplitude_mw*npj.sin(math.tau*frequency_mw*time)

            return return_value

        def coefficient_y(time):
            return 0

        def coefficient_z(time):
            return 0

        def coefficient_r(time):
            return_value = \
                (time < time_thermal_end)*excitation_amplitude \
                + (time >= time_zero_start)*(time < time_zero_end) \
                * excitation_amplitude \
                + (time >= time_one_start)*excitation_amplitude
            return return_value
    else:
        def coefficient_x(time):
            if time < time_one_start - duration_mw:
                return 0
            if time < time_one_start:
                return amplitude_mw*math.sin(math.tau*frequency_mw*time)
            return 0

        def coefficient_y(time):
            return 0

        def coefficient_z(time):
            return 0

        def coefficient_r(time):
            if time < time_thermal_end:
                return excitation_amplitude
            if time < time_zero_start:
                return 0
            if time < time_zero_end:
                return excitation_amplitude
            if time < time_one_start:
                return 0
            return excitation_amplitude

    coefficients = [coefficient_x, coefficient_y, coefficient_z, coefficient_r]

    return _general(
        coefficients, quiescent_magnetic_field, time_step, time_end,
        use_rotating
    )


def odmr(use_rotating=False, use_jax=False):
    time_start = 1e-6
    time_end = 1000e-6
    time_step = 150e-9
    frequency_start = 2.82e9
    frequency_width = 100e6

    quiescent_magnetic_field = np.array([0, 0, 1])*1e-3

    def coefficient_x(time):
        if time > time_start:
            phase = math.tau*(
                frequency_start*(time - time_start)
                + frequency_width*((time - time_start)**2)
                / (time_end - time_start)/2
            )
            return 100e-6*math.sin(phase)
        else:
            return 0

    def coefficient_y(time):
        return 0

    def coefficient_z(time):
        return 0

    def coefficient_r(time):
        return 0.01

    coefficients = [coefficient_x, coefficient_y, coefficient_z, coefficient_r]

    return _general(
        coefficients, quiescent_magnetic_field, time_step, time_end,
        use_rotating
    )
