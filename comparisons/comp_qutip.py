import qutip as qt
import numpy as np
import math

from comparisons.lindbladians import contrast
from comparisons.general import loop


def main():
    loop(init_qutip, None)


def init_qutip():
    coefficients, generators_coherent, generators_jump, density_initial, \
        time_step, time_end = contrast()[1]

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

    density_initial = qt.Qobj(density_initial)

    time = np.arange(0, time_end, time_step)

    densities = []
    wall_durations = []
    max_steps = []

    max_step = 100e-9
    max_step_multiple = 1/3

    error_min = math.inf
    strikes = 0
    strikes_max = 3
    strike_aim = 3/4
    index = 0


def run_raw_qutip(run_arguments: dict):
    hamiltonian = run_arguments["hamiltonian"]
    jumps = run_arguments["jumps"]
    density_initial = run_arguments["density_initial"]
    time = run_arguments["time"]
    max_step = run_arguments["max_step"]

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
    density = np.array([state.data.to_array() for state in results.states])

    return time, density
