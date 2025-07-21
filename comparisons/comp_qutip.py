import qutip as qt
import qutip_jax as qtj
import jax.numpy as npj
import numpy as np
import math
import warnings

from comparisons.lindbladians import contrast
from comparisons.general import loop, make_strikes


def main(lindbladian="contrast", use_jax=True):
    init_arguments = {
        "lindbladian": lindbladian,
        "use_jax": use_jax
    }
    loop(init_qutip, init_arguments)


def init_qutip(lindbladian="contrast", use_jax=False):
    # Suppress qutip's (wrong) FutureWarning.
    warnings.filterwarnings("ignore", category=FutureWarning)

    if lindbladian == "contrast":
        generate_return, generate_return_comparison = contrast()

    coefficients, generators_coherent, generators_jump, density_initial, \
        time_step, time_end = generate_return_comparison

    if use_jax:
        qt.settings.core["default_dtype"] = "jax"

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

    if use_jax:
        for index in range(len(hamiltonian)):
            if type(hamiltonian[index]) is list:
                hamiltonian[index] = hamiltonian[index][0].to("jax")
                # hamiltonian[index][0] = hamiltonian[index][0].to("jax")
            else:
                hamiltonian[index] = hamiltonian[index].to("jax")
        for index in range(len(jumps)):
            if type(jumps[index]) is list:
                jumps[index] = jumps[index][0].to("jax")
                # jumps[index][0] = jumps[index][0].to("jax")
            else:
                jumps[index] = jumps[index].to("jax")

    density_initial = qt.Qobj(density_initial)

    if use_jax:
        time = npj.arange(0, time_end, time_step)
    else:
        time = np.arange(0, time_end, time_step)

    run_arguments = {
        "time_step": time_step,
        "time_end": time_end,
        "density_operator_initial": density_initial,
        "generate_return": generate_return,
        "hamiltonian": hamiltonian,
        "jumps": jumps,
        "time": time,
        "run_raw": run_raw_qutip,
        "sweep_display_label": "Max integration step size (s)",
        "sweep_display_code": "max_steps",
        "sweep_display_units": "s",
        "sweep_parameter_code": "max_step",
        "sweep_parameter_units": "s",
        "use_jax": use_jax
    }

    trial_name = "qutip"
    if use_jax:
        trial_name += "_jax"

    max_steps = []
    run_arguments["max_steps"] = max_steps

    # if use_jax:
    #     max_step = 1
    # else:
    max_step = 100e-9
    max_step_multiple = 1/3

    strikes_dict = make_strikes()
    strikes_dict["max_step_multiple"] = max_step_multiple
    run_arguments["strikes_dict"] = strikes_dict

    return trial_name, run_arguments, max_step


def run_raw_qutip(**run_arguments: dict):
    hamiltonian = run_arguments["hamiltonian"]
    jumps = run_arguments["jumps"]
    density_initial = run_arguments["density_operator_initial"]
    time = run_arguments["time"]
    max_step = run_arguments["max_step"]
    use_jax = run_arguments["use_jax"]

    if use_jax:
        results = qt.mesolve(
            H=hamiltonian,
            c_ops=jumps,
            rho0=density_initial,
            tlist=time,
            options=qt.Options(
                # dt0=max_step,
                method="diffrax",
                max_steps=int(100e6)
            )
        )
    else:
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
