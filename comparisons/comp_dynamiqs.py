import dynamiqs as dq
import numpy as np
import jax.numpy as npj

from comparisons.lindbladians import contrast
from comparisons.general import loop, make_strikes


def main(lindbladian="contrast"):
    init_arguments = {
        "lindbladian": lindbladian
    }
    loop(init_dynamiqs, init_arguments)


def init_dynamiqs(lindbladian="contrast"):
    if lindbladian == "contrast":
        generate_return, generate_return_comparison = contrast(use_jax=True)

    coefficients, generators_coherent, generators_jump, \
        density_operator_initial, time_step, time_end = \
        generate_return_comparison

    dq.set_precision("double")

    hamiltonians = [
        dq.constant(generators_coherent[3]),
        dq.modulated(coefficients[0], generators_coherent[0]),
        dq.modulated(coefficients[1], generators_coherent[1]),
        dq.modulated(coefficients[2], generators_coherent[2]),
    ]
    # hamiltonian = sum(hamiltonians)
    hamiltonian = \
        hamiltonians[0] + hamiltonians[1] + hamiltonians[2] + hamiltonians[3]
    density_operator_initial = density_operator_initial

    jumps = []
    jump_coefficient = lambda t: npj.sqrt(coefficients[3](t))
    for jump in generators_jump[1]:
        jumps.append(jump)
    for jump in generators_jump[0]:
        jumps.append(dq.modulated(jump_coefficient, jump))

    time = np.arange(0, time_end, time_step)

    run_arguments = {
        "time_step": time_step,
        "time_end": time_end,
        "density_operator_initial": density_operator_initial,
        "generate_return": generate_return,
        "hamiltonian": hamiltonian,
        "jumps": jumps,
        "time": time,
        "run_raw": run_raw_dynamiqs,
        "sweep_display_label": "Max integration step size (s)",
        "sweep_display_code": "max_steps",
        "sweep_display_units": "s",
        "sweep_parameter_code": "max_step",
        "sweep_parameter_units": "s"
    }

    trial_name = "dynamiqs"

    max_steps = []
    run_arguments["max_steps"] = max_steps

    # if use_jax:
    #     max_step = 1
    # else:
    max_step = time_step
    max_step_multiple = 1/3

    strikes_dict = make_strikes()
    strikes_dict["max_step_multiple"] = max_step_multiple
    run_arguments["strikes_dict"] = strikes_dict

    return trial_name, run_arguments, max_step


def run_raw_dynamiqs(**run_arguments):
    hamiltonian = run_arguments["hamiltonian"]
    jumps = run_arguments["jumps"]
    density_operator_initial = run_arguments["density_operator_initial"]
    time = run_arguments["time"]
    max_step = run_arguments["max_step"]

    results = dq.mesolve(
        hamiltonian, jumps, density_operator_initial, time,
        method=dq.method.Tsit5(
            max_steps=int(100e6), rtol=max_step, atol=max_step),
        options=dq.Options(progress_meter=False)
    )
    densities = results.states.to_numpy()

    return time, densities
