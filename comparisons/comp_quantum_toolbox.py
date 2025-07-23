import os
import h5py
import numpy as np

from comparisons.lindbladians import contrast
from comparisons.general import loop, make_strikes


def main(lindbladian="contrast"):
    init_arguments = {
        "lindbladian": lindbladian
    }
    loop(init_quantum_toolbox, init_arguments)


def init_quantum_toolbox(lindbladian="contrast"):
    if lindbladian == "contrast":
        generate_return, generate_return_comparison = contrast()

    coefficients, generators_coherent, generators_jump, density_initial, \
        time_step, time_end = generate_return_comparison

    run_arguments = {
        "time_step": time_step,
        "time_end": time_end,
        "density_operator_initial": density_initial,
        "generate_return": generate_return,
        "run_raw": run_raw_quantum_toolbox,
        "generators_coherent": generators_coherent,
        "generators_jump": generators_jump,
        "sweep_display_label": "Max integration step size (s)",
        "sweep_display_code": "max_steps",
        "sweep_display_units": "s",
        "sweep_parameter_code": "max_step",
        "sweep_parameter_units": "s",
        "do_overwrite_wall_duration": True
    }

    trial_name = "quantum_toolbox_jl"

    max_steps = []
    run_arguments["max_steps"] = max_steps

    max_step = 10e-9
    max_step_multiple = 1/3

    strikes_dict = make_strikes()
    strikes_dict["max_step_multiple"] = max_step_multiple
    run_arguments["strikes_dict"] = strikes_dict

    return trial_name, run_arguments, max_step


def run_raw_quantum_toolbox(**run_arguments):
    time_step = run_arguments["time_step"]
    time_end = run_arguments["time_end"]
    density_initial = run_arguments["density_operator_initial"]
    generators_coherent = run_arguments["generators_coherent"]
    generators_jump = run_arguments["generators_jump"]
    max_step = run_arguments["max_step"]

    with h5py.File("to_julia.h5", "w") as h5_file:
        h5_file.attrs["time_step"] = time_step
        h5_file.attrs["time_end"] = time_end
        h5_file.attrs["max_step"] = max_step
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

    os.system("julia --project=julia_env comparisons/comp_quantum_toolbox.jl")

    with h5py.File("from_julia.h5", "r") as h5_file:
        time = np.asarray(h5_file["time"])
        density = np.asarray(h5_file["density"])
        wall_duration = h5_file.attrs["wall_duration"]

    os.remove("to_julia.h5")
    os.remove("from_julia.h5")

    return time, density, wall_duration
