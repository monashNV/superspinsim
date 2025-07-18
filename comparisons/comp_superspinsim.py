import numpy as np

from superspinsim import generate_simulator
from comparisons.general import loop, make_strikes
from comparisons.lindbladians import contrast


def main(lindbladian="contrast", use_rotating=False, number_of_exponentials=2):
    init_arguments = {
        "use_rotating": use_rotating,
        "number_of_exponentials": number_of_exponentials,
        "lindbladian": lindbladian
    }
    loop(init_superspinsim, init_arguments)


def init_superspinsim(
        lindbladian="contrast", use_rotating=False, number_of_exponentials=2):
    if lindbladian == "contrast":
        generate_return, generate_return_comparison = contrast(use_rotating)
    if use_rotating:
        lindbladian, generators_list, vectorisation_map, vectors_real, \
            inv_vectors_real, doubles, singles, _ = generate_return
    else:
        lindbladian, generators_list, vectorisation_map, _ = generate_return
    time_step = generate_return_comparison[-2]
    time_end = generate_return_comparison[-1]
    density_operator_initial = generate_return_comparison[-3]

    run_arguments = {
        "time_step": time_step,
        "time_end": time_end,
        "density_operator_initial": density_operator_initial,
        "generate_return": generate_return,
        "use_rotating": use_rotating,
        "number_of_exponentials": number_of_exponentials,
        "run_raw": run_raw_superspinsim,
        "sweep_display_label": "Integration step size (s)",
        "sweep_display_code": "fine_steps",
        "sweep_display_units": "s",
        "sweep_parameter_code": "fine_division",
        "sweep_parameter_units": None
    }

    trial_name = "superspinsim"

    fine_steps = []
    run_arguments["fine_steps"] = fine_steps

    fine_division = 16
    fine_division_multiple = 1.3

    strikes_dict = make_strikes()
    strikes_dict["fine_division_multiple"] = fine_division_multiple
    run_arguments["strikes_dict"] = strikes_dict

    return trial_name, run_arguments, fine_division


def run_raw_superspinsim(**run_arguments):
    use_rotating = run_arguments["use_rotating"]
    generate_return = run_arguments["generate_return"]
    number_of_exponentials = run_arguments["number_of_exponentials"]
    fine_division = run_arguments["fine_division"]
    density_operator_initial = run_arguments["density_operator_initial"]
    time_end = run_arguments["time_end"]
    time_step = run_arguments["time_step"]

    if use_rotating:
        lindbladian, generators_list, vectorisation_map, vectors_real, \
            inv_vectors_real, doubles, singles, _ = generate_return
        simulator = generate_simulator(
            lindbladian, generators=np.array(generators_list),
            vectorisation_map=vectorisation_map,
            number_of_exponentials=number_of_exponentials,
            number_of_fine_divisions=fine_division,
            use_rotating=True,
            vectors_real=vectors_real, inv_vectors_real=inv_vectors_real,
            doubles=doubles, singles=singles
        )
    else:
        lindbladian, generators_list, vectorisation_map, _ = \
            generate_return
        simulator = generate_simulator(
            lindbladian, generators=np.array(generators_list),
            vectorisation_map=vectorisation_map,
            number_of_exponentials=number_of_exponentials,
            number_of_fine_divisions=fine_division,
        )

    time, density = simulator(
        density_operator_initial, 0, time_end, time_step)

    return time, density
