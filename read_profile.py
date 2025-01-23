import numpy as np

import h5py

from matplotlib import pyplot as plt
from cmcrameri import cm

datatype = np.float64


def read_h5():
    with h5py.File("profile/profile.h5", "r") as h5_file:
        kernel_name_ids = []
        kernel_start_times = []
        kernel_end_times = []
        kernel_trace_h5 = h5_file["CUPTI_ACTIVITY_KIND_KERNEL"]
        for entry_h5 in kernel_trace_h5:
            kernel_start_times.append(entry_h5[0])
            kernel_end_times.append(entry_h5[1])
            kernel_name_ids.append(entry_h5[9])

        kernel_names = []
        id_map_h5 = h5_file["StringIds"]
        for name_id in kernel_name_ids:
            name = id_map_h5[name_id][1].decode("utf-8")
            kernel_names.append(name)

        memory_name_ids = []
        memory_start_times = []
        memory_end_times = []
        memory_trace_h5 = h5_file["CUPTI_ACTIVITY_KIND_MEMCPY"]
        for entry_h5 in memory_trace_h5:
            memory_start_times.append(entry_h5[0])
            memory_end_times.append(entry_h5[1])
            memory_name_ids.append(entry_h5[9])

        memory_names = []
        id_map_h5 = h5_file["ENUM_CUDA_MEMCPY_OPER"]
        for name_id in memory_name_ids:
            name = id_map_h5[name_id][2].decode("utf-8")
            memory_names.append(name)

    names = kernel_names + memory_names
    start_times = kernel_start_times + memory_start_times
    end_times = kernel_end_times + memory_end_times

    # names = kernel_names
    # start_times = kernel_start_times
    # end_times = kernel_end_times

    start_times = np.array(start_times)
    end_times = np.array(end_times)
    order = np.argsort(start_times)
    start_times = start_times[order]
    end_times = end_times[order]
    names = [names[order_index] for order_index in order]
    print(order)

    return names, start_times, end_times


def calculate_durations(start_times, end_times):
    durations = []
    for start_time, end_time in zip(start_times, end_times):
        durations.append(end_time - start_time)
    return durations


def clean_names(names):
    for name_index, name in enumerate(names):
        names[name_index] = name.partition("kernel")[0].strip("_")


def split_into_trials(names, start_times, end_times):
    trials = []
    trial = {"names": [], "durations": []}
    for name, duration in zip(names, durations):
        trial["names"].append(name)
        trial["durations"].append(duration)
        if name == "Host-To-Device":
            trials.append(trial)
            trial = {"names": [], "durations": []}
    return trials


def accumulate(trials, verbose=False):
    for trial_index, trial in enumerate(trials):
        durations_total = {}
        times_executed = {}
        for name, duration in zip(trial["names"], trial["durations"]):
            if name in durations_total.keys():
                durations_total[name] += duration
                times_executed[name] += 1
            else:
                durations_total[name] = duration
                times_executed[name] = 1

        duration_total = 0
        for duration in durations_total.values():
            duration_total += duration

        trial["durations_total"] = durations_total
        trial["duration_total"] = duration_total
        trial["times_executed"] = times_executed

        if verbose:
            print(f"TRIAL {trial_index}")
            print(
                f"| {'Kernel':32s} | {'Times executed':16} | "
                f"{'Total duration (ns)':24} |"
            )
            print(f"|-{'-'*32}-|-{'-'*16}-|-{'-'*24}-|")
            for name, duration in durations_total.items():
                print(
                    f"| {name:32s} | {times_executed[name]:16d} | "
                    f"{duration:24d} |"
                )

            print(f"| {'TOTAL':32s} | {'':16s} | {duration_total:24d} |")
            print("")


def generate_slices(trials):
    durations_total = []
    durations_slice = {}
    for trial in trials:
        durations_total.append(trial["duration_total"])
        for name, duration in trial["durations_total"].items():
            if name in durations_slice.keys():
                durations_slice[name].append(duration)
            else:
                durations_slice[name] = [duration]

    durations_total = np.array(durations_total, dtype=datatype)
    for name, duration in durations_slice.items():
        durations_slice[name] = np.array(duration, dtype=datatype)

    return durations_total, durations_slice


def combine_related(durations_slice):
    durations_important = {}
    durations_important["Quadrature sample"] = \
        durations_slice["calculate_time_quadrature"] \
        + durations_slice["sample"] \
        + durations_slice["combine_coefficients"] \
        + durations_slice["quadrature_combine"]
    durations_important["Lie algebra to matrix"] = \
        durations_slice["calculate_differential"] \
        + durations_slice["scale_differential_basic"]
    durations_important["Cayley transform"] = \
        durations_slice["calculate_cayley"]
    durations_important["Squaring"] = \
        durations_slice["repeated_quartic_superoperator"]
    durations_important["Accumulate time evolution"] = \
        durations_slice["id_superoperator"] \
        + durations_slice["basic_combine"]
    durations_important["Apply to state"] = \
        durations_slice["calculate_time_basic"] \
        + durations_slice["apply_time_evolution"]

    return durations_important


def plot(durations_total, durations_slice):
    trial = np.arange(durations_total.shape[0])

    plt.figure()
    plt.plot(trial, durations_total*1e-9, ".-k", label="Total")
    number_of_plots = len(durations_slice)
    for plot_index, (name, duration) in enumerate(durations_slice.items()):
        plt.plot(
            trial,
            duration*1e-9,
            ".--",
            color=cm.hawaii(plot_index/number_of_plots),
            label=name
        )
    plt.xlabel("Trial index")
    plt.yscale("log")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.draw()


if __name__ == "__main__":
    names, start_times, end_times = read_h5()
    durations = calculate_durations(start_times, end_times)
    clean_names(names)
    trials = split_into_trials(names, start_times, end_times)
    accumulate(trials, verbose=True)
    durations_total, durations_slice = generate_slices(trials)
    durations_important = combine_related(durations_slice)
    plot(durations_total, durations_slice)
    plot(durations_total, durations_important)
    plt.show()
