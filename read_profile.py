import numpy as np

import h5py

from matplotlib import pyplot as plt
from cmcrameri import cm
    
datatype = np.float64

if __name__ == "__main__":
    with h5py.File("profile/profile.h5", "r") as h5_file:
        name_ids = []
        start_times = []
        end_times = []
        kernel_trace_h5 = h5_file["CUPTI_ACTIVITY_KIND_KERNEL"]
        for entry_h5 in kernel_trace_h5:
            start_times.append(entry_h5[0])
            end_times.append(entry_h5[1])
            name_ids.append(entry_h5[9])

        names = []
        id_map_h5 = h5_file["StringIds"]
        for name_id in name_ids:
            name = id_map_h5[name_id][1].decode("utf-8")
            names.append(name)

    durations = []
    for start_time, end_time in zip(start_times, end_times):
        durations.append(end_time - start_time)

    for name_index, name in enumerate(names):
        names[name_index] = name.partition("kernel")[0].strip("_")

    trials = []
    trial  = None
    for name, duration in zip(names, durations):
        if name == names[0]:
            if trial is not None:
                trials.append(trial)
            trial = {"names": [], "durations": []}
        trial["names"].append(name)
        trial["durations"].append(duration)
    trials.append(trial)

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

        print(f"TRIAL {trial_index}")
        print(f"| {'Kernel':32s} | {'Times executed':16} | {'Total duration (ns)':24} |")
        print(f"|-{'-'*32}-|-{'-'*16}-|-{'-'*24}-|")
        for name, duration in durations_total.items():
            print(f"| {name:32s} | {times_executed[name]:16d} | {duration:24d} |")

        print(f"| {'TOTAL':32s} | {'':16s} | {duration_total:24d} |")
        print("")

    durations_total = []
    durations_slice = {}
    for trial in trials:
        durations_total.append(trial["duration_total"])
        for name, duration in trial["durations_total"].items():
            if name in durations_slice.keys():
                durations_slice[name].append(duration)
            else:
                durations_slice[name] = [duration]
    print(durations_total)

    durations_total = np.array(durations_total, dtype=datatype)
    for name, duration in durations_slice.items():
        durations_slice[name] = np.array(duration, dtype=datatype)

    durations_important = {}
    durations_important["Quadrature sample"] = durations_slice["calculate_time_quadrature"] \
        + durations_slice["sample"] \
        + durations_slice["combine_coefficients"] \
        + durations_slice["quadrature_combine"]
    durations_important["Lie algebra to matrix"] = durations_slice["calculate_differential"] \
        + durations_slice["scale_differential_basic"]
    durations_important["Cayley transform"] = durations_slice["calculate_cayley"]
    durations_important["Squaring"] = durations_slice["repeated_quartic_superoperator"]
    durations_important["Accumulate time evolution"] = durations_slice["id_superoperator"] \
        + durations_slice["basic_combine"]
    durations_important["Apply to state"] = durations_slice["calculate_time_basic"] \
        + durations_slice["apply_time_evolution"]

    trial = np.arange(durations_total.size)

    plt.figure()
    plt.plot(trial, np.log10(durations_total) - 9, ".-k", label="Total")
    number_of_plots = len(durations_important)
    for plot_index, (name, duration) in enumerate(durations_important.items()):
        plt.plot(trial, np.log10(duration) - 9, ".--", color=cm.hawaii(plot_index/number_of_plots), label=name)
    plt.xlabel("Trial index")
    plt.ylabel("lg Time (s)")
    plt.legend()
    plt.draw()
    plt.show()
    print(trial, duration_total)
