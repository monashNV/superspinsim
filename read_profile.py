import numpy as np

import h5py

from matplotlib import pyplot as plt
from cmcrameri import cm

from scipy import optimize as spo

from pogger import Pogger, Read

datatype = np.float64

START_TIME_HEADING = "start"
END_TIME_HEADING = "end"
KERNEL_NAME_ID_HEADING = "demangledName"
MEMORY_NAME_ID_HEADING = "copyKind"

pogger = Pogger("superspinsim-benchmarks")


def read_h5():
    with h5py.File("profile/profile.h5", "r") as h5_file:
        kernel_name_ids = []
        kernel_start_times = []
        kernel_end_times = []
        kernel_trace_h5 = h5_file["CUPTI_ACTIVITY_KIND_KERNEL"]
        for entry_h5 in kernel_trace_h5:
            kernel_start_times.append(entry_h5[START_TIME_HEADING])
            kernel_end_times.append(entry_h5[END_TIME_HEADING])
            kernel_name_ids.append(entry_h5[KERNEL_NAME_ID_HEADING])

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
            memory_start_times.append(entry_h5[START_TIME_HEADING])
            memory_end_times.append(entry_h5[END_TIME_HEADING])
            memory_name_ids.append(entry_h5[MEMORY_NAME_ID_HEADING])

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
        name = name.partition("kernel")[0].strip("_")
        names[name_index] = name.split("::")[-1].strip("_")


def split_into_trials(names, start_times, end_times):
    trials = []
    trial = {"names": [], "durations": []}
    to_device_count = 0
    first = True
    for name, duration in zip(names, durations):
        trial["names"].append(name)
        trial["durations"].append(duration)
        if name == "Host-to-Device":
            to_device_count += 1
            if to_device_count == 4:
                trial_new = {"names": trial["names"][-4:], "durations": trial["durations"][-4:]}
                trial["names"] = trial["names"][:-4]
                trial["durations"] = trial["durations"][:-4]
                if not first:
                    trials.append(trial)
                first = False
                trial = trial_new
                to_device_count = 0
    trials.append(trial)
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
    if "calculate_cayley" in durations_slice:
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
    durations_important["To VRAM"] = durations_slice["Host-to-Device"]
    durations_important["From VRAM"] = durations_slice["Device-to-Host"]

    return durations_important


if __name__ == "__main__":
    @pogger.record()
    def plot_durations(durations_total, durations_slice):
        trial = np.arange(durations_total.shape[0])

        plt.figure(label="durations")
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

    @pogger.record(("error", "durations", "power"), (None, "ns", None))
    def fit_errors(error, durations_total):
        def hyperbolic_function(x, x0, y0, m, c):
            return np.sqrt((m*(x - x0))**2 + y0**2) + c

        error_log = np.log10(error)
        durations_log = np.log10(durations_total)
        hyperbolic_fit, hyperbolic_fit_cov = spo.curve_fit(
            hyperbolic_function,
            error_log,
            durations_log,
            [0, -2, 1, -4]
        )

        power = 1/hyperbolic_fit[2]
        print(f"Power scaling: {power}")

        error_span = np.geomspace(error[0], error[-1])
        durations_fit = 10**hyperbolic_function(
                np.log10(error_span),
                hyperbolic_fit[0],
                hyperbolic_fit[1],
                hyperbolic_fit[2],
                hyperbolic_fit[3]
            )

        plt.figure(label="fit")
        plt.loglog(durations_total*1e-9, error, "k.", label="Measured")
        plt.loglog(
            durations_fit*1e-9, error_span,
            "k--", label=f"Fit (Power = {power:.2f})"
        )
        plt.xlabel("GPU time (s)")
        plt.ylabel("RMS error")
        plt.legend()
        plt.draw()

        return error, durations_total, power

    names, start_times, end_times = read_h5()
    durations = calculate_durations(start_times, end_times)
    clean_names(names)
    print(names[:10])
    trials = split_into_trials(names, start_times, end_times)
    accumulate(trials, verbose=True)
    durations_total, durations_slice = generate_slices(trials)
    durations_important = combine_related(durations_slice)

    pogger.set_context("durations_all")
    for name, duration in durations_slice.items():
        pogger.write_array(name, duration, "ns")

    pogger.set_context("durations_important")
    for name, duration in durations_important.items():
        pogger.write_array(name, duration, "ns")

    pogger.set_context("durations_all")
    plot_durations(durations_total, durations_slice)
    pogger.set_context("durations_important")
    plot_durations(durations_total, durations_important)

    durations_total = durations_total[:-1]

    with open("profile/datetime", "r") as file_previous_log:
        previous_log = file_previous_log.readline().strip()
    read = Read("superspinsim-benchmarks", previous_log)
    error = read.read_array("errors", "error_analysis")
    # error = read.read_array("errors", "quartics")

    pogger.set_context()
    pogger.write_value("previous_log", previous_log)

    pogger.set_context("errors")
    fit_errors(error, durations_total)

    plt.show()
