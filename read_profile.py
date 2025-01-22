import h5py

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

    durations_total = {}
    times_executed = {}
    for name, duration in zip(names, durations):
        if name in durations_total.keys():
            durations_total[name] += duration
            times_executed[name] += 1
        else:
            durations_total[name] = duration
            times_executed[name] = 1

    duration_total = 0
    for duration in durations_total.values():
        duration_total += duration

    print(f"| {'Kernel':32s} | {'Times executed':16} | {'Total duration (ns)':24} |")
    print(f"|-{'-'*32}-|-{'-'*16}-|-{'-'*24}-|")
    for name, duration in durations_total.items():
        print(f"| {name:32s} | {times_executed[name]:16d} | {duration:24d} |")

    print(f"| {'TOTAL':32s} | {'':16s} | {duration_total:24d} |")
