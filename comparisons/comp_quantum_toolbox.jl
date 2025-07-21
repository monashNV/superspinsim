using HDF5
using QuantumToolbox

# Read from python
h5_file = h5open("to_julia.h5", "r")
time_step = attrs(h5_file)["time_step"]
time_end = attrs(h5_file)["time_end"]
close(h5_file)

print(time_step)
print(time_end)

times = 0:time_step:time_end
print(times)

# Send to python
h5_file = h5open("from_julia.h5", "w")
h5_file["times"] = collect(times)
close(h5_file)
