using HDF5
using QuantumToolbox

println("Hello world, this is me")

h5_file = h5open("julia_save.h5", "w")

attributes(h5_file)["hello world"] = "This is me"

close(h5_file)
