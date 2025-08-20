using HDF5
using QuantumToolbox

function generate_coefficients_contrast()
	duration_excitation = 3e-6
	duration_relax_wait = 250e-9
	duration_mw = 1/(2*10e6)
	frequency_mw = 2.87e9 - 280e6
	amplitude_mw = sqrt(1/2)*2*10e6/28e9
	excitation_amplitude = 0.1

	time_thermal_start = 0
	time_thermal_end = time_thermal_start + duration_excitation
	time_zero_start = duration_excitation + duration_relax_wait
	time_zero_end = time_zero_start + duration_excitation
	time_one_start = 2*duration_excitation + 2*duration_relax_wait + duration_mw
	time_one_end = time_one_start + duration_excitation
	time_end = 10e-6
	tau = 2*pi

	function coefficient_x(p, t)
		if t < time_one_start - duration_mw
		    return 0.0
		end
		if t < time_one_start
			return amplitude_mw*sin(tau*frequency_mw*t)
		end
		return 0.0
	end

	coefficient_y(p, t) = 0.0
	coefficient_z(p, t) = 0.0

	function coefficient_l(p, t)
		if t < time_thermal_end
			return sqrt(excitation_amplitude)
		end
		if t < time_zero_start
			return 0.0
		end
		if t < time_zero_end
			return sqrt(excitation_amplitude)
		end
		if t < time_one_start
			return 0.0
		end
		return sqrt(excitation_amplitude)
        end

	return coefficient_x, coefficient_y, coefficient_z, coefficient_l
end

function generate_coefficients_odmr()
	time_start = 1e-6
	time_end = 1000e-6
	time_step = 150e-9
	frequency_start = 2.82e9
	frequency_width = 100e6

	tau = 2*pi

	function coefficient_x(p, t)
		if time > time_start
			phase = tau*(frequency_start*(time - time_start) + frequency_width*((time - time_start)^2) / (time_end - time_start)/2)
			return 100e-6*sin(phase)
		end
		return 0.0
	end

	coefficient_y(p, t) = 0.0
	coefficient_z(p, t) = 0.0

	function coefficient_l(p, t)
		return sqrt(0.01)
        end

	return coefficient_x, coefficient_y, coefficient_z, coefficient_l
end

# Read from python
h5_file = h5open("to_julia.h5", "r")
time_step = attrs(h5_file)["time_step"]
time_end = attrs(h5_file)["time_end"]
max_step = attrs(h5_file)["max_step"]
lindbladian = attrs(h5_file)["lindbladian"]

use_cuda = false

if use_cuda
	using CUDA
	CUDA.allowscalar(false)
end

density_operator_initial = Qobj(transpose(read(h5_file["density_operator_initial"])))
generator_x = Qobj(transpose(read(h5_file["generators_coherent/1"])))
generator_y = Qobj(transpose(read(h5_file["generators_coherent/2"])))
generator_z = Qobj(transpose(read(h5_file["generators_coherent/3"])))
generator_0 = Qobj(transpose(read(h5_file["generators_coherent/4"])))

index = 1
jumps_static = []
while haskey(h5_file, "generators_jump_static/"*string(index))
	if use_cuda
		push!(jumps_static, cu(Qobj(transpose(read(h5_file["generators_jump_static/"*string(index)])))))
	else
		push!(jumps_static, Qobj(transpose(read(h5_file["generators_jump_static/"*string(index)]))))
	end
	global index += 1
end

index = 1
jumps_dynamic = []
while haskey(h5_file, "generators_jump_dynamic/"*string(index))
	push!(jumps_dynamic, Qobj(transpose(read(h5_file["generators_jump_dynamic/"*string(index)]))))
	global index += 1
end

close(h5_file)

# Convert to QuantumToolboxJL language
time_list = 0:time_step:time_end
if use_cuda
	density_operator_initial = cu(Qobj(density_operator_initial))
	time_list = cu(time_list)
else
	density_operator_initial = Qobj(density_operator_initial)
end
if lindbladian == "contrast"
	coefficient_x, coefficient_y, coefficient_z, coefficient_l = generate_coefficients_contrast()
end
if lindbladian == "odmr"
	coefficient_x, coefficient_y, coefficient_z, coefficient_l = generate_coefficients_odmr()
end
if use_cuda
	hamiltonian = cu(QobjEvo((
		generator_0,
		(generator_x, coefficient_x),
		(generator_y, coefficient_y),
		(generator_z, coefficient_z)
	)))
else
	hamiltonian = QobjEvo((
		generator_0,
		(generator_x, coefficient_x),
		(generator_y, coefficient_y),
		(generator_z, coefficient_z)
	))
end

jumps_dynamic_true = []
for jump_dynamic in jumps_dynamic
	if use_cuda
		push!(jumps_dynamic_true, cu(QobjEvo(jump_dynamic, coefficient_l)))
	else
		push!(jumps_dynamic_true, QobjEvo(jump_dynamic, coefficient_l))
	end
end
jumps = vcat(jumps_static, jumps_dynamic_true)

wall_time_start = time()

result = mesolve(
	hamiltonian,
	density_operator_initial,
	time_list,
	jumps,
	reltol=max_step,
	abstol=max_step,
	dtmax=max_step,
	progress_bar=false,
	maxiters=100e9
)
density = result.states

time_list = collect(time_list)
density = stack(transpose.(get_data.(density)))

wall_time_end = time()
wall_duration = wall_time_end - wall_time_start

# Send to python
h5_file = h5open("from_julia.h5", "w")

h5_file["time"] = time_list
h5_file["density"] = density
attributes(h5_file)["wall_duration"] = wall_duration

close(h5_file)
