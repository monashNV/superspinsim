using HDF5
using QuantumToolbox
using CUDA

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

# Read from python
h5_file = h5open("to_julia.h5", "r")
time_step = attrs(h5_file)["time_step"]
time_end = attrs(h5_file)["time_end"]
max_step = attrs(h5_file)["max_step"]

use_cuda = true
CUDA.allowscalar(false)

density_operator_initial = Qobj(read(h5_file["density_operator_initial"]))
generator_x = Qobj(read(h5_file["generators_coherent/1"]))
generator_y = Qobj(read(h5_file["generators_coherent/2"]))
generator_z = Qobj(read(h5_file["generators_coherent/3"]))
generator_0 = Qobj(read(h5_file["generators_coherent/4"]))

index = 1
jumps_static = []
while haskey(h5_file, "generators_jump_static/"*string(index))
	if use_cuda
		push!(jumps_static, Qobj(transpose(read(h5_file["generators_jump_static/"*string(index)]))))
	else
		push!(jumps_static, cu(Qobj(transpose(read(h5_file["generators_jump_static/"*string(index)])))))
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
time = 0:time_step:time_end
if use_cuda
	density_operator_initial = cu(Qobj(density_operator_initial))
	time = cu(time)
else
	density_operator_initial = Qobj(density_operator_initial)
end
coefficient_x, coefficient_y, coefficient_z, coefficient_l = generate_coefficients_contrast()
if use_cuda
	hamiltonian = QobjEvo((
		generator_0,
		(generator_x, coefficient_x),
		(generator_y, coefficient_y),
		(generator_z, coefficient_z)
	))
else
	hamiltonian = cu(QobjEvo((
		generator_0,
		(generator_x, coefficient_x),
		(generator_y, coefficient_y),
		(generator_z, coefficient_z)
	)))
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

result = mesolve(
	hamiltonian,
	density_operator_initial,
	time,
	jumps,
	reltol=max_step,
	dtmax=max_step,
	progress_bar=false,
	maxiters=100e6
)
density = result.states

# Send to python
h5_file = h5open("from_julia.h5", "w")

h5_file["time"] = collect(time)
h5_file["density"] = stack(get_data.(density))

close(h5_file)
