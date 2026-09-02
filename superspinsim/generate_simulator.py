from superspinsim.quadratures import samples as samples_dict
from superspinsim.quadratures import weights as weights_dict

import math
import numpy as np

import numba as nb
import numba.cuda as nc

from numba.core.runtime import rtsys

import warnings


def generate_simulator(
        sampler: callable,
        generators: np.ndarray,
        vectorisation_map: np.ndarray = None,
        use_cuda: bool = True,
        datatype=np.float64,
        use_residual: bool = True,
        number_of_quartic_repeats: int = 35,  # 35,
        number_of_exponentials: int = 2,
        number_of_fine_divisions: int = 1,

        use_rotating: bool = False,
        vectors_real: np.ndarray = None,
        inv_vectors_real: np.ndarray = None,
        doubles: np.ndarray = None,
        singles: np.ndarray = None,

        use_kernel: bool = False,
        full_projection: np.ndarray = None,
        image_projection: np.ndarray = None,

        use_unitary: bool = False,
        elimination: np.ndarray = None,
        duplication: np.ndarray = None,

        verbose: bool = False,

        use_cayley: bool = False
        ):

    if use_cayley:
        raise "Cayley not implemented"
    if use_unitary:
        print("Unitary mode not functional; currently being debugged")

    scaling_for_quartics: datatype = \
        4.0**number_of_quartic_repeats

    generators = generators.astype(datatype)
    operator_size = generators.shape[1]
    operator_size_density = vectorisation_map.shape[0]

    if use_cuda:
        def _get_dimensions_for_gpu(size, stride):
            submatrix_size = min(size, stride)
            number_of_submatrices = int(math.ceil(size/submatrix_size))
            return submatrix_size, number_of_submatrices

        stride = 32
        submatrix_size, number_of_submatrices = _get_dimensions_for_gpu(
            operator_size, stride)
        submatrix_size_density, number_of_submatrices_density = \
            _get_dimensions_for_gpu(operator_size_density, stride)
    else:
        cpu_options = {
            "nopython": True,
            "error_model": "numpy",
            "fastmath": True,
            "parallel": True,
            "boundscheck": True
        }

    if number_of_exponentials == 1:
        if verbose:
            print("Using the CF2:1 method with 1 node.")
        sample_quadrature = np.array(samples_dict["1_gl"], dtype=datatype)
        weights = np.array(weights_dict["2_1_gl"], dtype=datatype)
    elif number_of_exponentials == 2:
        if verbose:
            print("Using the CF4:2 method with 2 nodes.")
        sample_quadrature = np.array(samples_dict["2_gl"], dtype=datatype)
        weights = np.array(weights_dict["4_2_gl"], dtype=datatype)
    elif number_of_exponentials == 3:
        if verbose:
            print("Using the CF4:3 method with 2 nodes.")
        sample_quadrature = np.array(samples_dict["2_gl"], dtype=datatype)
        weights = np.array(weights_dict["4_3_gl"], dtype=datatype)
    elif number_of_exponentials == 5:
        if verbose:
            print("Using the CF6:5 method with 3 nodes.")
        sample_quadrature = np.array(samples_dict["3_gl"], dtype=datatype)
        weights = np.array(weights_dict["6_5_gl"], dtype=datatype)
    elif number_of_exponentials == 6:
        if verbose:
            print("Using the CF6:6 method with 3 nodes.")
        sample_quadrature = np.array(samples_dict["3_gl"], dtype=datatype)
        weights = np.array(weights_dict["6_6_gl"], dtype=datatype)

    if use_rotating:
        # Make a copy of the eigensystem
        vectors_real = vectors_real.copy()
        inv_vectors_real = inv_vectors_real.copy()
        doubles = doubles.copy()
        singles = singles.copy()
        if doubles.shape[0] == 0:
            doubles = doubles.reshape((0, 2))
        doubles_size = doubles.shape[0]
        singles_size = singles.size

    if verbose:
        print("Declaring methods")

    # Time grid ---------------------------------------------------------------

    def _calculate_time(time, time_index, time_start, time_step):
        time[time_index] = time_start + time_step*(time_index + 1)

    def _calculate_time_quadrature(
            time, time_index, time_start, time_step, sample):
        time_step_start = time_start + time_step*time_index
        for sample_index in range(sample.size):
            time[time_index*sample.size + sample_index] = \
                time_step_start + time_step*sample[sample_index] \
                / number_of_fine_divisions

    if use_cuda:
        _calculate_time = nc.jit(_calculate_time, device=True)
        _calculate_time_quadrature = nc.jit(
            _calculate_time_quadrature, device=True)

        def _calculate_time_basic_kernel(time, time_start, time_step):
            time_index = nc.blockDim.x*nc.blockIdx.x + nc.threadIdx.x
            if time_index < time.size:
                _calculate_time(time, time_index, time_start, time_step)

        _calculate_time_basic_kernel = nc.jit(_calculate_time_basic_kernel)

        def _calculate_time_quadrature_kernel(
                time, time_sample, time_start, time_step, sample):
            time_index = nc.blockDim.x*nc.blockIdx.x + nc.threadIdx.x
            if time_index < time.size:
                _calculate_time_quadrature(
                    time_sample, time_index, time_start, time_step, sample)

        _calculate_time_quadrature_kernel = nc.jit(
            _calculate_time_quadrature_kernel)

    else:
        _calculate_time = nb.jit(_calculate_time)
        _calculate_time_quadrature = nb.jit(_calculate_time_quadrature)

        def _calculate_time_basic_loop(time, time_start, time_step):
            for time_index in nb.prange(time.size):
                _calculate_time(time, time_index, time_start, time_step)

        _calculate_time_basic_loop = nb.jit(
            _calculate_time_basic_loop, **cpu_options
        )

        def _calculate_time_quadrature_loop(
                time, time_sample, time_start, time_step, sample):
            for time_index in nb.prange(time.size):
                _calculate_time_quadrature(
                    time_sample, time_index, time_start, time_step, sample)

        _calculate_time_quadrature_loop = nb.jit(
            _calculate_time_quadrature_loop, **cpu_options
        )

    def _calculate_time_basic_run(time, time_start, time_step):
        if use_cuda:
            grid_size = (int(math.ceil(time.size/32)), 1)
            block_size = (32, 1)
            _calculate_time_basic_kernel[grid_size, block_size] \
                (time, time_start, time_step)
        else:
            _calculate_time_basic_loop(time, time_start, time_step)

    def _calculate_time_quadrature_run(
            time, time_sample, time_start, time_step, sample):
        if use_cuda:
            grid_size = (int(math.ceil(time.size/32)), 1)
            block_size = (32, 1)
            _calculate_time_quadrature_kernel[grid_size, block_size] \
                (time, time_sample, time_start, time_step, sample)
        else:
            _calculate_time_quadrature_loop(
                time, time_sample, time_start, time_step, sample)

    # Sampling ----------------------------------------------------------------

    def _generate_sampler(sampler):
        if use_cuda:
            sampler_device = nc.jit(sampler, device=True)

            def sample_kernel(times, coefficients):
                time_index = nc.blockDim.x*nc.blockIdx.x + nc.threadIdx.x
                if time_index < times.size:
                    for generator_index in range(coefficients.shape[1]):
                        coefficients[time_index, generator_index] = 0.0

                    sampler_device(
                        times[time_index], coefficients[time_index, :])

            sample_kernel = nc.jit(sample_kernel)

        else:
            sampler_device = nb.jit(sampler)

            def sample_loop(times, coefficients):
                for time_index in nb.prange(times.size):
                    for generator_index in range(coefficients.shape[1]):
                        coefficients[time_index, generator_index] = 0.0

                    sampler_device(
                        times[time_index], coefficients[time_index, :])

            sample_loop = nb.jit(sample_loop, **cpu_options)

        def sample_run(times, coefficients):
            if use_cuda:
                grid_size = (int(math.ceil(times.size/32)), 1)
                block_size = (32, 1)
                sample_kernel[grid_size, block_size](times, coefficients)
            else:
                sample_loop(times, coefficients)

        return sample_run

    # Make sampler GPU/CPU compatible
    sample_run = _generate_sampler(sampler)

    # Quadrature --------------------------------------------------------------

    def _combine_coefficients(
            coefficient, weighted_coefficient, weight,
            exponential_index, coefficient_index):
        scratch = 0
        for trace_index in range(weight.shape[1]):
            if use_cuda:
                scratch = nc.fma(
                    weight[exponential_index, trace_index],
                    coefficient[trace_index, coefficient_index],
                    scratch
                )
            else:
                scratch += \
                    weight[exponential_index, trace_index] \
                    * coefficient[trace_index, coefficient_index]
        weighted_coefficient[exponential_index, coefficient_index] = scratch

    if use_cuda:
        _combine_coefficients = nc.jit(_combine_coefficients, device=True)

        def _combine_coefficients_kernel(
                coefficients, weighted_coefficients, weights):
            coef_index = nc.threadIdx.x
            weight_index = nc.threadIdx.y
            t_index = nc.blockIdx.x
            if coef_index < weighted_coefficients.shape[1] \
                    and weight_index < weights.shape[0]:
                _combine_coefficients(
                    coefficients[t_index*weights.shape[1]:
                                 (t_index + 1)*weights.shape[1], :],
                    weighted_coefficients[
                        t_index*weights.shape[0]:
                        (t_index + 1)*weights.shape[0], :],
                    weights,
                    weight_index,
                    coef_index
                )

        _combine_coefficients_kernel = nc.jit(_combine_coefficients_kernel)

    else:
        _combine_coefficients = nb.jit(_combine_coefficients)

        def _combine_coefficients_loop(
                coefficients, weighted_coefficients, weights):
            for t_index in nb.prange(weighted_coefficients.shape[0]//weights.shape[0]):
                for coef_index in nb.prange(weighted_coefficients.shape[1]):
                    for weight_index in nb.prange(weights.shape[0]):
                        _combine_coefficients(
                            coefficients[t_index*weights.shape[1]:
                                         (t_index + 1)*weights.shape[1], :],
                            weighted_coefficients[
                                t_index*weights.shape[0]:
                                (t_index + 1)*weights.shape[0], :],
                            weights,
                            weight_index,
                            coef_index
                        )

        _combine_coefficients_loop = nb.jit(
            _combine_coefficients_loop, **cpu_options)

    def _combine_coefficients_run(
            coefficients, weighted_coefficients, weights):
        if use_cuda:
            grid_size = (weighted_coefficients.shape[0]//weights.shape[0], 1)
            block_size = (weighted_coefficients.shape[1], weights.shape[0])
            _combine_coefficients_kernel[grid_size, block_size] \
                (coefficients, weighted_coefficients, weights)
        else:
            _combine_coefficients_loop(
                coefficients, weighted_coefficients, weights)

    # Matrix form -------------------------------------------------------------

    def _calculate_differential(
            time_step, generator, coefficient, differential, y_index, x_index):
        scratch: datatype = 0.0

        for generator_index in range(generator.shape[0]):
            if use_cuda:
                scratch = nc.fma(
                    coefficient[generator_index],
                    generator[generator_index, y_index, x_index],
                    scratch
                )
            else:
                scratch += \
                    coefficient[generator_index] \
                    * generator[generator_index, y_index, x_index]

        differential[y_index, x_index] = time_step*scratch

    if use_cuda:
        _calculate_differential = nc.jit(_calculate_differential, device=True)

        # if use_rotating:
        #     def _calculate_differential_kernel(
        #             time_step, generator, coefficient, differential):
        #         x_index = nc.threadIdx.x + stride*nc.blockIdx.y
        #         y_index = nc.threadIdx.y + stride*nc.blockIdx.z
        #         coef_index = nc.blockIdx.x
        #         if x_index < operator_size and y_index < operator_size:
        #             _calculate_differential(
        #                 time_step,
        #                 generator[
        #                     coef_index % number_of_exponentials, :, :, :
        #                 ],
        #                 coefficient[coef_index, :],
        #                 differential[coef_index, :, :], y_index, x_index
        #             )
        # else:
        def _calculate_differential_kernel(
                time_step, generator, coefficient, differential):
            x_index = nc.threadIdx.x + stride*nc.blockIdx.y
            y_index = nc.threadIdx.y + stride*nc.blockIdx.z
            coef_index = nc.blockIdx.x
            if x_index < operator_size and y_index < operator_size:
                _calculate_differential(
                    time_step, generator, coefficient[coef_index, :],
                    differential[coef_index, :, :], y_index, x_index
                )

        _calculate_differential_kernel = nc.jit(_calculate_differential_kernel)

    else:
        _calculate_differential = nb.jit(_calculate_differential)

        # if use_rotating:
        #     def _calculate_differential_loop(
        #             time_step, generator, coefficient, differential):
        #         x_index = nc.threadIdx.x + stride*nc.blockIdx.y
        #         y_index = nc.threadIdx.y + stride*nc.blockIdx.z
        #         coef_index = nc.blockIdx.x
        #         if x_index < operator_size and y_index < operator_size:
        #             _calculate_differential(
        #                 time_step,
        #                 generator[
        #                     coef_index % number_of_exponentials, :, :, :
        #                 ],
        #                 coefficient[coef_index, :],
        #                 differential[coef_index, :, :], y_index, x_index
        #             )
        # else:
        def _calculate_differential_loop(
                time_step, generator, coefficient, differential):
            for t_index in nb.prange(coefficient.shape[0]):
                for x_index in nb.prange(operator_size):
                    for y_index in nb.prange(operator_size):
                        _calculate_differential(
                            time_step, generator, coefficient[t_index, :],
                            differential[t_index, :, :], y_index, x_index
                        )

        _calculate_differential_loop = nb.jit(
            _calculate_differential_loop, **cpu_options)

    def _calculate_differential_run(
            time_step, generator, coefficient, differential):
        if use_cuda:
            grid_size = (
                coefficient.shape[0], number_of_submatrices,
                number_of_submatrices
            )
            block_size = (submatrix_size, submatrix_size)
            _calculate_differential_kernel[grid_size, block_size] \
                (time_step, generator, coefficient, differential)
        else:
            _calculate_differential_loop(
                time_step, generator, coefficient, differential)

    def _calculate_differential_rotating(
            time_step, generator, coefficient, weight, differential, y_index,
            x_index):
        scratch: datatype = 0
        scratch_mult_0: datatype = 0
        scratch_mult_1: datatype = 0
        for node_index in range(generator.shape[0]):
            for generator_index in range(generator.shape[1]):
                scratch_mult_0 = \
                    generator[node_index, generator_index, y_index, x_index] \
                    * coefficient[node_index, generator_index]
                scratch_mult_1 = time_step*weight[node_index]
                if use_cuda:
                    scratch = nc.fma(scratch_mult_0, scratch_mult_1, scratch)
                else:
                    scratch += scratch_mult_0*scratch_mult_1
        differential[y_index, x_index] = scratch

    if use_cuda:
        _calculate_differential_rotating = \
            nc.jit(_calculate_differential_rotating, device=True)

        def _calculate_differential_rotating_kernel(
                time_step, generator, coefficient, weight, differential):
            x_index = nc.threadIdx.x + stride*nc.blockIdx.y
            y_index = nc.threadIdx.y + stride*nc.blockIdx.z
            coefficient_index_start = \
                (nc.blockIdx.x//weight.shape[0]) \
                * weight.shape[1]
            coefficient_index_end = \
                coefficient_index_start + weight.shape[1]
            if x_index < operator_size and y_index < operator_size:
                _calculate_differential_rotating(
                    time_step, generator,
                    coefficient[
                        coefficient_index_start:coefficient_index_end, :],
                    weight[nc.blockIdx.x % weight.shape[0], :],
                    differential[nc.blockIdx.x, :, :],
                    y_index, x_index
                )

        _calculate_differential_rotating_kernel = \
            nc.jit(_calculate_differential_rotating_kernel)
    else:
        _calculate_differential_rotating = \
            nb.jit(_calculate_differential_rotating)

        def _calculate_differential_rotating_loop(
                time_step, generator, coefficient, weight, differential):
            for t_index in nb.prange(differential.shape[0]):
                coefficient_index_start = \
                    (t_index//weight.shape[0]) \
                    * weight.shape[1]
                coefficient_index_end = \
                    coefficient_index_start + weight.shape[1]
                for x_index in nb.prange(operator_size):
                    for y_index in nb.prange(operator_size):
                        _calculate_differential_rotating(
                        time_step, generator,
                        coefficient[
                            coefficient_index_start:coefficient_index_end, :],
                        weight[t_index % weight.shape[0], :],
                        differential[t_index, :, :],
                        y_index, x_index
                        )

        _calculate_differential_rotating_loop = \
            nb.jit(_calculate_differential_rotating_loop, **cpu_options)

    def _calculate_differential_rotating_run(
            time_step, generator, coefficient, weight, differential):
        if use_cuda:
            grid_size = (
                differential.shape[0], number_of_submatrices,
                number_of_submatrices
            )
            block_size = (submatrix_size, submatrix_size)
            _calculate_differential_rotating_kernel[grid_size, block_size](
                time_step, generator, coefficient, weight, differential)
        else:
            _calculate_differential_rotating_loop(
                time_step, generator, coefficient, weight, differential)

    def _scale_differential_basic(differential, y_index, x_index):
        differential[y_index, x_index] /= scaling_for_quartics
        if not use_residual:
            if y_index == x_index:
                differential[y_index, x_index] += 1.0

    if use_cuda:
        _scale_differential_basic = nc.jit(
            _scale_differential_basic, device=True)

        def _scale_differential_basic_kernel(differential):
            x_index = nc.threadIdx.x + stride*nc.blockIdx.y
            y_index = nc.threadIdx.y + stride*nc.blockIdx.z
            coef_index = nc.blockIdx.x
            if x_index < operator_size and y_index < operator_size:
                _scale_differential_basic(
                    differential[coef_index, :, :], y_index, x_index)

        _scale_differential_basic_kernel = nc.jit(
            _scale_differential_basic_kernel)
    else:
        _scale_differential_basic = nb.jit(
            _scale_differential_basic)

        def _scale_differential_basic_loop(differential):
            for t_index in nb.prange(differential.shape[0]):
                for x_index in nb.prange(operator_size):
                    for y_index in nb.prange(operator_size):
                        _scale_differential_basic(
                            differential[t_index, :, :], y_index, x_index)

        _scale_differential_basic_loop = nb.jit(
            _scale_differential_basic_loop, **cpu_options)

    def _scale_differential_basic_run(differential):
        if use_cuda:
            grid_size = (
                differential.shape[0], number_of_submatrices,
                number_of_submatrices
            )
            block_size = (submatrix_size, submatrix_size)
            _scale_differential_basic_kernel[grid_size, block_size] \
                (differential)
        else:
            _scale_differential_basic_loop(differential)


    # Cayley ------------------------------------------------------------------

    def _negate_superoperator(positive, negative, y_index, x_index):
        negative[y_index, x_index] = -positive[y_index, x_index]

    if use_cuda:
        _negate_superoperator = nc.jit(_negate_superoperator, device=True)

    #     def _calculate_cayley_kernel(differential):
    #         if nc.threadIdx.y < operator_size \
    #                 and nc.threadIdx.x < wavefunction_size:
    #             scratch = nc.shared.array(
    #                 (operator_size, operator_size, 2),
    #                 dtype=datatype
    #             )

    #             for x_index_stride in range(wavefunction_size):
    #                 _negate_superoperator(
    #                     differential[nc.blockIdx.x, :, :, :],
    #                     scratch,
    #                     nc.threadIdx.y,
    #                     nc.threadIdx.x + x_index_stride*wavefunction_size
    #                 )
    #             nc.syncthreads()

    #             diff = differential[nc.blockIdx.x, :, :, :]

    #             for node_index in range(operator_size):
    #                 # Scale row
    #                 if nc.threadIdx.x == 1:

    #                     node_real = scratch[node_index, node_index, 0]
    #                     node_imag = scratch[node_index, node_index, 1]
    #                     div_real = (1 + node_real) \
    #                         / ((1 + node_real)**2 + node_imag**2)
    #                     div_imag = -node_imag \
    #                         / ((1 + node_real)**2 + node_imag**2)
    #                 nc.syncthreads()

    #                 if nc.threadIdx.x == 1:
    #                     eval_real = \
    #                         div_real*scratch[node_index, nc.threadIdx.y, 0] \
    #                         - div_imag*scratch[node_index, nc.threadIdx.y, 1]
    #                     eval_imag = \
    #                         div_real*scratch[node_index, nc.threadIdx.y, 1] \
    #                         + div_imag*scratch[node_index, nc.threadIdx.y, 0]
    #                     if nc.threadIdx.y == node_index:
    #                         eval_real -= \
    #                             node_real*div_real - node_imag*div_imag
    #                         eval_imag -= \
    #                             node_imag*div_real + node_real*div_imag
    #                     scratch[node_index, nc.threadIdx.y, 0] = eval_real
    #                     scratch[node_index, nc.threadIdx.y, 1] = eval_imag

    #                     eval_real = \
    #                         div_real*diff[node_index, nc.threadIdx.y, 0] \
    #                         - div_imag*diff[node_index, nc.threadIdx.y, 1]
    #                     eval_imag = \
    #                         div_real*diff[node_index, nc.threadIdx.y, 1] \
    #                         + div_imag*diff[node_index, nc.threadIdx.y, 0]
    #                     if nc.threadIdx.y == node_index:
    #                         eval_real -= \
    #                             node_real*div_real - node_imag*div_imag
    #                         eval_imag -= \
    #                             node_imag*div_real + node_real*div_imag
    #                     diff[node_index, nc.threadIdx.y, 0] = eval_real
    #                     diff[node_index, nc.threadIdx.y, 1] = eval_imag

    #                 nc.syncthreads()

    #                 # Eliminate rows
    #                 for x_index_stride in range(wavefunction_size):
    #                     x_index = nc.threadIdx.x \
    #                         + x_index_stride*wavefunction_size

    #                     if x_index != node_index:
    #                         scale_real = -scratch[x_index, node_index, 0]
    #                         scale_imag = -scratch[x_index, node_index, 1]
    #                     nc.syncthreads()

    #                     if x_index != node_index:
    #                         eval_real = scale_real \
    #                             * scratch[x_index, nc.threadIdx.y, 0] \
    #                             - scale_imag \
    #                             * scratch[x_index, nc.threadIdx.y, 1]
    #                         eval_imag = scale_real \
    #                             * scratch[x_index, nc.threadIdx.y, 1] \
    #                             + scale_imag \
    #                             * scratch[x_index, nc.threadIdx.y, 0]
    #                         if nc.threadIdx.y == node_index:
    #                             eval_real += scale_real
    #                             eval_imag += scale_imag
    #                     nc.syncthreads()

    #                     if x_index != node_index:
    #                         scratch[x_index, nc.threadIdx.y, 0] += eval_real
    #                         scratch[x_index, nc.threadIdx.y, 1] += eval_imag
    #                     nc.syncthreads()

    #                     if x_index != node_index:
    #                         eval_real = scale_real \
    #                             * diff[x_index, nc.threadIdx.y, 0] \
    #                             - scale_imag*diff[x_index, nc.threadIdx.y, 1]
    #                         eval_imag = scale_real \
    #                             * diff[x_index, nc.threadIdx.y, 1] \
    #                             + scale_imag*diff[x_index, nc.threadIdx.y, 0]
    #                         if nc.threadIdx.y == node_index:
    #                             eval_real += scale_real
    #                             eval_imag += scale_imag
    #                     nc.syncthreads()

    #                     if x_index != node_index:
    #                         diff[x_index, nc.threadIdx.y, 0] += eval_real
    #                         diff[x_index, nc.threadIdx.y, 1] += eval_imag
    #                     nc.syncthreads()

    #     _calculate_cayley_kernel = nc.jit(_calculate_cayley_kernel)

    # def _calculate_cayley_run(differential):
    #     if use_cuda:
    #         grid_size = (differential.shape[0], 1)
    #         block_size = (wavefunction_size, operator_size)
    #         _calculate_cayley_kernel[grid_size, block_size](differential)

    # Repeated squaring -------------------------------------------------------

    def _square_superoperator(inp, out, y_index, x_index):
        if use_residual:
            out_scratch: datatype = 2*inp[y_index, x_index]
        else:
            out_scratch: datatype = 0.0

        for trace_index in range(operator_size):
            # TODO: unroll?
            if use_cuda:
                out_scratch = nc.fma(
                        inp[y_index, trace_index],
                        inp[trace_index, x_index],
                        out_scratch
                )
            else:
                out_scratch += \
                    inp[y_index, trace_index]*inp[trace_index, x_index]

        out[y_index, x_index] = out_scratch

    def _multiply_superoperator(left, right, out, y_index, x_index):
        if use_residual:
            out_scratch: datatype = \
                left[y_index, x_index] + right[y_index, x_index]
        else:
            out_scratch: datatype = 0.0

        for trace_index in range(operator_size):
            if use_cuda:
                out_scratch = nc.fma(
                        left[y_index, trace_index],
                        right[trace_index, x_index],
                        out_scratch
                )
            else:
                out_scratch += \
                    left[y_index, trace_index]*right[trace_index, x_index]
        out[y_index, x_index] = out_scratch

    def _copy_superoperator(original, clone, y_index, x_index):
        clone[y_index, x_index] = original[y_index, x_index]

    if use_cuda:
        # Compile squaring
        _square_superoperator = nc.jit(_square_superoperator, device=True)
        _multiply_superoperator = nc.jit(_multiply_superoperator, device=True)
        _copy_superoperator = nc.jit(_copy_superoperator, device=True)

        # Wrap in kernel
        def _square_superoperator_kernel(inp, out):
            x_index = nc.threadIdx.x + stride*nc.blockIdx.y
            y_index = nc.threadIdx.y + stride*nc.blockIdx.z
            t_index = nc.blockIdx.x
            if x_index < operator_size and y_index < operator_size:
                inp_sample = inp[t_index, :, :]
                out_sample = out[t_index, :, :]
                _square_superoperator(inp_sample, out_sample, y_index, x_index)

        _square_superoperator_kernel = nc.jit(
            _square_superoperator_kernel
        )

        def _multiply_superoperator_kernel(left, right, out):
            x_index = nc.threadIdx.x + stride*nc.blockIdx.y
            y_index = nc.threadIdx.y + stride*nc.blockIdx.z
            t_index = nc.blockIdx.x
            if x_index < operator_size and y_index < operator_size:
                left_sample = left[t_index, :, :]
                right_sample = right[t_index, :, :]
                out_sample = out[t_index, :, :]
                _multiply_superoperator(
                    left_sample, right_sample, out_sample, y_index, x_index)

        _multiply_superoperator_kernel = nc.jit(
            _multiply_superoperator_kernel
        )

        def _multiply_superoperator_quadrature_kernel(
                left, right, out, offset):
            x_index = nc.threadIdx.x + stride*nc.blockIdx.y
            y_index = nc.threadIdx.y + stride*nc.blockIdx.z
            t_index = nc.blockIdx.x
            if x_index < operator_size and y_index < operator_size:
                left_sample = left[
                    offset + number_of_exponentials*t_index, :, :]
                right_sample = right[t_index, :, :]
                out_sample = out[t_index, :, :]
                _multiply_superoperator(
                    left_sample, right_sample, out_sample, y_index, x_index)

        _multiply_superoperator_quadrature_kernel = nc.jit(
            _multiply_superoperator_quadrature_kernel
        )

        def _copy_superoperator_quadrature_kernel(original, clone):
            x_index = nc.threadIdx.x + stride*nc.blockIdx.y
            y_index = nc.threadIdx.y + stride*nc.blockIdx.z
            t_index = nc.blockIdx.x
            if x_index < operator_size and y_index < operator_size:
                original_sample = original[t_index, :, :]
                clone_sample = clone[t_index, :, :]
                _copy_superoperator(
                    original_sample, clone_sample, y_index, x_index)

        _copy_superoperator_quadrature_kernel = nc.jit(
            _copy_superoperator_quadrature_kernel
        )
    else:
        # Compile squaring
        _square_superoperator = nb.jit(_square_superoperator)
        _multiply_superoperator = nb.jit(_multiply_superoperator)
        _copy_superoperator = nb.jit(_copy_superoperator)

        # Wrap in kernel
        def _square_superoperator_loop(inp, out):
            for t_index in nb.prange(inp.shape[0]):
                for x_index in nb.prange(operator_size):
                    for y_index in nb.prange(operator_size):
                        inp_sample = inp[t_index, :, :]
                        out_sample = out[t_index, :, :]
                        _square_superoperator(
                            inp_sample, out_sample, y_index, x_index)

        _square_superoperator_loop = nb.jit(
            _square_superoperator_loop, **cpu_options
        )

        def _multiply_superoperator_loop(left, right, out):
            for t_index in nb.prange(left.shape[0]):
                for x_index in nb.prange(operator_size):
                    for y_index in nb.prange(operator_size):
                        left_sample = left[t_index, :, :]
                        right_sample = right[t_index, :, :]
                        out_sample = out[t_index, :, :]
                        _multiply_superoperator(
                            left_sample, right_sample, out_sample,
                            y_index, x_index
                        )

        _multiply_superoperator_loop = nb.jit(
            _multiply_superoperator_loop, **cpu_options
        )

        def _multiply_superoperator_quadrature_loop(
                left, right, out, offset):
            for t_index in nb.prange(left.shape[0]//number_of_exponentials):
                for x_index in nb.prange(operator_size):
                    for y_index in nb.prange(operator_size):
                        left_sample = left[
                            offset + number_of_exponentials*t_index, :, :]
                        right_sample = right[t_index, :, :]
                        out_sample = out[t_index, :, :]
                        _multiply_superoperator(
                            left_sample, right_sample, out_sample,
                            y_index, x_index
                        )

        _multiply_superoperator_quadrature_loop = nb.jit(
            _multiply_superoperator_quadrature_loop, **cpu_options
        )

        def _copy_superoperator_quadrature_loop(original, clone):
            # print(original.shape)
            # print(clone.shape)
            for t_index in nb.prange(original.shape[0]):
                for x_index in nb.prange(operator_size):
                    for y_index in nb.prange(operator_size):
                        original_sample = original[t_index, :, :]
                        clone_sample = clone[t_index, :, :]
                        _copy_superoperator(
                            original_sample, clone_sample, y_index, x_index)

        _copy_superoperator_quadrature_loop = nb.jit(
            _copy_superoperator_quadrature_loop, **cpu_options
        )

    def _repeated_quartic_superoperator_run(superoperators, scratch):
        if use_cuda:
            grid_size = (
                superoperators.shape[0], number_of_submatrices,
                number_of_submatrices
            )
            block_size = (submatrix_size, submatrix_size)
            for _ in range(number_of_quartic_repeats):
                _square_superoperator_kernel[grid_size, block_size](
                        superoperators, scratch)
                _square_superoperator_kernel[grid_size, block_size](
                        scratch, superoperators)
        else:
            for _ in range(number_of_quartic_repeats):
                _square_superoperator_loop(superoperators, scratch)
                _square_superoperator_loop(scratch, superoperators)

    # Rotating frame ----------------------------------------------------------

    if use_rotating:
        def _apply_eig_double(inp, out, doubles, y_index, x_index):
            if use_residual:
                scratch_real: datatype = \
                    (1 + doubles[y_index, 0])*inp[2*y_index, x_index] \
                    - doubles[y_index, 1]*inp[2*y_index + 1, x_index]

                scratch_imag: datatype = \
                    doubles[y_index, 1]*inp[2*y_index, x_index] \
                    + (1 + doubles[y_index, 0])*inp[2*y_index + 1, x_index]

                if x_index == 2*y_index:
                    scratch_real += doubles[y_index, 0]
                    scratch_imag += doubles[y_index, 1]
                if x_index == 2*y_index + 1:
                    scratch_real -= doubles[y_index, 1]
                    scratch_imag += doubles[y_index, 0]

            else:
                scratch_real: datatype = \
                    doubles[y_index, 0]*inp[2*y_index, x_index] \
                    - doubles[y_index, 1]*inp[2*y_index + 1, x_index]
                scratch_imag: datatype = \
                    doubles[y_index, 1]*inp[2*y_index, x_index] \
                    + doubles[y_index, 0]*inp[2*y_index + 1, x_index]

            out[2*y_index, x_index] = scratch_real
            out[2*y_index + 1, x_index] = scratch_imag

        def _apply_eig_single(inp, out, singles, y_index, x_index):
            if use_residual:
                scratch: datatype = \
                    (1 + singles[y_index])*inp[y_index, x_index]
                if y_index + 2*doubles_size == x_index:
                    scratch += singles[y_index]
            else:
                scratch = singles[y_index]*inp[y_index, x_index]

            out[y_index, x_index] = scratch

        if use_cuda:
            _apply_eig_double = nc.jit(_apply_eig_double, device=True)
            _apply_eig_single = nc.jit(_apply_eig_single, device=True)

            def _apply_eig_double_kernel(inp, out, doubles):
                x_index = nc.threadIdx.x + stride*nc.blockIdx.y
                y_index = nc.threadIdx.y + stride*nc.blockIdx.z
                t_index = nc.blockIdx.x
                if x_index < operator_size and y_index < doubles_size:
                    _apply_eig_double(
                        inp[t_index, :, :], out[t_index, :, :],
                        doubles, y_index, x_index
                    )

            def _apply_eig_single_kernel(inp, out, singles):
                x_index = nc.threadIdx.x + stride*nc.blockIdx.y
                y_index = nc.threadIdx.y + stride*nc.blockIdx.z
                t_index = nc.blockIdx.x
                if x_index < operator_size and y_index < singles_size:
                    _apply_eig_single(
                        inp[t_index, 2*doubles_size:, :],
                        out[t_index, 2*doubles_size:, :],
                        singles, y_index, x_index
                    )

            _apply_eig_double_kernel = nc.jit(_apply_eig_double_kernel)
            _apply_eig_single_kernel = nc.jit(_apply_eig_single_kernel)

        else:
            _apply_eig_double = nb.jit(_apply_eig_double)
            _apply_eig_single = nb.jit(_apply_eig_single)

            def _apply_eig_double_loop(inp, out, doubles):
                for t_index in nb.prange(inp.shape[0]):
                    for x_index in nb.prange(operator_size):
                        for y_index in nb.prange(doubles_size):
                            _apply_eig_double(
                                inp[t_index, :, :], out[t_index, :, :],
                                doubles, y_index, x_index
                            )

            def _apply_eig_single_loop(inp, out, singles):
                for t_index in nb.prange(inp.shape[0]):
                    for x_index in nb.prange(operator_size):
                        for y_index in nb.prange(doubles_size):
                            _apply_eig_single(
                                inp[t_index, 2*doubles_size:, :],
                                out[t_index, 2*doubles_size:, :],
                                singles, y_index, x_index
                            )

            _apply_eig_double_loop = nb.jit(
                _apply_eig_double_loop, **cpu_options)
            _apply_eig_single_loop = nb.jit(
                _apply_eig_single_loop, **cpu_options)

        def _apply_eig_run(superoperators, scratch, doubles, singles):
            if use_cuda:
                grid_size = (
                    superoperators.shape[0], number_of_submatrices,
                )
                block_size = (submatrix_size, submatrix_size)

                if doubles_size > 0:
                    _apply_eig_double_kernel[grid_size, block_size](
                        superoperators, scratch, doubles
                    )
                if singles_size > 0:
                    _apply_eig_single_kernel[grid_size, block_size](
                        superoperators, scratch, singles
                    )
                _copy_superoperator_quadrature_kernel[grid_size, block_size](
                    scratch, superoperators
                )
            else:
                if doubles_size > 0:
                    _apply_eig_double_loop(superoperators, scratch, doubles)
                if singles_size > 0:
                    _apply_eig_single_loop(superoperators, scratch, singles)
                _copy_superoperator_quadrature_loop(scratch, superoperators)

    def _multiply_superoperator_right(right, inp, out, y_index, x_index):
        if use_residual:
            out_scratch: datatype = \
                right[y_index, x_index]
            if y_index == x_index:
                out_scratch -= 1
        else:
            out_scratch: datatype = 0.0

        for trace_index in range(right.shape[0]):
            if use_cuda:
                out_scratch = nc.fma(
                        inp[y_index, trace_index],
                        right[trace_index, x_index],
                        out_scratch
                )
            else:
                out_scratch += \
                    inp[y_index, trace_index]*right[trace_index, x_index]
        out[y_index, x_index] = out_scratch

    def _multiply_superoperator_left(left, inp, out, y_index, x_index):
        if use_residual:
            out_scratch: datatype = \
                left[y_index, x_index]
            if y_index == x_index:
                out_scratch -= 1
        else:
            out_scratch: datatype = 0.0

        for trace_index in range(left.shape[1]):
            if use_cuda:
                out_scratch = nc.fma(
                        left[y_index, trace_index],
                        inp[trace_index, x_index],
                        out_scratch
                )
            else:
                out_scratch += \
                    left[y_index, trace_index]*inp[trace_index, x_index]
        out[y_index, x_index] = out_scratch

    def _add_superoperator(shift, inp, out, y_index, x_index):
        out[y_index, x_index] = \
            shift[y_index, x_index] + inp[y_index, x_index]

    if use_cuda:
        _multiply_superoperator_right = nc.jit(
            _multiply_superoperator_right, device=True)
        _multiply_superoperator_left = nc.jit(
            _multiply_superoperator_left, device=True)
        _add_superoperator = nc.jit(_add_superoperator, device=True)

        def _apply_global_sandwich_right_kernel(right, inp, out):
            x_index = nc.threadIdx.x + stride*nc.blockIdx.y
            y_index = nc.threadIdx.y + stride*nc.blockIdx.z
            t_index = nc.blockIdx.x
            if x_index < right.shape[1] and y_index < right.shape[0]:
                _multiply_superoperator_right(
                    right, inp[t_index, :, :],
                    out[t_index, :, :], y_index, x_index
                )

        def _apply_global_sandwich_left_kernel(left, inp, out):
            x_index = nc.threadIdx.x + stride*nc.blockIdx.y
            y_index = nc.threadIdx.y + stride*nc.blockIdx.z
            t_index = nc.blockIdx.x
            if x_index < left.shape[0] and y_index < left.shape[0]:
                _multiply_superoperator_left(
                    left, inp[t_index, :, :], out[t_index, :, :],
                    y_index, x_index
                )

        def _apply_global_addition_kernel(shift, inp, out):
            x_index = nc.threadIdx.x + stride*nc.blockIdx.y
            y_index = nc.threadIdx.y + stride*nc.blockIdx.z
            t_index = nc.blockIdx.x
            if x_index < shift.shape[0] and y_index < shift.shape[0]:
                _add_superoperator(
                    shift, inp[t_index, :, :], out[t_index, :, :],
                    y_index, x_index
                )

        _apply_global_sandwich_right_kernel = nc.jit(
            _apply_global_sandwich_right_kernel
        )
        _apply_global_sandwich_left_kernel = nc.jit(
            _apply_global_sandwich_left_kernel
        )
        _apply_global_addition_kernel = nc.jit(_apply_global_addition_kernel)

    else:
        _multiply_superoperator_right = nb.jit(_multiply_superoperator_right)
        _multiply_superoperator_left = nb.jit(_multiply_superoperator_left)
        _add_superoperator = nb.jit(_add_superoperator)

        def _apply_global_sandwich_right_loop(right, inp, out):
            for t_index in nb.prange(inp.shape[0]):
                for x_index in nb.prange(right.shape[1]):
                    for y_index in nb.prange(right.shape[0]):
                        _multiply_superoperator_right(
                            right, inp[t_index, :, :],
                            out[t_index, :, :], y_index, x_index
                        )

        def _apply_global_sandwich_left_loop(left, inp, out):
            for t_index in nb.prange(inp.shape[0]):
                for x_index in nb.prange(left.shape[0]):
                    for y_index in nb.prange(left.shape[0]):
                        _multiply_superoperator_left(
                            left, inp[t_index, :, :], out[t_index, :, :],
                            y_index, x_index
                        )

        def _apply_global_addition_loop(shift, inp, out):
            for t_index in nb.prange(inp.shape[0]):
                for x_index in nb.prange(shift.shape[1]):
                    for y_index in nb.prange(shift.shape[0]):
                        _add_superoperator(
                            shift, inp[t_index, :, :], out[t_index, :, :],
                            y_index, x_index
                        )

        _apply_global_sandwich_right_loop = nb.jit(
            _apply_global_sandwich_right_loop, **cpu_options
        )
        _apply_global_sandwich_left_loop = nb.jit(
            _apply_global_sandwich_left_loop, **cpu_options
        )
        _apply_global_addition_loop = nb.jit(
            _apply_global_addition_loop, **cpu_options
        )

    def _apply_global_sandwich_run(left, right, time_evolution, scratch):
        if use_cuda:
            grid_size = (
                time_evolution.shape[0], number_of_submatrices_density,
                number_of_submatrices_density
            )
            block_size = (submatrix_size_density, submatrix_size_density)

            _apply_global_sandwich_right_kernel[grid_size, block_size](
                right, time_evolution, scratch)
            _apply_global_sandwich_left_kernel[grid_size, block_size](
                left, scratch, time_evolution)
        else:
            _apply_global_sandwich_right_loop(right, time_evolution, scratch)
            _apply_global_sandwich_left_loop(left, scratch, time_evolution)

    def _apply_global_addition_run(shift, time_evolution, scratch):
        if use_cuda:
            grid_size = (
                time_evolution.shape[0], number_of_submatrices_density,
                number_of_submatrices_density
            )
            block_size = (submatrix_size_density, submatrix_size_density)

            _apply_global_addition_kernel[grid_size, block_size](
                shift, time_evolution, scratch)
            _apply_global_addition_kernel[grid_size, block_size](
                shift, scratch, time_evolution)
        else:
            _apply_global_addition_loop(shift, time_evolution, scratch)
            _apply_global_addition_loop(shift, scratch, time_evolution)

    # Combine samples at different quadrature nodes ---------------------------

    if use_cuda:
        def _id_superoperator_kernel(time_evolutions):
            x_index = nc.threadIdx.x + stride*nc.blockIdx.y
            y_index = nc.threadIdx.y + stride*nc.blockIdx.z
            t_index = nc.blockIdx.x
            if x_index < operator_size and y_index < operator_size:
                time_evolutions[t_index, y_index, x_index] = 0
                if not use_residual:
                    if y_index == x_index:
                        time_evolutions[t_index, y_index, x_index] = 1

        _id_superoperator_kernel = nc.jit(_id_superoperator_kernel)

        def _basic_combine_kernel(time_evolutions, time_index, scratch):
            x_index = nc.threadIdx.x + stride*nc.blockIdx.y
            y_index = nc.threadIdx.y + stride*nc.blockIdx.z
            if x_index < operator_size and y_index < operator_size:
                _multiply_superoperator(
                    time_evolutions[time_index + 1, :, :],
                    time_evolutions[time_index, :, :], scratch,
                    y_index, x_index
                )

        def _basic_combine_copy_kernel(time_evolutions, time_index, scratch):
            x_index = nc.threadIdx.x + stride*nc.blockIdx.y
            y_index = nc.threadIdx.y + stride*nc.blockIdx.z
            if x_index < operator_size and y_index < operator_size:
                _copy_superoperator(
                    scratch, time_evolutions[time_index + 1, :, :],
                    y_index, x_index
                )

        _basic_combine_kernel = nc.jit(_basic_combine_kernel)
        _basic_combine_copy_kernel = nc.jit(_basic_combine_copy_kernel)

    else:
        def _id_superoperator_loop(time_evolutions):
            for t_index in nb.prange(time_evolutions.shape[0]):
                for x_index in nb.prange(operator_size):
                    for y_index in nb.prange(operator_size):
                        time_evolutions[t_index, y_index, x_index] = 0
                        if not use_residual:
                            if y_index == x_index:
                                time_evolutions[t_index, y_index, x_index] = 1

        _id_superoperator_loop = nb.jit(_id_superoperator_loop, **cpu_options)

        def _basic_combine_loop(time_evolutions, time_index, scratch):
            for x_index in nb.prange(operator_size):
                for y_index in nb.prange(operator_size):
                    _multiply_superoperator(
                        time_evolutions[time_index + 1, :, :],
                        time_evolutions[time_index, :, :], scratch,
                        y_index, x_index
                    )

        def _basic_combine_copy_loop(time_evolutions, time_index, scratch):
            for x_index in nb.prange(operator_size):
                for y_index in nb.prange(operator_size):
                    _copy_superoperator(
                        scratch, time_evolutions[time_index + 1, :, :],
                        y_index, x_index
                    )

        _basic_combine_loop = nb.jit(_basic_combine_loop, **cpu_options)
        _basic_combine_copy_loop = nb.jit(
            _basic_combine_copy_loop, **cpu_options)

    def _quadrature_combine_run(exponentials, time_evolution, scratch):
        if use_cuda:
            grid_size = (
                time_evolution.shape[0], number_of_submatrices,
                number_of_submatrices
            )
            block_size = (submatrix_size, submatrix_size)

        for exponential_index in range(0, number_of_exponentials, 2):
            if use_cuda:
                _multiply_superoperator_quadrature_kernel[grid_size, block_size](
                    exponentials, time_evolution, scratch,
                        number_of_exponentials - exponential_index - 1)
            else:
                _multiply_superoperator_quadrature_loop(
                    exponentials, time_evolution, scratch,
                    number_of_exponentials - exponential_index - 1)
            if exponential_index + 1 < number_of_exponentials:
                if use_cuda:
                    _multiply_superoperator_quadrature_kernel[
                        grid_size, block_size](
                        exponentials, scratch, time_evolution,
                        number_of_exponentials - exponential_index - 2)
                else:
                    _multiply_superoperator_quadrature_loop(
                        exponentials, scratch, time_evolution,
                        number_of_exponentials - exponential_index - 2)
            else:
                if use_cuda:
                    _copy_superoperator_quadrature_kernel[
                        grid_size, block_size](scratch, time_evolution)
                else:
                    _copy_superoperator_quadrature_loop(
                        scratch, time_evolution)

    def _id_superoperator_run(time_evolution):
        if use_cuda:
            grid_size = (
                time_evolution.shape[0], number_of_submatrices,
                number_of_submatrices
            )
            block_size = (submatrix_size, submatrix_size)
            _id_superoperator_kernel[grid_size, block_size] \
                (time_evolution)
        else:
            _id_superoperator_loop(time_evolution)

    def _basic_combine_run(time_evolutions, scratch):
        if use_cuda:
            grid_size = (1, number_of_submatrices, number_of_submatrices)
            block_size = (submatrix_size, submatrix_size)
        for time_index in range(0, time_evolutions.shape[0] - 1):
            if use_cuda:
                _basic_combine_kernel[grid_size, block_size] \
                    (time_evolutions, time_index, scratch)
                _basic_combine_copy_kernel[grid_size, block_size] \
                    (time_evolutions, time_index, scratch)
            else:
                # print(time_index)
                _basic_combine_loop(time_evolutions, time_index, scratch)
                _basic_combine_copy_loop(time_evolutions, time_index, scratch)

    # Accumulate --------------------------------------------------------------

    def _multiply_superoperator_operator(superoperator, operator, out, index):
        if use_residual:
            scratch = operator[index]
        else:
            scratch: datatype = 0.0

        for trace_index in range(operator_size_density):
            if use_cuda:
                scratch = nc.fma(
                    superoperator[index, trace_index],
                    operator[trace_index],
                    scratch
                )
            else:
                scratch += \
                    superoperator[index, trace_index]*operator[trace_index]

        out[index] = scratch

    if use_cuda:
        _multiply_superoperator_operator = nc.jit(
            _multiply_superoperator_operator,
            device=True
        )

        def _apply_time_evolution_kernel(
                time_evolutions, density_operator_initial, density_operators):
            x_index = nc.threadIdx.x
            t_index = nc.blockIdx.x
            if x_index < operator_size_density:
                _multiply_superoperator_operator(
                    time_evolutions[t_index, :, :],
                    density_operator_initial,
                    density_operators[t_index, :],
                    x_index
                )

        _apply_time_evolution_kernel = nc.jit(_apply_time_evolution_kernel)
    else:
        _multiply_superoperator_operator = nb.jit(
            _multiply_superoperator_operator)

        def _apply_time_evolution_loop(
                time_evolutions, density_operator_initial, density_operators):
            for t_index in nb.prange(time_evolutions.shape[0]):
                for x_index in nb.prange(operator_size_density):
                    _multiply_superoperator_operator(
                        time_evolutions[t_index, :, :],
                        density_operator_initial,
                        density_operators[t_index, :],
                        x_index
                    )

        _apply_time_evolution_loop = nb.jit(
            _apply_time_evolution_loop, **cpu_options)

    def _apply_time_evolution_run(
            time_evolutions, density_operator_initial, density_operators):
        if use_cuda:
            grid_size = (time_evolutions.shape[0], 1)
            block_size = (operator_size_density, 1)
            _apply_time_evolution_kernel[grid_size, block_size] \
                (time_evolutions, density_operator_initial, density_operators)
        else:
            _apply_time_evolution_loop(
                time_evolutions, density_operator_initial, density_operators)

    # Unitary -----------------------------------------------------------------

    def _kronecker_product(
            time_evolutions_unitary, time_evolutions, y_index, x_index):
        y_index_out = y_index//(time_evolutions.shape[0]//2)
        x_index_out = x_index//(time_evolutions.shape[1]//2)
        y_index_in = y_index % (time_evolutions.shape[0]//2)
        x_index_in = x_index % (time_evolutions.shape[1]//2)

        out_r: datatype = time_evolutions[2*y_index_out, 2*x_index_out]
        out_i: datatype = time_evolutions[2*y_index_out + 1, 2*x_index_out]
        in_r: datatype = time_evolutions[2*y_index_in, 2*x_index_in]
        in_i: datatype = time_evolutions[2*y_index_in + 1, 2*x_index_in]

        scratch_r: datatype = out_r*in_r + out_i*in_i
        scratch_i: datatype = out_r*in_i - out_i*in_r

        if use_residual:
            scratch_r += (y_index_out == x_index_out)*out_r
            scratch_r += (y_index_in == x_index_in)*out_r
            scratch_i += (y_index_out == x_index_out)*out_i
            scratch_i -= (y_index_in == x_index_in)*out_i

        time_evolutions_unitary[y_index, x_index, 0] = scratch_r
        time_evolutions_unitary[y_index, x_index, 1] = scratch_i

    # Simulation --------------------------------------------------------------

    def simulate(
            density_operator_initial,
            time_start, time_end, time_step):

        if verbose:
            print("Starting simulator")

        # Remove numba warnings
        warnings.simplefilter("ignore", nb.core.errors.NumbaPerformanceWarning)

        # Time
        number_of_samples = int((time_end - time_start)/time_step)

        # Convert density operator into real matrix
        if verbose:
            print("Get flat density")
        wavefunction_size = density_operator_initial.shape[0]
        density_operator_initial_real = np.empty(
            (wavefunction_size, wavefunction_size, 2),
            dtype=datatype
        )
        density_operator_initial_real[:, :, 0] = \
            np.real(density_operator_initial)
        density_operator_initial_real[:, :, 1] = \
            np.imag(density_operator_initial)
        density_operator_initial = density_operator_initial_real

        # Flatten density operator
        density_operator_initial_flat = \
            np.empty(operator_size_density, dtype=datatype)
        for operator_index in range(vectorisation_map.shape[0]):
            y_index = vectorisation_map[operator_index, 0]
            x_index = vectorisation_map[operator_index, 1]
            c_index = vectorisation_map[operator_index, 2]

            density_operator_initial_flat[operator_index] = \
                density_operator_initial[y_index, x_index, c_index]

        # Project into equivalence classes
        if use_kernel:
            if verbose:
                print("Get density in equivalence class")
            # density_operator_initial_flat_projection = \
            #     image_projection.T@density_operator_initial_flat
            # density_operator_initial_flat_kernel = \
            #     density_operator_initial_flat \
            #     - image_projection@density_operator_initial_flat_projection

            # density_operator_initial_flat_kernel = \
            #     density_operator_initial_flat \
            #     - image_projection@image_projection.T \
            #     @ density_operator_initial_flat

        # Rotating frame
        if use_rotating:
            if verbose:
                print(
                    "Calculate block diagonal matrix exponential for "
                    "generalised rotating frame."
                )

            if verbose:
                print("  Declare memory")
            time_sample_zero = np.empty(
                (sample_quadrature.size + 1), dtype=datatype)
            time_sample_zero[:-1] = \
                (sample_quadrature)*time_step/number_of_fine_divisions
            time_sample_zero[-1] = time_step/number_of_fine_divisions
            # print(time_sample_zero)
            generators_rotating = np.empty(
                (
                    time_sample_zero.size - 1, generators.shape[0],
                    generators.shape[1], generators.shape[2]
                ), dtype=datatype
            )

            if verbose:
                print("  Calculate 1x1 exponentials")
            singles_forward = \
                np.empty((time_sample_zero.size, singles.size), dtype=datatype)
            singles_backward = np.empty_like(singles_forward)
            for index in range(time_sample_zero.size - use_residual):
                singles_forward[index, :] = \
                    np.exp(singles*time_sample_zero[index])
                singles_backward[index, :] = \
                    np.exp(-singles*time_sample_zero[index])

            if verbose:
                print("  Calculate 2x2 exponentials")
            doubles_forward = np.empty(
                (time_sample_zero.size, doubles.shape[0], doubles.shape[1]),
                dtype=datatype
            )
            doubles_backward = np.empty_like(doubles_forward)
            for index in range(time_sample_zero.size - use_residual):
                attenuation = np.exp(doubles[:, 0]*time_sample_zero[index])
                amplification = np.exp(-doubles[:, 0]*time_sample_zero[index])
                sine = np.sin(doubles[:, 1]*time_sample_zero[index])
                cosine = np.cos(doubles[:, 1]*time_sample_zero[index])
                doubles_forward[index, :, 0] = attenuation*cosine
                doubles_forward[index, :, 1] = -attenuation*sine
                doubles_backward[index, :, 0] = amplification*cosine
                doubles_backward[index, :, 1] = amplification*sine

            if use_residual:
                singles_forward[-1, :] = \
                    np.expm1(singles*time_sample_zero[-1])
                sin2 = -2*(np.sin(doubles[:, 1]*time_sample_zero[-1]/2)**2)
                sin1 = np.sin(doubles[:, 1]*time_sample_zero[-1])
                expmr = np.expm1(doubles[:, 0]*time_sample_zero[-1])
                doubles_forward[-1, :, 0] = (1 + expmr)*sin2 + expmr
                doubles_forward[-1, :, 1] = -(1 + expmr)*sin1

            if verbose:
                print("  Bring generators into generalised rotating frame")
            for sample_index in range(time_sample_zero.size - 1):
                generators_rotating[sample_index, :, :, :] = generators

                for eigen_index in range(doubles.shape[0]):
                    temp_real = \
                        doubles_backward[sample_index, eigen_index, 0] \
                        * generators_rotating[
                            sample_index, :, 2*eigen_index, :] \
                        - doubles_backward[sample_index, eigen_index, 1] \
                        * generators_rotating[
                            sample_index, :, 2*eigen_index + 1, :]
                    temp_imag = \
                        doubles_backward[sample_index, eigen_index, 1] \
                        * generators_rotating[
                            sample_index, :, 2*eigen_index, :] \
                        + doubles_backward[sample_index, eigen_index, 0] \
                        * generators_rotating[
                            sample_index, :, 2*eigen_index + 1, :]
                    generators_rotating[sample_index, :, 2*eigen_index, :] = \
                        temp_real
                    generators_rotating[
                        sample_index, :, 2*eigen_index + 1, :] = temp_imag

                for eigen_index in range(singles.shape[0]):
                    generators_rotating[
                        sample_index, :, 2*doubles.shape[0] + eigen_index, :
                    ] *= singles_backward[sample_index, eigen_index]

                for eigen_index in range(doubles.shape[0]):
                    temp_real = doubles_forward[sample_index, eigen_index, 0] \
                        * generators_rotating[
                            sample_index, :, :, 2*eigen_index] \
                        + doubles_forward[sample_index, eigen_index, 1] \
                        * generators_rotating[
                            sample_index, :, :, 2*eigen_index + 1]
                    temp_imag = doubles_forward[sample_index, eigen_index, 1] \
                        * generators_rotating[
                            sample_index, :, :, 2*eigen_index] \
                        - doubles_forward[sample_index, eigen_index, 0] \
                        * generators_rotating[
                            sample_index, :, :, 2*eigen_index + 1]
                    generators_rotating[sample_index, :, :, 2*eigen_index] = \
                        temp_real
                    generators_rotating[
                        sample_index, :, :, 2*eigen_index + 1] = -temp_imag

                for eigen_index in range(singles.shape[0]):
                    generators_rotating[
                        sample_index, :, :, 2*doubles.shape[0] + eigen_index
                    ] *= singles_forward[sample_index, eigen_index]

            # density_operator_initial_flat = \
            #     inv_vectors_real@density_operator_initial_flat

        # Declare VRAM
        if use_cuda:
            if verbose:
                print("Declare VRAM")

            # Time
            if verbose:
                print("  Declare time VRAM")
            time_device = nc.device_array(
                number_of_samples, dtype=datatype)

            # Gauss-Legendre quadrature definition
            if verbose:
                print("  Declare Magnus VRAM")
                print("    Quadrature times")
            sample_quadrature_device = nc.to_device(sample_quadrature)

            # Storage for Gauss-Legendre quadrature points
            if verbose:
                print("    Quadrature times expanded")
            time_sample_device = nc.device_array(
                number_of_samples*sample_quadrature_device.size,
                dtype=datatype)

            # Weights for commutator-free integrator
            if verbose:
                print("    Quadrature weights")
            weights_device = nc.to_device(weights)

            if verbose:
                print("  Declare superoperator VRAM")
            # Storage for coefficients of superoperators of Lindbladian
            coefficients_device = nc.device_array(
                (time_sample_device.size, generators.shape[0]),
                dtype=datatype
            )
            weighted_coefficients_device = nc.device_array(
                (weights_device.shape[0]*number_of_samples,
                 generators.shape[0]),
                dtype=datatype
            )

            # Basis for the Lindbladian
            # print(generators.shape)
            if verbose:
                print("  Declare basis VRAM")
            if use_rotating:
                # print(doubles)
                # print(singles)
                # print(doubles*time_step/number_of_fine_divisions)
                # print(singles*time_step/number_of_fine_divisions)
                if doubles_size > 0:
                    doubles_forward_device = nc.to_device(
                        doubles_forward[-1, :, :])
                else:
                    doubles_forward_device = 0
                # print(doubles_forward[-1, :, :])
                if singles_size > 0:
                    singles_forward_device = nc.to_device(
                        singles_forward[-1, :])
                else:
                    singles_forward_device = 0
                # print(singles_forward[-1, :])
                generators_device = nc.to_device(generators_rotating)
            else:
                generators_device = nc.to_device(generators)

            # Storage for individual exponentials of the commutator-free
            # integrator
            if verbose:
                print("  Declare superoperator VRAM")
            superoperators_device = nc.device_array(
                (
                    weighted_coefficients_device.shape[0],
                    operator_size, operator_size
                ), dtype=datatype)

            scratch_device = nc.device_array(
                (
                    weighted_coefficients_device.shape[0],
                    operator_size_density, operator_size_density
                ), dtype=datatype
            )

            # Storage for time evolution superoperators
            if verbose:
                print("  Declare time evolution VRAM")
            time_evolution_device = nc.device_array(
                (
                    time_device.shape[0],
                    operator_size_density, operator_size_density
                ), dtype=datatype
            )

            # Initial density operator
            if verbose:
                print("  Move initial state to GPU")
            density_operator_initial_device = nc.to_device(
                  density_operator_initial_flat)

            if use_rotating:
                # Diagonalisation
                if verbose:
                    print("  Move diagonalisation to GPU")
                vectors_real_device = nc.to_device(vectors_real)
                inv_vectors_real_device = nc.to_device(inv_vectors_real)

            if use_kernel:
                if verbose:
                    print("  Move kernel projection to GPU")
                image_projection_device = nc.to_device(image_projection)
                image_projection_transpose_device = nc.to_device(
                    image_projection.T)
                kernel_projection = -image_projection@image_projection.T
                kernel_projection += np.eye(kernel_projection.shape[0])
                kernel_projection_device = nc.to_device(kernel_projection/2)

            if use_unitary:
                # Diagonalisation
                if verbose:
                    print("  Move elimination to GPU")
                elimination_device = nc.to_device(elimination)
                duplication_device = nc.to_device(duplication)

            # Storage for evaluated density operators
            if verbose:
                print("  Declare density operator VRAM")
            density_operators_device = nc.device_array(
                (time_evolution_device.shape[0], operator_size_density),
                dtype=datatype)

            if verbose:
                print("Finished declaring VRAM")
        else:
            if verbose:
                print("Declare RAM")

            # Time
            if verbose:
                print("  Declare time RAM")
            time_device = np.empty(
                number_of_samples, dtype=datatype)

            # Gauss-Legendre quadrature definition
            if verbose:
                print("  Declare Magnus RAM")
                print("    Quadrature times")
            sample_quadrature_device = sample_quadrature

            # Storage for Gauss-Legendre quadrature points
            if verbose:
                print("    Quadrature times expanded")
            time_sample_device = np.empty(
                number_of_samples*sample_quadrature_device.size,
                dtype=datatype)

            # Weights for commutator-free integrator
            if verbose:
                print("    Quadrature weights")
            weights_device = weights

            if verbose:
                print("  Declare superoperator RAM")
            # Storage for coefficients of superoperators of Lindbladian
            coefficients_device = np.empty(
                (time_sample_device.size, generators.shape[0]),
                dtype=datatype
            )
            weighted_coefficients_device = np.empty(
                (
                    weights_device.shape[0]*number_of_samples,
                    generators.shape[0]
                ),
                dtype=datatype
            )

            # Basis for the Lindbladian
            # print(generators.shape)
            if verbose:
                print("  Declare basis RAM")
            if use_rotating:
                # print(doubles)
                # print(singles)
                # print(doubles*time_step/number_of_fine_divisions)
                # print(singles*time_step/number_of_fine_divisions)
                doubles_forward_device = (
                    doubles_forward[-1, :, :])
                # print(doubles_forward[-1, :, :])
                singles_forward_device = (
                    singles_forward[-1, :])
                # print(singles_forward[-1, :])
                generators_device = generators_rotating
            else:
                generators_device = generators

            # Storage for individual exponentials of the commutator-free
            # integrator
            if verbose:
                print("  Declare superoperator RAM")
            superoperators_device = np.empty(
                (
                    weighted_coefficients_device.shape[0],
                    operator_size, operator_size
                ),
                dtype=datatype
            )

            scratch_device = np.empty(
                (
                    weighted_coefficients_device.shape[0],
                    operator_size_density, operator_size_density
                ), dtype=datatype
            )

            # Storage for time evolution superoperators
            if verbose:
                print("  Declare time evolution RAM")
            time_evolution_device = np.empty(
                (
                    time_device.shape[0], operator_size_density,
                    operator_size_density
                ), dtype=datatype
            )

            # Initial density operator
            if verbose:
                print("  Move initial state to CPU")
            density_operator_initial_device = (
                  density_operator_initial_flat)

            if use_rotating:
                # Diagonalisation
                if verbose:
                    print("  Move diagonalisation to CPU")
                vectors_real_device = vectors_real
                inv_vectors_real_device = inv_vectors_real

            if use_kernel:
                if verbose:
                    print("  Move kernel projection to CPU")
                image_projection_device = image_projection
                image_projection_transpose_device = image_projection.T
                kernel_projection = -image_projection@image_projection.T
                kernel_projection += np.eye(kernel_projection.shape[0])
                kernel_projection_device = kernel_projection/2

            if use_unitary:
                # Diagonalisation
                if verbose:
                    print("  Move elimination to CPU")
                elimination_device = elimination
                duplication_device = duplication

            # Storage for evaluated density operators
            if verbose:
                print("  Declare density operator RAM")
            density_operators_device = np.empty(
                (time_evolution_device.shape[0], operator_size_density),
                dtype=datatype)

            if verbose:
                print("Finished declaring RAM")

        # Calculate time
        if verbose:
            print("Starting simulation loop")
        _calculate_time_basic_run(time_device, time_start, time_step)

        # Initialise time evolution operators to identities
        _id_superoperator_run(time_evolution_device)

        # Start loop of fine time stepping
        for repeat_index in range(number_of_fine_divisions):
            # Shift forward by fine time steps
            time_offset = repeat_index*time_step/number_of_fine_divisions

            # Calculate particular Gauss-Legendre sample points
            _calculate_time_quadrature_run(
                time_device, time_sample_device, time_start + time_offset,
                time_step, sample_quadrature_device)

            # print(_calculate_time_quadrature_loop.inspect_asm(_calculate_time_quadrature_loop.signatures[0]))
            # input("?")

            # Sample coefficients from user function
            # print(coefficients_device)
            sample_run(time_sample_device, coefficients_device)

            if use_rotating:
                _calculate_differential_rotating_run(
                    time_step/number_of_fine_divisions, generators_device,
                    coefficients_device, weights_device, superoperators_device)
            else:
                # Apply weighting to coefficients for commutator-free
                # integrator
                _combine_coefficients_run(
                    coefficients_device, weighted_coefficients_device,
                    weights_device)
                # print(weighted_coefficients_device)
                # print("")

                # Scale generators by time step and reduction for
                # exponentiation
                _calculate_differential_run(
                    time_step/number_of_fine_divisions, generators_device,
                    weighted_coefficients_device, superoperators_device
                )

            # Put Lindbladian superoperator in matrix form
            _scale_differential_basic_run(superoperators_device)
            # print(superoperators_device)

            # Repeatedly square (1 +) Lindbladian superoperator for
            # exponentiation
            _repeated_quartic_superoperator_run(
                superoperators_device, scratch_device)

            # Apply resultant time evolution superoperator to previous
            # calculation
            _quadrature_combine_run(
                superoperators_device, time_evolution_device,
                scratch_device[:time_evolution_device.shape[0], :, :]
                # scratch_device
            )

            if use_rotating:
                _apply_eig_run(
                    time_evolution_device,
                    scratch_device[:time_evolution_device.shape[0], :, :],
                    doubles_forward_device,
                    singles_forward_device
                )

        if verbose:
            print("Finished simulation loop")

        # Accumulate time evolution across all time steps
        if verbose:
            print("Combining time evolution steps")

        # print(time_evolution_device.shape, scratch_device[0, :, :].shape)
        _basic_combine_run(time_evolution_device, scratch_device[0, :, :])

        if use_rotating:
            if verbose:
                print("Moving out of the rotating frame")
            _apply_global_sandwich_run(
                vectors_real_device, inv_vectors_real_device,
                time_evolution_device,
                scratch_device[:time_evolution_device.shape[0], :, :]
            )

        if use_kernel:
            if verbose:
                print("Moving out of the equivalence class")
            _apply_global_sandwich_run(
                image_projection_device, image_projection_transpose_device,
                time_evolution_device,
                scratch_device[:time_evolution_device.shape[0], :, :]
            )
            _apply_global_addition_run(
                kernel_projection_device, time_evolution_device,
                scratch_device[:time_evolution_device.shape[0], :, :]
            )

        if use_unitary:
            if verbose:
                print("Moving from operator to superoperator form")
            _apply_global_sandwich_run(
                elimination_device, duplication_device, time_evolution_device,
                scratch_device[:time_evolution_device.shape[0], :, :]
            )

        # Apply time evolution superoperators to initial condition
        if verbose:
            print("Applying time evolution to initial state")
        _apply_time_evolution_run(
            time_evolution_device,
            density_operator_initial_device,
            density_operators_device
        )

        # Retrieve results from GPU
        if use_cuda:
            if verbose:
                print("Retrieving solution from GPU")
            time = time_device.copy_to_host()
            # time_evolution = time_evolution_device.copy_to_host()
            # print(time_evolution)
            density_operators_flat = density_operators_device.copy_to_host()
            # print(density_operators_flat)
        else:
            if verbose:
                print("Retrieving solution from CPU")
            time = time_device
            density_operators_flat = density_operators_device

        # if use_rotating:
        #     if verbose:
        #         print(
        #             "Moving out of generalised rotating frame diagonal basis"
        #         )
        #     density_operators_flat = \
        #         (vectors_real@density_operators_flat.reshape((
        #             density_operators_flat.shape[0],
        #             density_operators_flat.shape[1],
        #             1
        #         ))).reshape(density_operators_flat.shape)

        # if use_kernel:
        #     if verbose:
        #         print("Moving out of equivalence class")
        #     density_operators_flat = np.matvec(
        #         image_projection, density_operators_flat)
        #     density_operators_flat += \
        #         density_operator_initial_flat_kernel

        # Unflatten density operators
        if verbose:
            print("Unflatten density operators")
        density_operators = np.zeros(
            (density_operators_flat.shape[0],
             wavefunction_size, wavefunction_size, 2),
            dtype=datatype
        )
        for operator_index in range(vectorisation_map.shape[0]):
            y_index = vectorisation_map[operator_index, 0]
            x_index = vectorisation_map[operator_index, 1]
            c_index = vectorisation_map[operator_index, 2]

            density_operators[:, y_index, x_index, c_index] = \
                density_operators_flat[:, operator_index]

            # If we are dealing with a coherence, also add to other
            # off-diagonal
            if y_index != x_index:
                if c_index:
                    # Imaginary part
                    density_operators[:, x_index, y_index, c_index] = \
                        -density_operators_flat[:, operator_index]
                else:
                    # Real part
                    density_operators[:, x_index, y_index, c_index] = \
                        density_operators_flat[:, operator_index]

        # Make complex
        density_operators = \
            density_operators[:, :, :, 0] + 1j*density_operators[:, :, :, 1]

        # Remove numba warnings
        warnings.simplefilter(
            "default", nb.core.errors.NumbaPerformanceWarning)

        if verbose:
            print("Finished simulation")

        return time, density_operators

    return simulate
