from superspinsim.generators import superoperators as superoperator_basis_dict_nv
from superspinsim.generators import operators as operator_basis_dict_nv
from superspinsim.generators import vectorisation_map as vectorisation_map_nv

from superspinsim.quadratures import samples as samples_dict
from superspinsim.quadratures import weights as weights_dict

import superspinsim.nv

import math
import numpy as np

import numba as nb
import numba.cuda as nc

import warnings


def generate_simulator(
        sampler,
        generators=None,
        vectorisation_map=None,
        use_cuda=True,
        datatype=np.float64,
        use_residual=True,
        number_of_quartic_repeats=35,  # 35,
        number_of_exponentials=2,
        number_of_fine_divisions=1,
        use_cayley=False
        ):

    scaling_for_quartics: datatype = \
        4.0**number_of_quartic_repeats

    if generators is None:
        generators = np.array(list(superoperator_basis_dict_nv.values()),
                              dtype=datatype)
        vectorisation_map = vectorisation_map_nv
        wavefunction_size = list(operator_basis_dict_nv.values())[0].shape[0]

    generators = generators.astype(datatype)
    wavefunction_size = np.max(vectorisation_map[:, 0]) + 1
    operator_size = generators.shape[1]

    # A matrix of size 32*32 or larger cannot be have its entries allocated a
    # unique thread each, because that is too much for cuda.
    sqrt_block_size_max = 32
    operator_size_block = min(operator_size, sqrt_block_size_max)
    operator_stride_max = int(math.ceil(operator_size/operator_size_block))
    while operator_size_block*operator_size > 1024:
        operator_stride_max += 1
        operator_size_block = int(math.ceil(operator_size/operator_stride_max))

    if number_of_exponentials == 1:
        sample_quadrature = np.array(samples_dict["1_gl"], dtype=datatype)
        weights = np.array(weights_dict["2_1_gl"], dtype=datatype)
    elif number_of_exponentials == 2:
        sample_quadrature = np.array(samples_dict["2_gl"], dtype=datatype)
        weights = np.array(weights_dict["4_2_gl"], dtype=datatype)
    elif number_of_exponentials == 3:
        sample_quadrature = np.array(samples_dict["2_gl"], dtype=datatype)
        weights = np.array(weights_dict["4_3_gl"], dtype=datatype)
    elif number_of_exponentials == 5:
        sample_quadrature = np.array(samples_dict["3_gl"], dtype=datatype)
        weights = np.array(weights_dict["6_5_gl"], dtype=datatype)
    elif number_of_exponentials == 6:
        sample_quadrature = np.array(samples_dict["3_gl"], dtype=datatype)
        weights = np.array(weights_dict["6_6_gl"], dtype=datatype)

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

    def _calculate_time_basic_run(time, time_start, time_step):
        if use_cuda:
            grid_size = (int(math.ceil(time.size/32)), 1)
            block_size = (32, 1)
            _calculate_time_basic_kernel[grid_size, block_size] \
                (time, time_start, time_step)

    def _calculate_time_quadrature_run(
            time, time_sample, time_start, time_step, sample):
        if use_cuda:
            grid_size = (int(math.ceil(time.size/32)), 1)
            block_size = (32, 1)
            _calculate_time_quadrature_kernel[grid_size, block_size] \
                (time, time_sample, time_start, time_step, sample)

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

        def sample_run(times, coefficients):
            if use_cuda:
                grid_size = (int(math.ceil(times.size/32)), 1)
                block_size = (32, 1)
                sample_kernel[grid_size, block_size](times, coefficients)

        return sample_run

    # Make sampler GPU compatible
    sample_run = _generate_sampler(sampler)

    # Quadrature --------------------------------------------------------------

    def _combine_coefficients(
            coefficient, weighted_coefficient, weight,
            exponential_index, coefficient_index):
        scratch = 0
        for trace_index in range(weight.shape[1]):
            scratch = nc.fma(
                weight[exponential_index, trace_index],
                coefficient[trace_index, coefficient_index],
                scratch
            )
        weighted_coefficient[exponential_index, coefficient_index] = scratch

    if use_cuda:
        _combine_coefficients = nc.jit(_combine_coefficients, device=True)

        def _combine_coefficients_kernel(
                coefficients, weighted_coefficients, weights):
            if nc.threadIdx.x < weighted_coefficients.shape[1] \
                    and nc.threadIdx.y < weights.shape[0]:
                _combine_coefficients(
                    coefficients[nc.blockIdx.x*weights.shape[1]:
                                 (nc.blockIdx.x + 1)*weights.shape[1], :],
                    weighted_coefficients[
                        nc.blockIdx.x*weights.shape[0]:
                        (nc.blockIdx.x + 1)*weights.shape[0], :],
                    weights,
                    nc.threadIdx.y,
                    nc.threadIdx.x
                )

        _combine_coefficients_kernel = nc.jit(_combine_coefficients_kernel)

    def _combine_coefficients_run(
            coefficients, weighted_coefficients, weights):
        if use_cuda:
            grid_size = (weighted_coefficients.shape[0]//weights.shape[0], 1)
            block_size = (weighted_coefficients.shape[1], weights.shape[0])
            _combine_coefficients_kernel[grid_size, block_size] \
                (coefficients, weighted_coefficients, weights)

    # Matrix form -------------------------------------------------------------

    def _calculate_differential(
            time_step, generator, coefficient, differential, y_index, x_index):
        scratch: datatype = 0.0

        for generator_index in range(generator.shape[0]):
            scratch = nc.fma(
                coefficient[generator_index],
                generator[generator_index, y_index, x_index],
                scratch
            )

        differential[y_index, x_index] = time_step*scratch

    if use_cuda:
        _calculate_differential = nc.jit(_calculate_differential, device=True)

        def _calculate_differential_kernel(
                time_step, generator, coefficient, differential):
            if nc.threadIdx.y < operator_size:
                for x_index_stride in range(operator_stride_max):
                    x_index_use = \
                        nc.threadIdx.x + x_index_stride*operator_size_block
                    if x_index_use < operator_size:
                        _calculate_differential(
                            time_step,
                            generator,
                            coefficient[nc.blockIdx.x, :],
                            differential[nc.blockIdx.x, :, :],
                            nc.threadIdx.y,
                            x_index_use
                        )

        _calculate_differential_kernel = nc.jit(_calculate_differential_kernel)

    def _calculate_differential_run(
            time_step, generator, coefficient, differential):
        if use_cuda:
            grid_size = (coefficient.shape[0], 1)
            block_size = (operator_size_block, operator_size)
            _calculate_differential_kernel[grid_size, block_size] \
                (time_step, generator, coefficient, differential)

    def _scale_differential_basic(differential, y_index, x_index):
        if use_cayley:
            differential[y_index, x_index] /= 2*scaling_for_quartics
        else:
            differential[y_index, x_index] /= scaling_for_quartics

    if use_cuda:
        _scale_differential_basic = nc.jit(
            _scale_differential_basic, device=True)

        def _scale_differential_basic_kernel(differential):
            if nc.threadIdx.y < operator_size:
                for x_index_stride in range(operator_stride_max):
                    x_index_use = \
                        nc.threadIdx.x + x_index_stride*operator_size_block
                    if x_index_use < operator_size:
                        _scale_differential_basic(
                            differential[nc.blockIdx.x, :, :],
                            nc.threadIdx.y,
                            x_index_use
                        )

        _scale_differential_basic_kernel = nc.jit(
            _scale_differential_basic_kernel)

    def _scale_differential_basic_run(differential):
        if use_cuda:
            grid_size = (differential.shape[0], 1)
            block_size = (operator_size_block, operator_size)
            _scale_differential_basic_kernel[grid_size, block_size] \
                (differential)

    # Cayley ------------------------------------------------------------------

    def _negate_superoperator(positive, negative, y_index, x_index):
        negative[y_index, x_index] = -positive[y_index, x_index]

    if use_cuda:
        _negate_superoperator = nc.jit(_negate_superoperator, device=True)

        def _calculate_cayley_kernel(differential):
            if nc.threadIdx.y < operator_size \
                    and nc.threadIdx.x < wavefunction_size:
                scratch = nc.shared.array(
                    (operator_size, operator_size, 2),
                    dtype=datatype
                )

                for x_index_stride in range(wavefunction_size):
                    _negate_superoperator(
                        differential[nc.blockIdx.x, :, :, :],
                        scratch,
                        nc.threadIdx.y,
                        nc.threadIdx.x + x_index_stride*wavefunction_size
                    )
                nc.syncthreads()

                diff = differential[nc.blockIdx.x, :, :, :]

                for node_index in range(operator_size):
                    # Scale row
                    if nc.threadIdx.x == 1:

                        node_real = scratch[node_index, node_index, 0]
                        node_imag = scratch[node_index, node_index, 1]
                        div_real = (1 + node_real) \
                            / ((1 + node_real)**2 + node_imag**2)
                        div_imag = -node_imag \
                            / ((1 + node_real)**2 + node_imag**2)
                    nc.syncthreads()

                    if nc.threadIdx.x == 1:
                        eval_real = \
                            div_real*scratch[node_index, nc.threadIdx.y, 0] \
                            - div_imag*scratch[node_index, nc.threadIdx.y, 1]
                        eval_imag = \
                            div_real*scratch[node_index, nc.threadIdx.y, 1] \
                            + div_imag*scratch[node_index, nc.threadIdx.y, 0]
                        if nc.threadIdx.y == node_index:
                            eval_real -= \
                                node_real*div_real - node_imag*div_imag
                            eval_imag -= \
                                node_imag*div_real + node_real*div_imag
                        scratch[node_index, nc.threadIdx.y, 0] = eval_real
                        scratch[node_index, nc.threadIdx.y, 1] = eval_imag

                        eval_real = \
                            div_real*diff[node_index, nc.threadIdx.y, 0] \
                            - div_imag*diff[node_index, nc.threadIdx.y, 1]
                        eval_imag = \
                            div_real*diff[node_index, nc.threadIdx.y, 1] \
                            + div_imag*diff[node_index, nc.threadIdx.y, 0]
                        if nc.threadIdx.y == node_index:
                            eval_real -= \
                                node_real*div_real - node_imag*div_imag
                            eval_imag -= \
                                node_imag*div_real + node_real*div_imag
                        diff[node_index, nc.threadIdx.y, 0] = eval_real
                        diff[node_index, nc.threadIdx.y, 1] = eval_imag

                    nc.syncthreads()

                    # Eliminate rows
                    for x_index_stride in range(wavefunction_size):
                        x_index = nc.threadIdx.x \
                            + x_index_stride*wavefunction_size

                        if x_index != node_index:
                            scale_real = -scratch[x_index, node_index, 0]
                            scale_imag = -scratch[x_index, node_index, 1]
                        nc.syncthreads()

                        if x_index != node_index:
                            eval_real = scale_real \
                                * scratch[x_index, nc.threadIdx.y, 0] \
                                - scale_imag \
                                * scratch[x_index, nc.threadIdx.y, 1]
                            eval_imag = scale_real \
                                * scratch[x_index, nc.threadIdx.y, 1] \
                                + scale_imag \
                                * scratch[x_index, nc.threadIdx.y, 0]
                            if nc.threadIdx.y == node_index:
                                eval_real += scale_real
                                eval_imag += scale_imag
                        nc.syncthreads()

                        if x_index != node_index:
                            scratch[x_index, nc.threadIdx.y, 0] += eval_real
                            scratch[x_index, nc.threadIdx.y, 1] += eval_imag
                        nc.syncthreads()

                        if x_index != node_index:
                            eval_real = scale_real \
                                * diff[x_index, nc.threadIdx.y, 0] \
                                - scale_imag*diff[x_index, nc.threadIdx.y, 1]
                            eval_imag = scale_real \
                                * diff[x_index, nc.threadIdx.y, 1] \
                                + scale_imag*diff[x_index, nc.threadIdx.y, 0]
                            if nc.threadIdx.y == node_index:
                                eval_real += scale_real
                                eval_imag += scale_imag
                        nc.syncthreads()

                        if x_index != node_index:
                            diff[x_index, nc.threadIdx.y, 0] += eval_real
                            diff[x_index, nc.threadIdx.y, 1] += eval_imag
                        nc.syncthreads()

        _calculate_cayley_kernel = nc.jit(_calculate_cayley_kernel)

    def _calculate_cayley_run(differential):
        if use_cuda:
            grid_size = (differential.shape[0], 1)
            block_size = (wavefunction_size, operator_size)
            _calculate_cayley_kernel[grid_size, block_size](differential)

    # Repeated squaring -------------------------------------------------------

    def _square_superoperator(inp, out, y_index, x_index):
        r"""
        Calculates,

        $\Re(s) = \Re(a)^2 - \Im(a)^2 + 2\,\Re(a),$
        $\Re(s) = \Re(a)\Im(a) + \Im(a)\Re(a) + 2\,\Im(a).$

        This is equivalent to calculating the complex

        $S = A^2$

        where

        $S = 1 + s,$
        $A = 1 + a.$
        """

        if use_residual:
            out_scratch: datatype = 2*inp[y_index, x_index]
        else:
            out_scratch: datatype = 0.0

        for trace_index in range(operator_size):
            # TODO: unroll?
            out_scratch = nc.fma(
                    inp[y_index, trace_index],
                    inp[trace_index, x_index],
                    out_scratch
            )

        out[y_index, x_index] = out_scratch

    def _multiply_superoperator(
            left, right, out, y_index, x_index):
        r"""
        Calculates,

        $\Re(c) = \Re(a)\Re(b) - \Im(a)\Im(b) + \Re(a) + \Re(b),$
        $\Im(c) = \Re(a)\Im(b) + \Im(a)\Re(b) + \Im(a) + \Im(b).$

        This is equivalent to calculating the complex

        $C = A\,B$

        where

        $C = 1 + c,$
        $A = 1 + a,$
        $B = 1 + b.$
        """

        if use_residual:
            out_scratch: datatype = \
                left[y_index, x_index] + right[y_index, x_index]
        else:
            out_scratch: datatype = 0.0

        for trace_index in range(operator_size):
            out_scratch = nc.fma(
                    left[y_index, trace_index],
                    right[trace_index, x_index],
                    out_scratch
            )
        out[y_index, x_index] = out_scratch

    if use_cuda:
        # Compile squaring
        _square_superoperator = nc.jit(_square_superoperator, device=True)

        # Wrap in kernel
        def _repeated_quartic_superoperator_kernel(superoperators):
            superoperator = superoperators[nc.blockIdx.x, :, :]
            scratch = nc.shared.array(
                (operator_size, operator_size), datatype
            )

            for _ in range(number_of_quartic_repeats):
                if nc.threadIdx.y < operator_size:
                    for x_index_stride in range(operator_stride_max):
                        x_index_use = \
                            nc.threadIdx.x + x_index_stride*operator_size_block
                        if x_index_use < operator_size:
                            _square_superoperator(
                                superoperator, scratch,
                                nc.threadIdx.y, x_index_use
                            )
                nc.syncthreads()

                if nc.threadIdx.y < operator_size:
                    for x_index_stride in range(operator_stride_max):
                        x_index_use = \
                            nc.threadIdx.x + x_index_stride*operator_size_block
                        if x_index_use < operator_size:
                            _square_superoperator(
                                scratch, superoperator,
                                nc.threadIdx.y, x_index_use
                            )
                nc.syncthreads()

        _repeated_quartic_superoperator_kernel = nc.jit(
            _repeated_quartic_superoperator_kernel
        )

    def _repeated_quartic_superoperator_run(superoperators):
        """
        Effort:

        Parallel: N^3
        Series: 4 N K
        """

        if use_cuda:
            number_of_blocks = (superoperators.shape[0], 1)
            block_shape = (operator_size_block, operator_size)
            _repeated_quartic_superoperator_kernel[
                number_of_blocks, block_shape](superoperators)

    # Combine samples at different quadrature nodes ---------------------------

    def _copy_superoperator(original, clone, y_index, x_index):
        clone[y_index, x_index] = original[y_index, x_index]

    if use_cuda:
        _multiply_superoperator = nc.jit(_multiply_superoperator, device=True)
        _copy_superoperator = nc.jit(_copy_superoperator, device=True)

        def _quadrature_combine_kernel(superoperators, time_evolutions):
            if nc.threadIdx.y < operator_size:
                scratch = nc.shared.array(
                    (operator_size, operator_size),
                    dtype=datatype
                )

                for exponential_index in range(
                        0, number_of_exponentials, 2):
                    for x_index_stride in range(operator_stride_max):
                        x_index_use = \
                            nc.threadIdx.x + x_index_stride*operator_size_block
                        if x_index_use < operator_size:
                            _multiply_superoperator(
                                superoperators[
                                    # number_of_exponentials*nc.blockIdx.x
                                    # + exponential_index,
                                    number_of_exponentials*(nc.blockIdx.x + 1)
                                    - exponential_index - 1,
                                    :, :],
                                time_evolutions[nc.blockIdx.x, :, :],
                                scratch,
                                nc.threadIdx.y,
                                x_index_use
                            )
                    nc.syncthreads()

                    if exponential_index + 1 < number_of_exponentials:
                        for x_index_stride in range(operator_stride_max):
                            x_index_use = \
                                nc.threadIdx.x + x_index_stride*operator_size_block
                            if x_index_use < operator_size:
                                _multiply_superoperator(
                                    superoperators[
                                        # number_of_exponentials*nc.blockIdx.x
                                        # + exponential_index + 1,
                                        number_of_exponentials*(nc.blockIdx.x + 1)
                                        - exponential_index - 2,
                                        :, :],
                                    scratch,
                                    time_evolutions[nc.blockIdx.x, :, :],
                                    nc.threadIdx.y,
                                    x_index_use
                                )
                    else:
                        for x_index_stride in range(operator_stride_max):
                            x_index_use = \
                                nc.threadIdx.x + x_index_stride*operator_size_block
                            if x_index_use < operator_size:
                                _copy_superoperator(
                                    scratch,
                                    time_evolutions[nc.blockIdx.x, :, :],
                                    nc.threadIdx.y,
                                    x_index_use
                                )

                    nc.syncthreads()

        _quadrature_combine_kernel = nc.jit(_quadrature_combine_kernel)

        def _id_superoperator_kernel(time_evolutions):
            if nc.threadIdx.y < operator_size:
                for x_index_stride in range(operator_stride_max):
                    x_index_use = \
                        nc.threadIdx.x + x_index_stride*operator_size_block
                    if x_index_use < operator_size:
                        time_evolutions[
                            nc.blockIdx.x, nc.threadIdx.y, x_index_use] = 0
                        if not use_residual:
                            if x_index_use == nc.threadIdx.y:
                                time_evolutions[
                                    nc.blockIdx.x, nc.threadIdx.y,
                                    x_index_use
                                ] = 1

        _id_superoperator_kernel = nc.jit(_id_superoperator_kernel)

        def _basic_combine_kernel(time_evolutions, time_index):
            scratch = nc.shared.array(
                (operator_size, operator_size),
                dtype=datatype
            )

            if nc.threadIdx.y < operator_size:
                for x_index_stride in range(operator_stride_max):
                    x_index_use = \
                        nc.threadIdx.x + x_index_stride*operator_size_block
                    if x_index_use < operator_size:
                        _multiply_superoperator(
                            time_evolutions[time_index + 1, :, :],
                            time_evolutions[time_index, :, :],
                            scratch,
                            nc.threadIdx.y,
                            x_index_use
                        )
                nc.syncthreads()

                for x_index_stride in range(operator_stride_max):
                    x_index_use = \
                        nc.threadIdx.x + x_index_stride*operator_size_block
                    if x_index_use < operator_size:
                        _copy_superoperator(
                            scratch,
                            time_evolutions[time_index + 1, :, :],
                            nc.threadIdx.y,
                            x_index_use
                        )
                nc.syncthreads()

        _basic_combine_kernel = nc.jit(_basic_combine_kernel)

    def _quadrature_combine_run(exponentials, time_evolution):
        if use_cuda:
            grid_size = (time_evolution.shape[0], 1)
            block_size = (operator_size_block, operator_size)
            _quadrature_combine_kernel[grid_size, block_size] \
                (exponentials, time_evolution)

    def _id_superoperator_run(time_evolution):
        if use_cuda:
            grid_size = (time_evolution.shape[0], 1)
            block_size = (operator_size_block, operator_size)
            _id_superoperator_kernel[grid_size, block_size] \
                (time_evolution)

    def _basic_combine_run(time_evolutions):
        if use_cuda:
            block_size = (operator_size_block, operator_size)
            for time_index in range(0, time_evolutions.shape[0] - 1):
                _basic_combine_kernel[(1, 1), block_size] \
                    (time_evolutions, time_index)

    # Accumulate --------------------------------------------------------------

    def _multiply_superoperator_operator(superoperator, operator, out, index):
        if use_residual:
            scratch = operator[index]
        else:
            scratch: datatype = 0.0

        for trace_index in range(operator_size):
            scratch = nc.fma(
                superoperator[index, trace_index],
                operator[trace_index],
                scratch
            )

        out[index] = scratch

    if use_cuda:
        _multiply_superoperator_operator = nc.jit(
            _multiply_superoperator_operator,
            device=True
        )

        def _apply_time_evolution_kernel(time_evolutions,
                                         density_operator_initial,
                                         density_operators):
            if nc.threadIdx.x < operator_size:
                _multiply_superoperator_operator(
                    time_evolutions[nc.blockIdx.x, :, :],
                    density_operator_initial,
                    density_operators[nc.blockIdx.x, :],
                    nc.threadIdx.x
                )

        _apply_time_evolution_kernel = nc.jit(_apply_time_evolution_kernel)

    def _apply_time_evolution_run(time_evolutions,
                                  density_operator_initial, density_operators):
        if use_cuda:
            grid_size = (time_evolutions.shape[0], 1)
            block_size = (operator_size, 1)
            _apply_time_evolution_kernel[grid_size, block_size] \
                (time_evolutions, density_operator_initial, density_operators)

    # Simulation --------------------------------------------------------------

    def simulate(
            density_operator_initial,
            time_start, time_end, time_step):

        # Remove numba warnings
        warnings.simplefilter("ignore", nb.core.errors.NumbaPerformanceWarning)

        # Time
        number_of_samples = int((time_end - time_start)/time_step)

        # Flatten density operator
        density_operator_initial_flat = \
            np.empty(vectorisation_map.shape[0], dtype=datatype)
        for operator_index in range(vectorisation_map.shape[0]):
            y_index = vectorisation_map[operator_index, 0]
            x_index = vectorisation_map[operator_index, 1]
            c_index = vectorisation_map[operator_index, 2]

            density_operator_initial_flat[operator_index] = \
                density_operator_initial[y_index, x_index, c_index]

        # Declare memory
        if use_cuda:
            # Time
            time_device = nc.device_array(
                number_of_samples, dtype=datatype)

            # Gauss-Legendre quadrature definition
            sample_quadrature_device = nc.to_device(sample_quadrature)

            # Storage for Gauss-Legendre quadrature points
            time_sample_device = nc.device_array(
                number_of_samples*sample_quadrature_device.size,
                dtype=datatype)

            # Weights for commutator-free integrator
            weights_device = nc.to_device(weights)

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
            generators_device = nc.to_device(generators)

            # Storage for individual exponentials of the commutator-free
            # integrator
            superoperators_device = nc.device_array(
                (weighted_coefficients_device.shape[0],
                 operator_size, operator_size),
                dtype=datatype)

            # Storage for time evolution superoperators
            time_evolution_device = nc.device_array(
                (time_device.shape[0], operator_size, operator_size),
                dtype=datatype
            )

            # Initial density operator
            density_operator_initial_device = nc.to_device(
                                  density_operator_initial_flat)

            # Storage for evaluated density operators
            density_operators_device = nc.device_array(
                (time_evolution_device.shape[0], operator_size),
                dtype=datatype)

        # Calculate time
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

            # Sample coefficients from user function
            sample_run(time_sample_device, coefficients_device)

            # Apply weighting to coefficients for commutator-free integrator
            _combine_coefficients_run(
                coefficients_device, weighted_coefficients_device,
                weights_device)

            # Scale generators by time step and reduction for exponentiation
            _calculate_differential_run(
                time_step/number_of_fine_divisions, generators_device,
                weighted_coefficients_device, superoperators_device
            )

            # Put Lindbladian superoperator in matrix form
            _scale_differential_basic_run(superoperators_device)

            # # Apply a Cayley transform (Pade 1,1) to the Lindbladian for
            # # smoother exponentiation
            # if use_cayley:
            #     _calculate_cayley_run(superoperators_device)

            # Repeatedly square (1 +) Lindbladian superoperator for
            # exponentiation
            _repeated_quartic_superoperator_run(superoperators_device)

            # Apply resultant time evolution superoperator to previous
            # calculation
            _quadrature_combine_run(
                superoperators_device, time_evolution_device)

        # Accumulate time evolution across all time steps
        _basic_combine_run(time_evolution_device)

        # Apply time evolution superoperators to initial condition
        _apply_time_evolution_run(
            time_evolution_device,
            density_operator_initial_device,
            density_operators_device
        )

        # Retrieve results from GPU
        if use_cuda:
            time = time_device.copy_to_host()
            # time_evolution = time_evolution_device.copy_to_host()
            # print(time_evolution)
            density_operators_flat = density_operators_device.copy_to_host()
            # print(density_operators_flat)

        # Unflatten density operators
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

        # Remove numba warnings
        warnings.simplefilter(
            "default", nb.core.errors.NumbaPerformanceWarning)

        return time, density_operators

    return simulate
