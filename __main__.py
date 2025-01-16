# https://doi.org/10.1016/j.apnum.2005.11.004

from util import colour_complex_matrix as _colour_complex_matrix
from generators import superoperators as superperator_basis_dict

from quadratures import samples as samples_dict
from quadratures import weights as weights_dict

import math
import numpy as np
import numba as nb
import numba.cuda as nc

import scipy.linalg as sl

import matplotlib.pyplot as plt
from PIL import Image as pli
from cmcrameri import cm


meta_use_cuda = True
meta_datatype = np.float64
meta_use_residual = True
meta_wavefunction_size = 7
meta_operator_size = meta_wavefunction_size**2
meta_number_of_quartic_repeats = 23
meta_scaling_for_quartics: meta_datatype = 4.0**meta_number_of_quartic_repeats
meta_number_of_exponentials = 1
meta_use_cayley = True

if meta_number_of_exponentials == 1:
    sample_quadrature = samples_dict["1_gl"]
    weights = weights_dict["1_1_gl"]
elif meta_number_of_exponentials == 2:
    sample_quadrature = samples_dict["2_gl"]
    weights = weights_dict["4_2_gl"]
elif meta_number_of_exponentials == 3:
    sample_quadrature = samples_dict["2_gl"]
    weights = weights_dict["4_3_gl"]
elif meta_number_of_exponentials == 5:
    sample_quadrature = samples_dict["3_gl"]
    weights = weights_dict["6_5_gl"]
elif meta_number_of_exponentials == 6:
    sample_quadrature = samples_dict["3_gl"]
    weights = weights_dict["6_6_gl"]

if meta_use_cuda:
    _synchronise_block = nc.syncthreads
    _fma = nc.fma
else:
    def _synchronise_block():
        return 1

    def _fma(x, y, z):
        return x*y + z


def _calculate_time(time, time_index, time_start, time_step):
    time[time_index] = time_start + time_step*(time_index + 1)


def _calculate_time_quadrature(
        time, time_index, time_start, time_step, sample):
    time_step_start = time_start + time_step*time_index
    for sample_index in range(sample.size):
        time[time_index*sample.size + sample_index] = \
            time_step_start + time_step*sample[sample_index]


if meta_use_cuda:
    _calculate_time = nc.jit(_calculate_time, device=True)
    _calculate_time_quadrature = nc.jit(
        _calculate_time_quadrature, device=True)

    def _calculate_time_basic_kernel(time, time_sample, time_start, time_step):
        time_index = nc.blockDim.x*nc.blockIdx.x + nc.threadIdx.x
        if time_index < time.size:
            _calculate_time(time, time_index, time_start, time_step)
            _calculate_time(time_sample, time_index, time_start, time_step)

    _calculate_time_basic_kernel = nc.jit(_calculate_time_basic_kernel)

    def _calculate_time_quadrature_kernel(
            time, time_sample, time_start, time_step, sample):
        time_index = nc.blockDim.x*nc.blockIdx.x + nc.threadIdx.x
        if time_index < time.size:
            _calculate_time(time, time_index, time_start, time_step)
            _calculate_time_quadrature(
                time_sample, time_index, time_start, time_step, sample)

    _calculate_time_quadrature_kernel = nc.jit(
        _calculate_time_quadrature_kernel)


def _calculate_time_basic_run(time, time_sample, time_start, time_step):
    if meta_use_cuda:
        grid_size = (int(math.ceil(time.size/32)), 1)
        block_size = (32, 1)
        _calculate_time_basic_kernel[grid_size, block_size] \
            (time, time_sample, time_start, time_step)


def _calculate_time_quadrature_run(time, time_sample, time_start, time_step, sample):
    if meta_use_cuda:
        grid_size = (int(math.ceil(time.size/32)), 1)
        block_size = (32, 1)
        _calculate_time_quadrature_kernel[grid_size, block_size] \
            (time, time_sample, time_start, time_step, sample)


def _generate_sampler(sampler):
    if meta_use_cuda:
        sampler_device = nc.jit(sampler, device=True)

        def sample_kernel(times, coefficients):
            time_index = nc.blockDim.x*nc.blockIdx.x + nc.threadIdx.x
            if time_index < times.size:
                for generator_index in range(coefficients.shape[1]):
                    coefficients[time_index, generator_index] = 0.0

                sampler_device(times[time_index], coefficients[time_index, :])

        sample_kernel = nc.jit(sample_kernel)

    def sample_run(times, coefficients):
        if meta_use_cuda:
            grid_size = (int(math.ceil(times.size/32)), 1)
            block_size = (32, 1)
            sample_kernel[grid_size, block_size](times, coefficients)

    return sample_run


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


if meta_use_cuda:
    _combine_coefficients = nc.jit(_combine_coefficients, device=True)

    def _combine_coefficients_kernel(
            coefficients, weighted_coefficients, weights):
        if nc.threadIdx.x < weighted_coefficients.shape[1] \
                and nc.threadIdx.y < weights.shape[0]:
            _combine_coefficients(
                coefficients[nc.blockIdx.x*weights.shape[1]:
                             (nc.blockIdx.x + 1)*weights.shape[1], :],
                weighted_coefficients[nc.blockIdx.x*weights.shape[0]:
                                      (nc.blockIdx.x + 1)*weights.shape[0], :],
                weights,
                nc.threadIdx.y,
                nc.threadIdx.x
            )

    _combine_coefficients_kernel = nc.jit(_combine_coefficients_kernel)


def _combine_coefficients_run(coefficients, weighted_coefficients, weights):
    if meta_use_cuda:
        grid_size = (weighted_coefficients.shape[0]//weights.shape[0], 1)
        block_size = (weighted_coefficients.shape[1], weights.shape[0])
        _combine_coefficients_kernel[grid_size, block_size] \
            (coefficients, weighted_coefficients, weights)


def _calculate_differential(
        time_step, generator, coefficient, differential, y_index, x_index):
    differential_real: meta_datatype = 0.0
    differential_imag: meta_datatype = 0.0

    for generator_index in range(generator.shape[0]):
        differential_real = nc.fma(
            coefficient[generator_index],
            generator[generator_index, y_index, x_index, 0],
            differential_real
        )
        differential_imag = nc.fma(
            coefficient[generator_index],
            generator[generator_index, y_index, x_index, 1],
            differential_imag
        )

    differential[y_index, x_index, 0] = time_step*differential_real
    differential[y_index, x_index, 1] = time_step*differential_imag


if meta_use_cuda:
    _calculate_differential = nc.jit(_calculate_differential, device=True)

    def _calculate_differential_kernel(
            time_step, generator, coefficient, differential):
        if nc.threadIdx.y < meta_operator_size \
                and nc.threadIdx.x < meta_wavefunction_size:
            for x_index_stride in range(meta_wavefunction_size):
                _calculate_differential(
                    time_step,
                    generator,
                    coefficient[nc.blockIdx.x, :],
                    differential[nc.blockIdx.x, :, :, :],
                    nc.threadIdx.y,
                    nc.threadIdx.x + x_index_stride*meta_wavefunction_size
                )

    _calculate_differential_kernel = nc.jit(_calculate_differential_kernel)


def _calculate_differential_run(
        time_step, generator, coefficient, differential):
    if meta_use_cuda:
        grid_size = (coefficient.shape[0], 1)
        block_size = (meta_wavefunction_size, meta_operator_size)
        _calculate_differential_kernel[grid_size, block_size] \
            (time_step, generator, coefficient, differential)


def _scale_differential_basic(differential, y_index, x_index):
    if meta_use_cayley:
        differential[y_index, x_index, 0] /= \
            2*(4**meta_number_of_quartic_repeats)
        differential[y_index, x_index, 1] /= \
            2*(4**meta_number_of_quartic_repeats)
    else:
        differential[y_index, x_index, 0] /= 4**meta_number_of_quartic_repeats
        differential[y_index, x_index, 1] /= 4**meta_number_of_quartic_repeats


if meta_use_cuda:
    _scale_differential_basic = nc.jit(_scale_differential_basic, device=True)

    def _scale_differential_basic_kernel(differential):
        if nc.threadIdx.y < meta_operator_size \
                and nc.threadIdx.x < meta_wavefunction_size:
            for x_index_stride in range(meta_wavefunction_size):
                _scale_differential_basic(
                    differential[nc.blockIdx.x, :, :, :],
                    nc.threadIdx.y,
                    nc.threadIdx.x + x_index_stride*meta_wavefunction_size
                )

    _scale_differential_basic_kernel = nc.jit(_scale_differential_basic_kernel)


def _scale_differential_basic_run(differential):
    if meta_use_cuda:
        grid_size = (differential.shape[0], 1)
        block_size = (meta_wavefunction_size, meta_operator_size)
        _scale_differential_basic_kernel[grid_size, block_size](differential)


def _negate_superoperator(positive, negative, y_index, x_index):
    negative[y_index, x_index, 0] = -positive[y_index, x_index, 0]
    negative[y_index, x_index, 1] = -positive[y_index, x_index, 1]


if meta_use_cuda:
    _negate_superoperator = nc.jit(_negate_superoperator, device=True)

    def _calculate_cayley_kernel(differential):
        if nc.threadIdx.y < meta_operator_size \
                and nc.threadIdx.x < meta_wavefunction_size:
            scratch = nc.shared.array(
                (meta_operator_size, meta_operator_size, 2),
                dtype=meta_datatype
            )

            for x_index_stride in range(meta_wavefunction_size):
                _negate_superoperator(
                    differential[nc.blockIdx.x, :, :, :],
                    scratch,
                    nc.threadIdx.y,
                    nc.threadIdx.x + x_index_stride*meta_wavefunction_size
                )
            nc.syncthreads()

            diff = differential[nc.blockIdx.x, :, :, :]

            for node_index in range(meta_operator_size):
                # Scale row
                if nc.threadIdx.x == 1:

                    node_real = scratch[node_index, node_index, 0]
                    node_imag = scratch[node_index, node_index, 1]
                    div_real = (1 + node_real) \
                        / ((1 + node_real)**2 + node_imag**2)
                    div_imag = -node_imag/((1 + node_real)**2 + node_imag**2)
                    nc.syncthreads()

                    eval_real = \
                        div_real*scratch[node_index, nc.threadIdx.y, 0] \
                        - div_imag*scratch[node_index, nc.threadIdx.y, 1]
                    eval_imag = \
                        div_real*scratch[node_index, nc.threadIdx.y, 1] \
                        + div_imag*scratch[node_index, nc.threadIdx.y, 0]
                    if nc.threadIdx.y == node_index:
                        eval_real -= node_real*div_real - node_imag*div_imag
                        eval_imag -= node_imag*div_real + node_real*div_imag
                    scratch[node_index, nc.threadIdx.y, 0] = eval_real
                    scratch[node_index, nc.threadIdx.y, 1] = eval_imag

                    eval_real = \
                        div_real*diff[node_index, nc.threadIdx.y, 0] \
                        - div_imag*diff[node_index, nc.threadIdx.y, 1]
                    eval_imag = \
                        div_real*diff[node_index, nc.threadIdx.y, 1] \
                        + div_imag*diff[node_index, nc.threadIdx.y, 0]
                    if nc.threadIdx.y == node_index:
                        eval_real -= node_real*div_real - node_imag*div_imag
                        eval_imag -= node_imag*div_real + node_real*div_imag
                    diff[node_index, nc.threadIdx.y, 0] = eval_real
                    diff[node_index, nc.threadIdx.y, 1] = eval_imag

                nc.syncthreads()

                # Eliminate rows
                for x_index_stride in range(meta_wavefunction_size):
                    x_index = nc.threadIdx.x \
                        + x_index_stride*meta_wavefunction_size

                    if x_index != node_index:
                        scale_real = -scratch[x_index, node_index, 0]
                        scale_imag = -scratch[x_index, node_index, 1]
                        nc.syncthreads()

                        eval_real = scale_real \
                            * scratch[x_index, nc.threadIdx.y, 0] \
                            - scale_imag*scratch[x_index, nc.threadIdx.y, 1]
                        eval_imag = scale_real \
                            * scratch[x_index, nc.threadIdx.y, 1] \
                            + scale_imag*scratch[x_index, nc.threadIdx.y, 0]
                        if nc.threadIdx.y == node_index:
                            eval_real += scale_real
                            eval_imag += scale_imag
                        nc.syncthreads()

                        scratch[x_index, nc.threadIdx.y, 0] += eval_real
                        scratch[x_index, nc.threadIdx.y, 1] += eval_imag
                        nc.syncthreads()

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

                        diff[x_index, nc.threadIdx.y, 0] += eval_real
                        diff[x_index, nc.threadIdx.y, 1] += eval_imag
                        nc.syncthreads()

    _calculate_cayley_kernel = nc.jit(_calculate_cayley_kernel)


def _calculate_cayley_run(differential):
    if meta_use_cuda:
        grid_size = (differential.shape[0], 1)
        block_size = (meta_wavefunction_size, meta_operator_size)
        _calculate_cayley_kernel[grid_size, block_size](differential)


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

    if meta_use_residual:
        out_scratch_real: meta_datatype = 2*inp[y_index, x_index, 0]
        out_scratch_imag: meta_datatype = 2*inp[y_index, x_index, 1]
    else:
        out_scratch_real: meta_datatype = 0.0
        out_scratch_imag: meta_datatype = 0.0

    for trace_index in range(meta_operator_size):
        # TODO: unroll?
        out_scratch_real = nc.fma(
                inp[y_index, trace_index, 0],
                inp[trace_index, x_index, 0],
                out_scratch_real
        )
        out_scratch_real = nc.fma(
                inp[y_index, trace_index, 1],
                -inp[trace_index, x_index, 1],
                out_scratch_real
        )
        out_scratch_imag = nc.fma(
                inp[y_index, trace_index, 0],
                inp[trace_index, x_index, 1],
                out_scratch_imag
        )
        out_scratch_imag = nc.fma(
                inp[y_index, trace_index, 1],
                inp[trace_index, x_index, 0],
                out_scratch_imag
        )

    out[y_index, x_index, 0] = out_scratch_real
    out[y_index, x_index, 1] = out_scratch_imag


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

    if meta_use_residual:
        out_scratch_real = \
            left[y_index, x_index, 0]
        out_scratch_imag = \
            left[y_index, x_index, 1]
        out_scratch_real += \
            right[y_index, x_index, 0]
        out_scratch_imag += \
            right[y_index, x_index, 1]

    for trace_index in range(meta_operator_size):
        out_scratch_real = nc.fma(
                left[y_index, trace_index, 0],
                right[trace_index, x_index, 0],
                out_scratch_real
        )
        out_scratch_real = nc.fma(
                left[y_index, trace_index, 1],
                -right[trace_index, x_index, 1],
                out_scratch_real
        )

        out_scratch_imag = nc.fma(
                left[y_index, trace_index, 0],
                right[trace_index, x_index, 1],
                out_scratch_imag
        )
        out_scratch_imag = nc.fma(
                left[y_index, trace_index, 1],
                right[trace_index, x_index, 0],
                out_scratch_imag
        )
    out[y_index, x_index, 0] = out_scratch_real
    out[y_index, x_index, 1] = out_scratch_imag


def _repeated_quartic_superoperator(
        superoperator, scratch, y_index, x_index_reduced):
    for _ in range(meta_number_of_quartic_repeats):
        for x_index_stride in range(meta_wavefunction_size):
            _square_superoperator(
                superoperator, scratch, y_index,
                x_index_reduced + x_index_stride*meta_wavefunction_size
            )
        nc.syncthreads()
        for x_index_stride in range(meta_wavefunction_size):
            _square_superoperator(
                scratch, superoperator, y_index,
                x_index_reduced + x_index_stride*meta_wavefunction_size
            )
        nc.syncthreads()


if meta_use_cuda:
    # Compile squaring
    _square_superoperator = nc.jit(_square_superoperator, device=True)
    _repeated_quartic_superoperator = nc.jit(
        _repeated_quartic_superoperator, device=True
    )

    # Wrap in kernel
    def _repeated_quartic_superoperator_kernel(superoperators):
        superoperator = superoperators[nc.blockIdx.x, :, :, :]
        scratch = nc.shared.array(
            (meta_operator_size, meta_operator_size, 2), meta_datatype
        )
        if nc.threadIdx.x < meta_wavefunction_size and \
                nc.threadIdx.y < meta_operator_size:
            _repeated_quartic_superoperator(
                superoperator, scratch, nc.threadIdx.y, nc.threadIdx.x
            )
    _repeated_quartic_superoperator_kernel = nc.jit(
        _repeated_quartic_superoperator_kernel
    )


nb.jit(nopython=True, parallel=True)
def _repeated_quartic_superoperator_run(superoperators):
    """
    Effort:

    Parallel: N^3
    Series: 4 N K
    """

    if meta_use_cuda:
        number_of_blocks = (superoperators.shape[0], 1)
        block_shape = (meta_wavefunction_size, meta_operator_size)
        _repeated_quartic_superoperator_kernel[number_of_blocks, block_shape] \
            (superoperators)

    else:
        for block_index in nb.prange(superoperators.shape[0]):
            superoperator = superoperators[block_index, :, :, :]
            scratch = np.empty(
                (meta_operator_size, meta_operator_size, 2), meta_datatype
            )
            for y_index in nb.prange(meta_operator_size):
                for x_index in nb.prange(meta_wavefunction_size):
                    _repeated_quartic_superoperator(
                        superoperator, scratch, y_index, x_index
                    )


def _copy_superoperator(original, clone, y_index, x_index):
    clone[y_index, x_index, 0] = original[y_index, x_index, 0]
    clone[y_index, x_index, 1] = original[y_index, x_index, 1]


if meta_use_cuda:
    _multiply_superoperator = nc.jit(_multiply_superoperator, device=True)
    _copy_superoperator = nc.jit(_copy_superoperator, device=True)

    def _quadrature_combine_kernel(superoperators, time_evolutions):
        if nc.threadIdx.x < meta_wavefunction_size \
                and nc.threadIdx.y < meta_operator_size:
            if meta_number_of_exponentials == 1:
                for x_index_stride in range(meta_wavefunction_size):
                    _copy_superoperator(
                        superoperators[nc.blockIdx.x, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )

            elif meta_number_of_exponentials == 2:
                for x_index_stride in range(meta_wavefunction_size):
                    _multiply_superoperator(
                        superoperators[2*nc.blockIdx.x + 1, :, :, :],
                        superoperators[2*nc.blockIdx.x, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )

            elif meta_number_of_exponentials == 3:
                for x_index_stride in range(meta_wavefunction_size):
                    _multiply_superoperator(
                        superoperators[3*nc.blockIdx.x + 1, :, :, :],
                        superoperators[3*nc.blockIdx.x, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )
                nc.syncthreads()

                for x_index_stride in range(meta_wavefunction_size):
                    _multiply_superoperator(
                        superoperators[3*nc.blockIdx.x + 2, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        superoperators[3*nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )
                nc.syncthreads()

                for x_index_stride in range(meta_wavefunction_size):
                    _copy_superoperator(
                        superoperators[3*nc.blockIdx.x, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )
                nc.syncthreads()

            # elif meta_number_of_exponentials == 5:
            #     for x_index_stride in range(meta_wavefunction_size):
            #         _multiply_superoperator(
            #             superoperators[5*nc.blockIdx.x + 1, :, :, :],
            #             superoperators[5*nc.blockIdx.x, :, :, :],
            #             time_evolutions[nc.blockIdx.x, :, :, :],
            #             nc.threadIdx.y,
            #             nc.threadIdx.x + meta_wavefunction_size*x_index_stride
            #         )
            #     nc.syncthreads()

            #     for x_index_stride in range(meta_wavefunction_size):
            #         _multiply_superoperator(
            #             superoperators[5*nc.blockIdx.x + 4, :, :, :],
            #             superoperators[5*nc.blockIdx.x + 3, :, :, :],
            #             superoperators[5*nc.blockIdx.x + 1, :, :, :],
            #             nc.threadIdx.y,
            #             nc.threadIdx.x + meta_wavefunction_size*x_index_stride
            #         )
            #     nc.syncthreads()

            #     for x_index_stride in range(meta_wavefunction_size):
            #         _multiply_superoperator(
            #             superoperators[5*nc.blockIdx.x + 2, :, :, :],
            #             time_evolutions[nc.blockIdx.x, :, :, :],
            #             superoperators[5*nc.blockIdx.x, :, :, :],
            #             nc.threadIdx.y,
            #             nc.threadIdx.x + meta_wavefunction_size*x_index_stride
            #         )
            #     nc.syncthreads()

            #     for x_index_stride in range(meta_wavefunction_size):
            #         _multiply_superoperator(
            #             superoperators[5*nc.blockIdx.x + 1, :, :, :],
            #             superoperators[5*nc.blockIdx.x, :, :, :],
            #             time_evolutions[nc.blockIdx.x, :, :, :],
            #             nc.threadIdx.y,
            #             nc.threadIdx.x + meta_wavefunction_size*x_index_stride
            #         )
            #     nc.syncthreads()

            elif meta_number_of_exponentials == 5:
                for x_index_stride in range(meta_wavefunction_size):
                    _multiply_superoperator(
                        superoperators[5*nc.blockIdx.x + 1, :, :, :],
                        superoperators[5*nc.blockIdx.x, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )
                nc.syncthreads()

                for x_index_stride in range(meta_wavefunction_size):
                    _multiply_superoperator(
                        superoperators[5*nc.blockIdx.x + 2, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        superoperators[5*nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )
                nc.syncthreads()

                for x_index_stride in range(meta_wavefunction_size):
                    _multiply_superoperator(
                        superoperators[5*nc.blockIdx.x + 3, :, :, :],
                        superoperators[5*nc.blockIdx.x, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )
                nc.syncthreads()

                for x_index_stride in range(meta_wavefunction_size):
                    _multiply_superoperator(
                        superoperators[5*nc.blockIdx.x + 4, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        superoperators[5*nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )
                nc.syncthreads()

                for x_index_stride in range(meta_wavefunction_size):
                    _copy_superoperator(
                        superoperators[5*nc.blockIdx.x, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )
                nc.syncthreads()

            elif meta_number_of_exponentials == 6:
                for x_index_stride in range(meta_wavefunction_size):
                    _multiply_superoperator(
                        superoperators[6*nc.blockIdx.x + 1, :, :, :],
                        superoperators[6*nc.blockIdx.x, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )
                nc.syncthreads()

                for x_index_stride in range(meta_wavefunction_size):
                    _multiply_superoperator(
                        superoperators[6*nc.blockIdx.x + 2, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        superoperators[6*nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )
                nc.syncthreads()

                for x_index_stride in range(meta_wavefunction_size):
                    _multiply_superoperator(
                        superoperators[6*nc.blockIdx.x + 3, :, :, :],
                        superoperators[6*nc.blockIdx.x, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )
                nc.syncthreads()

                for x_index_stride in range(meta_wavefunction_size):
                    _multiply_superoperator(
                        superoperators[6*nc.blockIdx.x + 4, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        superoperators[6*nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )
                nc.syncthreads()

                for x_index_stride in range(meta_wavefunction_size):
                    _multiply_superoperator(
                        superoperators[6*nc.blockIdx.x + 5, :, :, :],
                        superoperators[6*nc.blockIdx.x, :, :, :],
                        time_evolutions[nc.blockIdx.x, :, :, :],
                        nc.threadIdx.y,
                        nc.threadIdx.x + meta_wavefunction_size*x_index_stride
                    )
                nc.syncthreads()

    _quadrature_combine_kernel = nc.jit(_quadrature_combine_kernel)

    def _basic_combine_kernel(time_evolutions, time_index):
        scratch = nc.shared.array(
            (meta_operator_size, meta_operator_size, 2), dtype=meta_datatype
        )

        if nc.threadIdx.y < meta_operator_size \
                and nc.threadIdx.x < meta_wavefunction_size:
            for x_index_stride in range(meta_wavefunction_size):
                _multiply_superoperator(
                    time_evolutions[time_index + 1, :, :, :],
                    time_evolutions[time_index, :, :, :],
                    scratch,
                    nc.threadIdx.y,
                    nc.threadIdx.x + x_index_stride*meta_wavefunction_size
                )
            nc.syncthreads()

            for x_index_stride in range(meta_wavefunction_size):
                _copy_superoperator(
                    scratch,
                    time_evolutions[time_index + 1, :, :, :],
                    nc.threadIdx.y,
                    nc.threadIdx.x + x_index_stride*meta_wavefunction_size
                )
            nc.syncthreads()

    _basic_combine_kernel = nc.jit(_basic_combine_kernel)


def _quadrature_combine_run(exponentials, time_evolution):
    if meta_use_cuda:
        grid_size = (time_evolution.shape[0], 1)
        block_size = (meta_wavefunction_size, meta_operator_size)
        _quadrature_combine_kernel[grid_size, block_size] \
            (exponentials, time_evolution)


nb.jit(nopython=True, parallel=False)
def _basic_combine_run(time_evolutions):
    if meta_use_cuda:
        block_size = (meta_wavefunction_size, meta_operator_size)
        for time_index in range(0, time_evolutions.shape[0] - 1):
            _basic_combine_kernel[(1, 1), block_size] \
                (time_evolutions, time_index)


def _telescope_combine(time_evolution):
    pass


def _multiply_superoperator_operator(superoperator, operator, out, index):
    out[index, 0] = operator[index, 0]
    out[index, 1] = operator[index, 1]
    for trace_index in range(meta_operator_size):
        out[index, 0] = nc.fma(
            superoperator[index, trace_index, 0],
            operator[trace_index, 0],
            out[index, 0]
        )

        out[index, 0] = nc.fma(
            superoperator[index, trace_index, 1],
            -operator[trace_index, 1],
            out[index, 0]
        )

        out[index, 1] = nc.fma(
            superoperator[index, trace_index, 0],
            operator[trace_index, 1],
            out[index, 1]
        )

        out[index, 1] = nc.fma(
            superoperator[index, trace_index, 1],
            operator[trace_index, 0],
            out[index, 1]
        )


if meta_use_cuda:
    _multiply_superoperator_operator = nc.jit(
        _multiply_superoperator_operator,
        device=True
    )

    def _apply_time_evolution_kernel(time_evolutions, density_operator_initial,
                                     denisty_operators):
        if nc.threadIdx.x < meta_operator_size:
            _multiply_superoperator_operator(
                time_evolutions[nc.blockIdx.x, :, :, :],
                density_operator_initial,
                denisty_operators[nc.blockIdx.x, :, :],
                nc.threadIdx.x
            )

    _apply_time_evolution_kernel = nc.jit(_apply_time_evolution_kernel)


def _apply_time_evolution_run(time_evolutions,
                              density_operator_initial, denisty_operators):
    if meta_use_cuda:
        grid_size = (time_evolutions.shape[0], 1)
        block_size = (meta_operator_size, 1)
        _apply_time_evolution_kernel[grid_size, block_size] \
            (time_evolutions, density_operator_initial, denisty_operators)


def test_generators():
    print("Testing generators...")
    superoperators = np.array(
        list(superperator_basis_dict.values()), dtype=meta_datatype
    )

    superoperators *= 100

    if meta_use_cuda:
        superoperators_device = nc.to_device(
            superoperators/(2**(2*meta_number_of_quartic_repeats))
        )
    else:
        superoperators_device = superoperators \
            / (2**(2*meta_number_of_quartic_repeats))

    _repeated_quartic_superoperator_run(superoperators_device)
    if meta_use_cuda:
        transforms = superoperators_device.copy_to_host()

    transforms_true = transforms.copy()
    transforms_true[:, :, :, 0] += np.eye(meta_operator_size)

    # Visualise
    plt.figure()
    for transform_index in range(transforms_true.shape[0]):
        plt.subplot(3, 5, transform_index + 1)
        plt.imshow(_colour_complex_matrix(
            transforms_true[transform_index, :, :, :])
        )
        plt.title(list(superperator_basis_dict.keys())[transform_index])
        plt.axis("off")

    print("Done!")


def test_squaring():
    print("Testing squaring")

    number_of_superoperators = int(1e2)
    superoperators = np.random.normal(
        size=(
            number_of_superoperators,
            meta_operator_size,
            meta_operator_size,
            2
        ),
    )
    superoperators = np.array(superoperators, dtype=meta_datatype)
    superoperators /= 1e6*meta_wavefunction_size

    superoperators_visualise = _colour_complex_matrix(
        superoperators[0, :, :, :]
    )

    plt.figure()
    plt.imshow(superoperators_visualise)
    plt.title("Before")
    plt.draw()

    if meta_use_cuda:
        superoperators_device = nc.to_device(
            superoperators/(2**(2*meta_number_of_quartic_repeats))
        )
    else:
        superoperators_device = superoperators \
            / (2**(2*meta_number_of_quartic_repeats))

    _repeated_quartic_superoperator_run(superoperators_device)
    if meta_use_cuda:
        transforms = superoperators_device.copy_to_host()

    transforms_true = transforms.copy()
    transforms_true[:, :, :, 0] += np.eye(meta_operator_size)

    # Comparison with scipy.linalg
    do_compare_sl = True
    if do_compare_sl:
        superoperator_true_complex = superoperators[0, :, :, 0] \
            + 1j*superoperators[0, :, :, 1]
        transform_true_complex_sl = sl.expm(superoperator_true_complex)
        transform_true_sl = np.empty(
            (meta_operator_size, meta_operator_size, 2), dtype=meta_datatype
        )
        transform_true_sl[:, :, 0] = np.real(transform_true_complex_sl)
        transform_true_sl[:, :, 1] = np.imag(transform_true_complex_sl)

        transform_true_error_sl = transforms_true[0, :, :, :] \
            - transform_true_sl

    # Visualise
    transform_true_visualise = _colour_complex_matrix(
        transforms_true[0, :, :, :]
    )
    if do_compare_sl:
        transform_true_sl_visualise = _colour_complex_matrix(
            transform_true_sl[:, :, :]
        )
        transform_true_error_sl_visualise = _colour_complex_matrix(
            transform_true_error_sl[:, :, :]
        )

        transform_true_error_sl_f = np.sqrt(
            np.sum(transform_true_error_sl**2)
        )/(meta_operator_size**2)
        print(f"Error (Frobenius): {transform_true_error_sl_f}")

    plt.figure()
    if do_compare_sl:
        plt.subplot(1, 3, 1)
    plt.imshow(transform_true_visualise)
    plt.title("After")

    if do_compare_sl:
        plt.subplot(1, 3, 2)
        plt.imshow(transform_true_sl_visualise)
        plt.title("Scipy\nground truth")

        plt.subplot(1, 3, 3)
        plt.imshow(transform_true_error_sl_visualise)
        plt.title("Error")
        plt.draw()

    print("Done!")


def test_combination():
    print("Testing basic combination...")
    generators = np.array(
        list(superperator_basis_dict.values()), dtype=meta_datatype
    )

    superoperators = np.empty(
        (60, meta_operator_size, meta_operator_size, 2),
        dtype=meta_datatype
    )
    superoperators[:, :, :, :] = (math.tau/60)*(
        2*generators[0, :, :, :] + 2*generators[2, :, :, :]
        + 5*generators[5, :, :, :] + 5*generators[6, :, :, :]
        + 10*generators[3, :, :, :] + 10*generators[7, :, :, :]
        + generators[8, :, :, :] + generators[10, :, :, :]
        + 2*generators[9, :, :, :]
        + 2*generators[13, :, :, :]
    )

    density_operator_initial = np.zeros(
        (meta_wavefunction_size, meta_wavefunction_size, 2),
        dtype=meta_datatype
    )
    density_operator_initial[0, 0, 0] = 1/3
    density_operator_initial[1, 1, 0] = 1/3
    density_operator_initial[2, 2, 0] = 1/3

    density_operator_initial_flat = density_operator_initial.reshape(
        (meta_operator_size, 2))

    if meta_use_cuda:
        superoperators_device = nc.to_device(
            superoperators/(2**(2*meta_number_of_quartic_repeats)))
        density_operator_initial_device = nc.to_device(
                              density_operator_initial_flat)
        density_operators_device = nc.device_array(
            (superoperators_device.shape[0],
             meta_operator_size, 2),
            dtype=meta_datatype)
    else:
        superoperators_device = superoperators \
            / (2**(2*meta_number_of_quartic_repeats))

    _repeated_quartic_superoperator_run(superoperators_device)

    _basic_combine_run(superoperators_device)

    _apply_time_evolution_run(superoperators_device,
                              density_operator_initial_device,
                              density_operators_device)

    if meta_use_cuda:
        transforms = superoperators_device.copy_to_host()
        density_operators = density_operators_device.copy_to_host()

    density_operators = density_operators.reshape((density_operators.shape[0],
                                                   meta_wavefunction_size,
                                                   meta_wavefunction_size, 2))

    visualise_time_evolution(density_operators, transforms)

    print("Done!")


def plot_populations(time, density_operators):
    print("Plotting populations...")
    state_labels = ["+g", "0g", "-g", "+e", "0e", "-e", "s"]
    plt.figure()
    for state_index in range(density_operators.shape[1]):
        plt.plot(
            time,
            100*density_operators[:, state_index, state_index, 0],
            "-",
            color=cm.hawaii(state_index/meta_wavefunction_size),
            label=state_labels[state_index]
        )
    plt.xlabel("Time")
    plt.ylabel("Population (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.draw()
    print("Done!")


def visualise_time_evolution(density_operators, time_evolution):
    print("Creating animations...")

    frames = []

    # plt.figure()
    for density_operator_index in range(density_operators.shape[0]):
        # plt.subplot(6, 10, density_operator_index + 1)
        coloured = _colour_complex_matrix(
            density_operators[density_operator_index, :, :, :])
        # plt.imshow(coloured)
        # plt.axis("off")

        scale = 20
        frame = np.empty(
            (scale*meta_wavefunction_size, scale*meta_wavefunction_size, 3),
            dtype=np.uint8
        )
        for x_index in range(scale):
            for y_index in range(scale):
                frame[y_index::scale, x_index::scale] = coloured*255
                progress = int(frame.shape[1]*(density_operator_index + 1)/density_operators.shape[0])
                frame[-4:, :progress] = np.array([255, 255, 255])
        frame = pli.fromarray(frame)
        frames.append(frame)

    frames[0].save(
        "density_operator.gif",
        save_all=True,
        append_images=frames[1:],
        duration=1e3/density_operators.shape[0],
        loop=0
    )

    if time_evolution is None:
        return

    transforms_true = time_evolution.copy()
    transforms_true[:, :, :, 0] += np.eye(meta_operator_size)

    frames = []

    # plt.figure()
    for transform_index in range(transforms_true.shape[0]):
        # plt.subplot(6, 10, transform_index + 1)
        coloured = _colour_complex_matrix(
            transforms_true[transform_index, :, :, :])
        # plt.imshow(coloured)
        # plt.axis("off")

        scale = 10
        frame = np.empty(
            (scale*meta_operator_size, scale*meta_operator_size, 3),
            dtype=np.uint8
        )
        for x_index in range(scale):
            for y_index in range(scale):
                frame[y_index::scale, x_index::scale] = coloured*255
        frame = pli.fromarray(frame)
        frames.append(frame)

    frames[0].save(
        "time_evolution.gif",
        save_all=True,
        append_images=frames[1:],
        duration=1e3/15,
        loop=0
    )


def test_time_sample():
    print("Testing time sampling...")

    number_of_samples = 512
    if meta_use_cuda:
        time_device = nc.device_array(number_of_samples, dtype=meta_datatype)
        time_sample_device = nc.device_array(
            number_of_samples, dtype=meta_datatype)

    time_start: meta_datatype = 0.0
    time_step: meta_datatype = 1/512
    _calculate_time_basic_run(
        time_device, time_sample_device, time_start, time_step)

    if meta_use_cuda:
        time = time_device.copy_to_host()
        time_device = None

    # Sample coefficients
    generators = np.array(
        list(superperator_basis_dict.values()), dtype=meta_datatype
    )

    decay_amp = 50

    pulse_amp = 100
    pulse_time_0 = 0.1

    def sampler(time, coefficient):
        coefficient[8] = decay_amp
        coefficient[9] = 1.5*decay_amp
        coefficient[10] = decay_amp/3
        coefficient[11] = decay_amp/2

        coefficient[3] = 80

        if time < 0.1:
            pass
        elif time < 0.15:
            coefficient[13] = pulse_amp
        elif time < 0.2:
            pass
        elif time < 0.5:
            coefficient[0] = 20
        elif time < 0.85:
            pass
        elif time < 0.9:
            coefficient[13] = pulse_amp
        else:
            pass

    sample_run = _generate_sampler(sampler)

    if meta_use_cuda:
        coefficients_device = nc.device_array(
            (time_sample_device.size, generators.shape[0]),
            dtype=meta_datatype
        )

    sample_run(time_sample_device, coefficients_device)

    # Scale generators
    if meta_use_cuda:
        generators_device = nc.to_device(generators)
        superoperators_device = nc.device_array(
            (
                time_sample_device.shape[0], meta_operator_size,
                meta_operator_size, 2
            ), dtype=meta_datatype)

    _calculate_differential_run(
        time_step, generators_device,
        coefficients_device, superoperators_device
    )

    if meta_use_cuda:
        # coefficients = coefficients_device.copy_to_host()
        # print(coefficients)
        coefficients_device = None
        generators_device = None

    # Exponentiate
    _scale_differential_basic_run(superoperators_device)
    _repeated_quartic_superoperator_run(superoperators_device)

    # Integrate
    _basic_combine_run(superoperators_device)

    # Apply
    density_operator_initial = np.zeros(
        (meta_wavefunction_size, meta_wavefunction_size, 2),
        dtype=meta_datatype
    )
    density_operator_initial[0, 0, 0] = 1/3
    density_operator_initial[1, 1, 0] = 1/3
    density_operator_initial[2, 2, 0] = 1/3

    density_operator_initial_flat = density_operator_initial.reshape(
        (meta_operator_size, 2))

    if meta_use_cuda:
        density_operator_initial_device = nc.to_device(
                              density_operator_initial_flat)
        density_operators_device = nc.device_array(
            (superoperators_device.shape[0],
             meta_operator_size, 2),
            dtype=meta_datatype)

    _apply_time_evolution_run(
        superoperators_device,
        density_operator_initial_device,
        density_operators_device
    )

    if meta_use_cuda:
        time_evolution = superoperators_device.copy_to_host()
        superoperators_device = None
        density_operators = density_operators_device.copy_to_host()
        density_operators_device = None

    density_operators = density_operators.reshape((density_operators.shape[0],
                                                   meta_wavefunction_size,
                                                   meta_wavefunction_size, 2))

    visualise_time_evolution(density_operators, time_evolution)

    print("Done")


def test_time_sample_quadrature():
    print("Testing time sampling...")

    number_of_samples = 2
    if meta_use_cuda:
        sample_quadrature_device = nc.to_device(sample_quadrature)
        time_device = nc.device_array(number_of_samples, dtype=meta_datatype)
        time_sample_device = nc.device_array(
            number_of_samples*sample_quadrature_device.size,
            dtype=meta_datatype)

    time_start: meta_datatype = 0.0
    time_step: meta_datatype = 1/number_of_samples
    _calculate_time_quadrature_run(
        time_device, time_sample_device, time_start,
        time_step, sample_quadrature_device)

    if meta_use_cuda:
        time = time_device.copy_to_host()
        # time_sample = time_sample_device.copy_to_host()
        # print(time)
        # print(time_sample)
        time_device = None

    # Sample coefficients
    generators = np.array(
        list(superperator_basis_dict.values()), dtype=meta_datatype
    )

    resonance = 20
    rabi = 2

    def sampler(time, coefficient):
        # coefficient[0] = math.tau*rabi
        coefficient[0] = 2*math.tau*rabi*math.cos(math.tau*resonance*time)
        # coefficient[1] = math.tau*rabi*math.sin(math.tau*resonance*time)
        coefficient[2] = math.tau*resonance

    sample_run = _generate_sampler(sampler)

    if meta_use_cuda:
        coefficients_device = nc.device_array(
            (time_sample_device.size, generators.shape[0]),
            dtype=meta_datatype
        )

    sample_run(time_sample_device, coefficients_device)

    # Calculate weights for quadrature
    if meta_use_cuda:
        weights_device = nc.to_device(weights)
        weighted_coefficients_device = nc.device_array(
            (weights_device.shape[0]*number_of_samples, generators.shape[0]),
            dtype=meta_datatype
        )

    _combine_coefficients_run(
        coefficients_device, weighted_coefficients_device, weights_device)

    if meta_use_cuda:
        coefficients = coefficients_device.copy_to_host()
        weighted_coefficients = weighted_coefficients_device.copy_to_host()
        print(coefficients[:, 1])
        print(np.mean(weighted_coefficients[:, 1]))

    # Scale generators
    if meta_use_cuda:
        generators_device = nc.to_device(generators)
        superoperators_device = nc.device_array(
            (
                weighted_coefficients_device.shape[0], meta_operator_size,
                meta_operator_size, 2
            ), dtype=meta_datatype)

    _calculate_differential_run(
        time_step, generators_device,
        weighted_coefficients_device, superoperators_device
    )

    if meta_use_cuda:
        # coefficients = coefficients_device.copy_to_host()
        # print(coefficients)
        coefficients_device = None
        weighted_coefficients_device = None
        generators_device = None

    # Exponentiate
    _scale_differential_basic_run(superoperators_device)

    if meta_use_cayley:
        _calculate_cayley_run(superoperators_device)

    _repeated_quartic_superoperator_run(superoperators_device)

    # Evaluate quadrature
    if meta_use_cuda:
        time_evolution_device = nc.device_array(
            (time.shape[0], meta_operator_size, meta_operator_size, 2),
            dtype=meta_datatype
        )

    _quadrature_combine_run(superoperators_device, time_evolution_device)

    # Integrate
    _basic_combine_run(time_evolution_device)

    # Apply
    density_operator_initial = np.zeros(
        (meta_wavefunction_size, meta_wavefunction_size, 2),
        dtype=meta_datatype
    )
    density_operator_initial[0, 0, 0] = 1
    density_operator_initial[1, 1, 0] = 0
    density_operator_initial[2, 2, 0] = 0

    density_operator_initial_flat = density_operator_initial.reshape(
        (meta_operator_size, 2))

    if meta_use_cuda:
        density_operator_initial_device = nc.to_device(
                              density_operator_initial_flat)
        density_operators_device = nc.device_array(
            (time_evolution_device.shape[0],
             meta_operator_size, 2),
            dtype=meta_datatype)

    _apply_time_evolution_run(
        time_evolution_device,
        density_operator_initial_device,
        density_operators_device
    )

    if meta_use_cuda:
        time_evolution = time_evolution_device.copy_to_host()
        time_evolution_device = None
        density_operators = density_operators_device.copy_to_host()
        density_operators_device = None

    density_operators = density_operators.reshape((density_operators.shape[0],
                                                   meta_wavefunction_size,
                                                   meta_wavefunction_size, 2))

    # visualise_time_evolution(density_operators[::8, :, :, :], None)
    # plot_populations(time, density_operators)

    print("Done")


if __name__ == "__main__":
    # test_generators()
    # test_squaring()
    # test_combination()
    # test_time_sample()
    test_time_sample_quadrature()
    plt.show()
