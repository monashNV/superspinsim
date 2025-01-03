# https://doi.org/10.1016/j.apnum.2005.11.004

import math
import numpy as np
import numba as nb
import numba.cuda as nc

import scipy.linalg as sl

import matplotlib.pyplot as plt
from cmcrameri import cm


meta_use_cuda = True
meta_datatype = np.float64
meta_use_residual = True
meta_wavefunction_size = 7
meta_operator_size = meta_wavefunction_size**2
meta_number_of_quartic_repeats = 12

if meta_use_cuda:
    _synchronise_block = nc.syncthreads
    _fma = nc.fma
else:
    _synchronise_block = lambda: 1
    _fma = math.fma


weights_cf_4_2 = np.array([
        [(3 - 2*math.sqrt(3))/12, (3 + 2*math.sqrt(3))/12],
        [(3 + 2*math.sqrt(3))/12, (3 - 2*math.sqrt(3))/12]
])

nodes_cf_4_2 = np.array(
        [(3 - math.sqrt(3))/6, (3 - math.sqrt(3))/6]
)


def _calculate_time(time, time_sample):
    pass


def _sample(time, coefficient):
    pass


def _calculate_differential(time_step, generator, coefficient, differential):
    pass


def _massage_differential(differential):
    pass


def _square_superoperator(inp, out, column_index, row_index):
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
        out_scratch_real: meta_datatype = 2*inp[column_index, row_index, 0]
        out_scratch_imag: meta_datatype = 2*inp[column_index, row_index, 1]
    else:
        out_scratch_real: meta_datatype = 0.0
        out_scratch_imag: meta_datatype = 0.0

    for trace_index in range(meta_operator_size):
        # TODO: unroll?
        out_scratch_real = nc.fma(
                inp[column_index, trace_index, 0],
                inp[trace_index, row_index, 0],
                out_scratch_real
        )
        out_scratch_real = nc.fma(
                inp[column_index, trace_index, 1],
                -inp[trace_index, row_index, 1],
                out_scratch_real
        )
        out_scratch_imag = nc.fma(
                inp[column_index, trace_index, 0],
                inp[trace_index, row_index, 1],
                out_scratch_imag
        )
        out_scratch_imag = nc.fma(
                inp[column_index, trace_index, 1],
                inp[trace_index, row_index, 0],
                out_scratch_imag
        )

    out[column_index, row_index, 0] = out_scratch_real
    out[column_index, row_index, 1] = out_scratch_imag


def _multiply_superoperator(left, right, out, superoperator_index, column_index, row_index):
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
        out_scratch_real = left[superoperator_index, column_index, row_index, 0]
        out_scratch_imag = left[superoperator_index, column_index, row_index, 1]
        out_scratch_real += right[superoperator_index, column_index, row_index, 0]
        out_scratch_imag += right[superoperator_index, column_index, row_index, 1]

    for trace_index in range(meta_operator_size):
        out_scratch_real = nc.fma(
                left[superoperator_index, column_index, trace_index, 0],
                right[superoperator_index, trace_index, row_index, 0],
                out_scratch_real
        )
        out_scratch_real = nc.fma(
                left[superoperator_index, column_index, trace_index, 1],
                -right[superoperator_index, trace_index, row_index, 1],
                out_scratch_real
        )

        out_scratch_imag = nc.fma(
                left[superoperator_index, column_index, trace_index, 0],
                right[superoperator_index, trace_index, row_index, 1],
                out_scratch_imag
        )
        out_scratch_imag = nc.fma(
                left[superoperator_index, column_index, trace_index, 1],
                right[superoperator_index, trace_index, row_index, 0],
                out_scratch_imag
        )
    out[superoperator_index, column_index, row_index, 0] = out_scratch_real
    out[superoperator_index, column_index, row_index, 1] = out_scratch_imag


def _repeated_quartic_superoperator(superoperator, scratch, column_index, row_index_reduced):
    for _ in range(meta_number_of_quartic_repeats):
        for row_index_stride in range(meta_wavefunction_size):
            _square_superoperator(
                superoperator, scratch, column_index,
                row_index_reduced + row_index_stride*meta_wavefunction_size
            )
        _synchronise_block()
        for row_index_stride in range(meta_wavefunction_size):
            _square_superoperator(
                scratch, superoperator, column_index,
                row_index_reduced + row_index_stride*meta_wavefunction_size
            )
        _synchronise_block()

if meta_use_cuda:
    # Compile squaring
    _square_superoperator = nc.jit(_square_superoperator, device=True)
    _repeated_quartic_superoperator = nc.jit(_repeated_quartic_superoperator, device=True)

    # Wrap in kernel
    def _repeated_quartic_superoperator_kernel(superoperators):
        superoperator = superoperators[nc.blockIdx.x, :, :, :]
        scratch = nc.shared.array((meta_operator_size, meta_operator_size, 2), meta_datatype)
        if nc.threadIdx.x < meta_wavefunction_size and nc.threadIdx.y < meta_operator_size:
            _repeated_quartic_superoperator(superoperator, scratch, nc.threadIdx.y, nc.threadIdx.x)
    _repeated_quartic_superoperator_kernel = nc.jit(_repeated_quartic_superoperator_kernel)


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
        _repeated_quartic_superoperator_kernel[number_of_blocks, block_shape](superoperators)
    
    else:
        for block_index in nb.prange(superoperators.shape[0]):
            superoperator = superoperators[block_index, :, :, :]
            scratch = np.empty((meta_operator_size, meta_operator_size, 2), meta_datatype)
            for column_index in range(meta_operator_size):
                for row_index in range(meta_operator_size):
                    _repeated_quartic_superoperator(
                        superoperator, scratch, column_index, row_index
                    )


def _partial_combine(unitary, new_unitary):
    pass


def _telescope_combine(unitary):
    pass


def _colour_complex_matrix(inp):
    out = np.zeros(
        (meta_operator_size, meta_operator_size, 3),
        dtype=meta_datatype
    )
    out[:, :, 0] += (2/math.sqrt(6))*inp[:, :, 0]
    out[:, :, 1] += (-1/math.sqrt(6))*inp[:, :, 0] \
        + (1/math.sqrt(2))*inp[:, :, 1]
    out[:, :, 2] += (-1/math.sqrt(6))*inp[:, :, 0] \
        + (-1/math.sqrt(2))*inp[:, :, 1]
    # vmax = np.max(np.sqrt(np.sum(out**2, axis = 2)))
    # out /= 2*vmax
    out += 1/math.sqrt(3)
    out = np.clip(out, 0, 1)
    return out
    

if __name__ == "__main__":
    print("Testing squaring")

    number_of_superoperators = int(2e0)
    superoperators = np.random.normal(
        size=(number_of_superoperators, meta_operator_size, meta_operator_size, 2),
    )
    # superoperators = np.ones(
    #     (number_of_superoperators, meta_operator_size, meta_operator_size, 2),
    # )
    superoperators = np.array(superoperators, dtype=meta_datatype)
    superoperators /= 10e6*meta_wavefunction_size

    superoperators_visualise = _colour_complex_matrix(superoperators[0, :, :, :])

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
    superoperator_true_complex = superoperators[0, :, :, 0] \
        + 1j*superoperators[0, :, :, 1]
    transform_true_complex_sl = sl.expm(superoperator_true_complex)
    transform_true_sl = np.empty(
        (meta_operator_size, meta_operator_size, 2), dtype=meta_datatype
    )
    transform_true_sl[:, :, 0] = np.real(transform_true_complex_sl)
    transform_true_sl[:, :, 1] = np.imag(transform_true_complex_sl)

    transform_true_error_sl = transforms_true[0, :, :, :] - transform_true_sl

    # Visualise
    transform_true_visualise = _colour_complex_matrix(
        transforms_true[0, :, :, :]
    )
    transform_true_sl_visualise = _colour_complex_matrix(
        transform_true_sl[:, :, :]
    )
    transform_true_error_sl_visualise = _colour_complex_matrix(
        transform_true_error_sl[:, :, :]
    )

    transform_true_error_sl_f = np.sqrt(np.sum(transform_true_error_sl**2))/(meta_operator_size**2)
    print(f"Error (Frobenius): {transform_true_error_sl_f}")

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(transform_true_visualise)
    plt.title("After")

    plt.subplot(1, 3, 2)
    plt.imshow(transform_true_sl_visualise)
    plt.title("Scipy\nground truth")
    plt.draw()

    plt.subplot(1, 3, 3)
    plt.imshow(transform_true_error_sl_visualise)
    plt.title("Error")
    plt.draw()

    plt.show()
