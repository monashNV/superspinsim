# https://doi.org/10.1016/j.apnum.2005.11.004

from util import colour_complex_matrix as _colour_complex_matrix
from generators import superoperators as superperator_basis_dict

import math
import numpy as np
import numba as nb
import numba.cuda as nc

import scipy.linalg as sl

import matplotlib.pyplot as plt
# from cmcrameri import cm


meta_use_cuda = True
meta_datatype = np.float64
meta_use_residual = True
meta_wavefunction_size = 7
meta_operator_size = meta_wavefunction_size**2
meta_number_of_quartic_repeats = 23

if meta_use_cuda:
    _synchronise_block = nc.syncthreads
    _fma = nc.fma
else:
    def _synchronise_block():
        return 1

    def _fma(x, y, z):
        return x*y + z


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
        _synchronise_block()
        for x_index_stride in range(meta_wavefunction_size):
            _square_superoperator(
                scratch, superoperator, y_index,
                x_index_reduced + x_index_stride*meta_wavefunction_size
            )
        _synchronise_block()


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


def _partial_combine(time_evolution, new_time_evolution):
    pass


def _copy_superoperator(original, clone, y_index, x_index):
    clone[y_index, x_index, 0] = original[y_index, x_index, 0]
    clone[y_index, x_index, 1] = original[y_index, x_index, 1]


if meta_use_cuda:
    _multiply_superoperator = nc.jit(_multiply_superoperator, device=True)
    _copy_superoperator = nc.jit(_copy_superoperator, device=True)

    def _basic_combine_kernel(time_evolutions, time_index):
        scratch = nc.shared.array(
            (meta_operator_size, meta_operator_size, 2), dtype=meta_datatype
        )

        if nc.blockIdx.y < meta_operator_size \
                and nc.blockIdx.x < meta_wavefunction_size:
            for x_index_stride in range(meta_wavefunction_size):
                _multiply_superoperator(
                    time_evolutions[time_index, :, :, :],
                    time_evolutions[time_index + 1, :, :, :],
                    scratch,
                    nc.threadIdx.y,
                    nc.threadIdx.x + x_index_stride*meta_wavefunction_size
                )
            _synchronise_block()

            for x_index_stride in range(meta_wavefunction_size):
                _copy_superoperator(
                    scratch,
                    time_evolutions[time_index + 1, :, :, :],
                    nc.threadIdx.y,
                    nc.threadIdx.x + x_index_stride*meta_wavefunction_size
                )
            _synchronise_block()

    _basic_combine_kernel = nc.jit(_basic_combine_kernel)


nb.jit(nopython=True, parallel=False)
def _basic_combine_run(time_evolutions):
    if meta_use_cuda:
        block_size = (meta_wavefunction_size, meta_operator_size)
        for time_index in range(0, time_evolutions.shape[0] - 1):
            _basic_combine_kernel[(1, 1), block_size] \
                (time_evolutions, time_index)


def _telescope_combine(unitary):
    pass


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
        (15, meta_operator_size, meta_operator_size, 2),
        dtype=meta_datatype
    )
    superoperators[:, :, :, :] = (math.tau/15)*generators[0, :, :, :]

    if meta_use_cuda:
        superoperators_device = nc.to_device(
            superoperators/(2**(2*meta_number_of_quartic_repeats))
        )
    else:
        superoperators_device = superoperators \
            / (2**(2*meta_number_of_quartic_repeats))

    _repeated_quartic_superoperator_run(superoperators_device)

    _basic_combine_run(superoperators_device)

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
        plt.axis("off")

    print("Done!")


if __name__ == "__main__":
    # test_generators()
    # test_squaring()
    test_combination()
    plt.show()
