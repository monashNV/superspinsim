import numpy as np
from numpy import random as npr
from numba import cuda as nc
import time as tm


@nc.jit(device=True)
def residual_square(matrix_input, matrix_output):
    if not (nc.threadIdx.x < 9 and nc.threadIdx.y < 9):
        return
    matrix_output[nc.blockIdx.x, nc.threadIdx.y, nc.threadIdx.x] = \
        2*matrix_input[nc.blockIdx.x, nc.threadIdx.y, nc.threadIdx.x]
    for interior_index in range(9):
        matrix_output[nc.blockIdx.x, nc.threadIdx.y, nc.threadIdx.x] = \
            nc.fma(
                matrix_input[nc.blockIdx.x, nc.threadIdx.y, interior_index],
                matrix_input[nc.blockIdx.x, interior_index, nc.threadIdx.x],
                matrix_output[nc.blockIdx.x, nc.threadIdx.y, nc.threadIdx.x]
            )


@nc.jit
def multi_residual_square(matrix_input, matrix_output):
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()
    residual_square(matrix_input, matrix_output)
    nc.syncthreads()
    residual_square(matrix_output, matrix_input)
    nc.syncthreads()


if __name__ == "__main__":
    rng = npr.default_rng(1234567890)
    matrix = rng.standard_normal((1000000, 9, 9), dtype=np.float64)/pow(2, 32)

    wall_time_start = tm.time()

    matrix_device = nc.to_device(matrix)
    output_device = nc.device_array_like(matrix_device)

    wall_time_loaded = tm.time()

    multi_residual_square[(1000000, 1), (9, 9)](matrix_device, output_device)

    wall_time_computed = tm.time()

    output = output_device.copy_to_host()

    wall_time_copied = tm.time()

    print(wall_time_loaded - wall_time_start)
    print(wall_time_computed - wall_time_loaded)
    print(wall_time_copied - wall_time_computed)
    print(wall_time_copied - wall_time_start)

    print(output)
