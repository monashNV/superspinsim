import math
import numpy as np


def colour_complex_matrix(inp):
    out = np.zeros(
        (inp.shape[0], inp.shape[1], 3),
        dtype=inp.dtype
    )
    if len(inp.shape) == 2:
        new_inp = np.zeros(
            (inp.shape[0], inp.shape[1], 2),
            dtype=inp.dtype
        )
        new_inp[:, :, 0] = inp
        inp = new_inp
    out[:, :, 0] += (2/math.sqrt(6))*inp[:, :, 0]
    out[:, :, 1] += (-1/math.sqrt(6))*inp[:, :, 0] \
        + (1/math.sqrt(2))*inp[:, :, 1]
    out[:, :, 2] += (-1/math.sqrt(6))*inp[:, :, 0] \
        + (-1/math.sqrt(2))*inp[:, :, 1]
    # vmax = np.max(np.sqrt(np.sum(out**2, axis = 2)))
    # out /= 2*vmax
    # out += 1/math.sqrt(3)
    out += np.sqrt(inp[:, :, 0]**2 + inp[:, :, 1]**2).reshape(
        (inp.shape[0], inp.shape[1], 1))
    out = np.clip(out, 0, 1)
    return out
