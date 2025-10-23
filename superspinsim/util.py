import math
import numpy as np


def colour_complex_matrix(inp: np.ndarray):
    """Returns an RGB image representing a complex matrix.
    Complex phase is represented by hue.
    Can be used to plot a density matrix.

    Parameters
    ----------
    inp: numpy.ndarray
        The complex matrix to visualise.

    Returns
    -------
    out: numpy.ndarray
        An RGB matrix representing the complex matrix.
    """

    out = np.zeros(
        (inp.shape[0], inp.shape[1], 3),
        dtype=np.float64
    )
    if len(inp.shape) == 2:
        if np.iscomplexobj(inp):
            new_inp = np.zeros(
                (inp.shape[0], inp.shape[1], 2),
                dtype=np.float64
            )
            new_inp[:, :, 0] = inp.real
            new_inp[:, :, 1] = inp.imag
        else:
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
    out += 1 - np.sqrt(inp[:, :, 0]**2 + inp[:, :, 1]**2).reshape(
        (inp.shape[0], inp.shape[1], 1))
    out = np.clip(out, 0, 1)
    return out
