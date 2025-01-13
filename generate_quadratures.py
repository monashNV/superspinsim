import numpy as np
import math


meta_datatype = np.float64

# Q matrices ------------------------------------------------------------------

# Eq (22) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_Q_2_2_gl = np.array(
    [
        [1/2, 1/2],
        [-math.sqrt(3)/12, math.sqrt(3)/12]
    ],
    dtype=meta_datatype
)

# Eq (22) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_Q_3_3_gl = np.array(
    [
        [5/18, 4/9, 5/18],
        [-math.sqrt(15)/36, 0, math.sqrt(15)/36],
        [1/24, 0, 1/24]
    ],
    dtype=meta_datatype
)

# R matrices ------------------------------------------------------------------

# Eq (24) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_R_2 = np.array(
    [
        [1, 0],
        [0, 12]
    ],
    dtype=meta_datatype
)

# Eq (24) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_R_3 = np.array(
    [
        [9/4, 0, 15],
        [0, 12, 0],
        [-15, 0, 180]
    ],
    dtype=meta_datatype
)


# X matrices ------------------------------------------------------------------


def _fill_symetric_X(x):
    """
    Eq (33) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
    """

    number_of_exponentials = x.shape[0]
    number_of_exponentials_fill = number_of_exponentials//2
    number_of_samples = x.shape[1]
    for exponential_index in range(number_of_exponentials_fill):
        for sample_index in range(number_of_samples):
            x[number_of_exponentials - exponential_index - 1, sample_index] = \
                ((-1)**sample_index)*x[exponential_index, sample_index]


# Eq (37) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_X_4_2 = np.array(
    [
        [1/2, 1/6],
        [0, 0]
    ],
    dtype=meta_datatype
)
_fill_symetric_X(_X_4_2)
print("X42", _X_4_2)

# Eq (37) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_X_4_3 = np.array(
    [
        [0, 1/12],
        [1, 0],
        [0, 0]
    ],
    dtype=meta_datatype
)
_fill_symetric_X(_X_4_3)
print("X43", _X_4_3)

# Table 1 Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_X_6_5 = np.array(
    [
        [0.2, 0.08734395950888931101, 0.03734395950888931101],
        [0.34815492558797391479, 0.053438272547684150, 0.00584269157837031012],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ],
    dtype=meta_datatype
)
_X_6_5[2, 0] = 1/2 - 2*(_X_6_5[0, 0] + _X_6_5[1, 0])
_X_6_5[2, 2] = 1/12 - 2*(_X_6_5[0, 2] + _X_6_5[1, 2])
_fill_symetric_X(_X_6_5)
print("X65", _X_6_5)


# rho matrices ----------------------------------------------------------------

_rho_4_2_gl = _X_4_2@_R_2@_Q_2_2_gl
_rho_4_3_gl = _X_4_3@_R_2@_Q_2_2_gl
_rho_6_5_gl = _X_6_5@_R_3@_Q_3_3_gl
print("p42", _rho_4_2_gl)
print("p43", _rho_4_3_gl)
print("p65", _rho_6_5_gl)
