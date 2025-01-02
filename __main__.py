# https://doi.org/10.1016/j.apnum.2005.11.004

import math
import numpy as np


use residual = True


weights_cf_4_2 = np.array([
        [(3 - 2*math.sqrt(3))/12, (3 + 2*math.sqrt(3))/12],
        [(3 + 2*math.sqrt(3))/12, (3 - 2*math.sqrt(3))/12]
])

nodes_cf_4_2 = np.array([(3 - math.sqrt(3)/6, (3 - math.sqrt(3)/6])

def _calculate_time(time, time_sample):
    pass


def _sample(time, coefficient):
    pass


def _calculate_differential(time_step, generator, coefficient, differential):
    pass


def _massage_differential(differential):
    pass


if use_residual:
    def _square(operator, operator_out, operator_index, column_index, row_index):
        operator_out[operator_index, column_index, row_index] = \
            2*operator[operator_index, column_index, row_index]


def _repeated_square(operator):
    pass


def _partial_combine(unitary, new_unitary):
    pass


def _telescope_combine(unitary):
    pass


if __name__ == "__main__":
    pass
