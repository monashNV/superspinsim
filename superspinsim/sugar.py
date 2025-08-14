import numpy as np

from superspinsim.generate_simulator import generate_simulator
from superspinsim.generate_generators import \
    _generate_valid_indices, _generate_superoperators

def mesolve(
        H, rho0: np.ndarray, ti: float, tf: float, dt: float, c_ops=None,
        allowed: np.ndarray = None, use_rotating: bool = True):
    """
        A wrapper for superspinsim to provide a similar syntax to other
        simulators.
    """

    # Sort hamiltonians
    ham_quiescent, ham_time_dependent, ham_coefficients = \
        _sort_operators(H, "H")
    ham_quiescent = [sum(ham_quiescent)]
    # Sort collapse operators
    jump_quiescent, jump_time_dependent, jump_coefficients = \
        _sort_operators(c_ops, "c_ops")

    operator_dict = {}
    for gen_index, gen in enumerate(ham_quiescent):
        operator_dict[f"Hq{gen_index}"] = gen
    for gen_index, gen in enumerate(ham_time_dependent):
        operator_dict[f"Ht{gen_index}"] = gen
    for gen_index, gen in enumerate(jump_quiescent):
        operator_dict[f"Lq{gen_index}"] = gen
    for gen_index, gen in enumerate(jump_time_dependent):
        operator_dict[f"Lt{gen_index}"] = gen

    new_operator_dict = {}
    for label, operator in operator_dict.items():
        print(label, operator)
        new_operator = np.empty(
            (operator.shape[0], operator.shape[1], 2), np.float64)
        new_operator[:, :, 0] = np.real(operator)
        new_operator[:, :, 1] = np.imag(operator)
        new_operator_dict[label] = new_operator
    operator_dict = new_operator_dict

    if allowed is None:
        if len(ham_quiescent) > 0:
            allowed = np.ones_like(ham_quiescent[0])
        elif len(ham_time_dependent) > 0:
            allowed = np.ones_like(ham_time_dependent[0])
        elif len(jump_quiescent) > 0:
            allowed = np.ones_like(jump_quiescent[0])
        elif len(jump_time_dependent) > 0:
            allowed = np.ones_like(jump_time_dependent[0])
        else:
            raise TypeError("No operators specified.")
    valid_indices = _generate_valid_indices(allowed)

    superoperators = _generate_superoperators(operator_dict, valid_indices)
    print(superoperators)


def _sort_operators(operators, operator_label: str):
    type_error_message = \
        f"{operator_label} must be numpy.ndarray, an [np.ndarray, function]" \
        + "list, or list of either."
    quiescent = []
    time_dependent = []
    coefficients = []
    return_value = [quiescent, time_dependent, coefficients]

    # Are there any operators?
    if operators is None:
        return return_value

    # Is there just one operator?
    if isinstance(operators, np.ndarray):
        quiescent.append(operators)
        return return_value
    elif type(operators) is not list:
        raise TypeError(type_error_message)

    # Empty list
    if len(operators) < 1:
        return return_value

    # Two elements
    elif len(operators) == 2:
        # One time-dependent pair
        if _check_time_dependent_pair(operators, time_dependent, coefficients):
            return return_value

    # A longer list of operators
    for operator in operators:
        # Time-independent case
        if isinstance(operator, np.ndarray):
            quiescent.append(operator)
            continue

        # Time-dependent case
        if not _check_time_dependent_pair(
                operator, time_dependent, coefficients):
            raise TypeError(type_error_message)

    return return_value


def _check_time_dependent_pair(
        pair: list, time_dependent: list, coefficients: list):
    if len(pair) != 2:
        return False

    if isinstance(pair[0], np.ndarray) and callable(pair[1]):
        time_dependent.append(pair[0])
        coefficients.append(pair[1])
        return True
    return False
