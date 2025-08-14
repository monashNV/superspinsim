import numpy as np

from superspinsim.generate_simulator import generate_simulator


def mesolve(H, rho0: np.ndarray, ti: float, tf: float, dt: float, c_ops=None):
    """
        A wrapper for superspinsim to provide a similar syntax to other
        simulators.
    """

    # Sort hamiltonians
    ham_quiescent, ham_time_dependent, ham_coeficients = \
        _sort_operators(H, "H")
    # Sort collapse operators
    ham_quiescent, ham_time_dependent, ham_coeficients = \
        _sort_operators(c_ops, "c_ops")


def _sort_operators(operators, operator_label: str):
    type_error_message = \
        f"{operator_label} must be numpy.ndarray, an [np.ndarray, function]" \
        + "list, or list of either."
    quiescent = None
    time_dependent = []
    coefficients = []
    return_value = [quiescent, time_dependent, coefficients]

    # Are there any operators?
    if operators is None:
        return return_value

    # Is there just one operator?
    if isinstance(operators, np.ndarray):
        quiescent.append(operators)
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
            if quiescent is None:
                quiescent = operator.copy()
            else:
                quiescent += operator
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
