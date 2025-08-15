import numpy as np
from numba import cuda as nc

from superspinsim.generate_simulator import generate_simulator
from superspinsim.generate_generators import \
    _generate_valid_indices, _generate_superoperators, real_eig


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

    # Make superoperators
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

    # Compile time-independent superoperators
    superoperators_quiescent = _compile_superoperators(superoperators, "Hq")
    superoperators_quiescent += _compile_superoperators(superoperators, "Lq")
    if len(superoperators_quiescent):
        superoperator_quiescent = sum(superoperators_quiescent)
    else:
        superoperator_quiescent = None

    # Compile time-dependent superoperators
    superoperators_time_dependent = \
        _compile_superoperators(superoperators, "Ht")
    superoperators_time_dependent += \
        _compile_superoperators(superoperators, "Lt")

    if superoperator_quiescent is None:
        # Set the time-independent von-neumann to the shape of the dissipators
        if len(superoperator_quiescent):
            superoperator_quiescent = \
                np.zeros_like(np.zeros_like(superoperators_time_dependent[0]))
        else:
            raise TypeError("No operators defined.")

    if use_rotating:
        return_value = _make_rotating(
            superoperator_quiescent, superoperators_time_dependent)
        superoperators_time_dependent, vectors_real, inv_vectors_real, \
            doubles, singles = return_value
    else:
        superoperators_time_dependent = \
            [superoperator_quiescent] + superoperators_time_dependent
    superoperators_time_dependent = np.array(superoperators_time_dependent)
    print(superoperators_time_dependent)

    # Compile coefficients
    coefficients = []

    for coefficient in ham_coefficients:
        coefficient_device = nc.jit(device=True)(coefficient)
        coefficients.append(coefficient_device)

    for coefficient in jump_coefficients:
        coefficient_device = nc.jit(device=True)(coefficient)

        def coefficient_sqrt(time):
            return np.sqrt(coefficient_device(time))

        coefficient_device = nc.jit(device=True)(coefficient_sqrt)
        coefficients.append(coefficient_device)

    # def lindbladian(time, sample):
    #     for coefficient_index, coefficient in enumerate(coefficients):
    #         sample[coefficient_index] = coefficient(time)

    python_str = ""
    for coefficient_index in range(len(coefficients)):
        python_str += \
            f"coefficient_{coefficient_index}" \
            f" = coefficients[{coefficient_index}]\n"
    python_str += "\ndef lindbladian(time, sample):\n"
    python_tab = " "*4
    for coefficient_index in range(len(coefficients)):
        python_str += python_tab \
            + f"sample[{coefficient_index}]" \
            + f" = coefficient_{coefficient_index}(time)\n"

    local_dict = {"coefficients": coefficients}
    exec(python_str)  # globals=globals(), locals=local_dict)
    print(locals())
    # for local_key, local_element in local_dict.items():
    #     locals()[local_key] = local_element

    # lindbladian = local_dict["lindbladian"]
    # coefficient_0 = local_dict["coefficient_0"]

    # def lindbladian(time, sample):
    #     sample[0] = coefficient_0(time)

    # lindbladian = local_dict["lindbladian"]

    print(python_str)
    print(local_dict)

    simulator = generate_simulator(
        lindbladian, superoperators_time_dependent, valid_indices,
        number_of_exponentials=5, number_of_fine_divisions=10,
        use_rotating=True,
        vectors_real=vectors_real, inv_vectors_real=inv_vectors_real,
        doubles=doubles, singles=singles,
    )
    return_value = simulator(rho0, ti, tf, dt)
    print(return_value)
    return return_value


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


def _compile_superoperators(superoperators: dict, key_prefix: str):
    jump_index = 0
    superoperators_list = []
    while True:
        key = f"{key_prefix}{jump_index}"
        if key not in superoperators:
            break
        superoperators_list.append(superoperators.pop(key))
        jump_index += 1
    return superoperators_list


def _make_rotating(
        superoperator_quiescent: np.ndarray,
        superoperators_time_dependent: list[np.ndarray]):
    vectors_real, inv_vectors_real, doubles, singles = \
        real_eig(superoperator_quiescent)

    superoperators_time_dependent_new = [
        inv_vectors_real@generator@vectors_real
        for generator in superoperators_time_dependent
    ]

    return_value = (
        superoperators_time_dependent_new, vectors_real, inv_vectors_real,
        doubles, singles
    )

    return return_value
