import numpy as np
from numba import cuda as nc

import os
import sys
from importlib import util as ilu

import qutip as qt

from superspinsim.generate_simulator import generate_simulator
from superspinsim.generate_generators import \
    _generate_valid_indices, _generate_superoperators, real_eig, \
    generate_atoms, _generate_lindbladian


def mesolve(
        H, rho0: np.ndarray, ti: float, tf: float, dt: float, c_ops=None,
        allowed: np.ndarray = None, use_rotating: bool = True,
        number_of_exponentials: int = 5, number_of_fine_divisions: int = 10):
    """A wrapper for superspinsim to provide a similar syntax to other
    simulators, like `qutip.mesolve`.

    Parameters
    ----------
    H
        Hamiltonian of the system.
        Static Hamiltonians can be defined as either a `numpy.ndarray`, or a
        `qutip.Qobj`.
        Dynamic Hamiltonians can be defined using a two-element `list`, where
        the first element is a static Hamiltonian, and the second element is a
        python function which defines a time-dependent coefficient of said
        Hamiltonian.
        Multiple static and dynamic Hamiltonians can be combined by providing
        a `list` of Hamiltonians in the format described above.
    rho0: `numpy.ndarray`
        The initial density matrix of the system.
    ti: `float`
        The initial time of the simulation.
    tf: `float`
        The final time of the simulation.
    dt: `float`
        The time step at which to provide solutions.
        These steps are conducted in parallel.
    c_ops (optional)
        Jump (collapse) operators of the system.
        Static jump operators can be defined as either a `numpy.ndarray`, or a
        `qutip.Qobj`.
        Dynamic jump operators can be defined using a two-element `list`, where
        the first element is a static jump operator, and the second element is
        a python function which defines a time-dependent coefficient of said
        jump operator.
        Multiple static and dynamic jump operators can be combined by providing
        a `list` of jump operators in the format described above.
    allowed: `numpy.ndarray` (optional)
        A mask (matrix of zeroes and ones) of the locations of allowed non-zero
        density matrix elements.
        This can be used to exclude certain elements that are known to be zero,
        *i.e.*, the coherences of between known incoherent states.
    use_rotating: `bool` (default is `True`)
        Whether or not to transform the problem into the generalised rotating
        frame.
    number_of_exponentials: `int` (default is 5)
        The choice of which commutator-free magnus integrator to use, labelled
        by the number of exponentials per step used in said integrator.
    number_of_fine_divisions: `int` (default is 10)
        The number of integration steps to conduct per every time step given in
        the solution.
        These steps are not conducted in parallel.

    Returns
    -------
    time: np.ndarray
        Time samples in seconds.
    density: np.ndarray
        Evaluated density matrices at the times given by :obj"`time` .
    """

    ham_quiescent, ham_time_dependent, ham_coefficients = \
        _sort_operators(H, "H")
    ham_quiescent = [sum(ham_quiescent)]
    jump_quiescent, jump_time_dependent, jump_coefficients = \
        _sort_operators(c_ops, "c_ops")

    superoperators_time_dependent, valid_indices, rotating_dict = \
        _make_superoperators(
            ham_quiescent, ham_time_dependent, jump_quiescent,
            jump_time_dependent, allowed, use_rotating
        )

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

    # Write coefficient function factory
    import_name, import_path, import_directory = \
        _write_python_module(len(coefficients))
    spec = ilu.spec_from_file_location(import_name, import_path)
    import_module = ilu.module_from_spec(spec)
    sys.modules[import_name] = import_module
    spec.loader.exec_module(import_module)

    lindbladian = import_module.make_lindbladian(coefficients)

    simulator = generate_simulator(
        lindbladian, superoperators_time_dependent, valid_indices,
        number_of_exponentials=number_of_exponentials,
        number_of_fine_divisions=number_of_fine_divisions,
        use_rotating=use_rotating, **rotating_dict
    )
    return_value = simulator(rho0, ti, tf, dt)

    # Cleanup
    _remove_python_module(import_path, import_directory)

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
    elif isinstance(operators, qt.Qobj):
        quiescent.append(operators.full())
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
        if isinstance(operator, qt.Qobj):
            quiescent.append(operator.full())
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

    if isinstance(pair[0], qt.Qobj) and callable(pair[1]):
        time_dependent.append(pair[0].full())
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


def _write_python_module(number_of_coefficients: int):
    PYTHON_TAB = " "*4
    python_str = ""
    python_str += "def make_lindbladian(coefficients):\n"
    for coefficient_index in range(number_of_coefficients):
        python_str += PYTHON_TAB + \
            f"coefficient_{coefficient_index}" \
            f" = coefficients[{coefficient_index}]\n"
    python_str += "\n" + PYTHON_TAB + "def lindbladian(time, sample):\n"
    for coefficient_index in range(number_of_coefficients):
        python_str += 2*PYTHON_TAB \
            + f"sample[{coefficient_index}]" \
            + f" = coefficient_{coefficient_index}(time)\n"
    python_str += "\n" + PYTHON_TAB + "return lindbladian"

    # Save as python module
    import_name = "jit_package"
    temp_directory = "~/.temp"
    temp_directory = os.path.expanduser(temp_directory)
    import_directory = f"{temp_directory}/superspinsim"
    if not os.path.exists(import_directory):
        os.makedirs(import_directory)
    import_path = f"{import_directory}/{import_name}"
    import_index = 0
    while os.path.exists(f"{import_path}_{import_index}.py"):
        import_index += 1
    import_path = f"{import_path}_{import_index}.py"
    with open(import_path, "w") as file:
        file.write(python_str)

    return import_name, import_path, import_directory


def _remove_python_module(import_path: str, import_directory: str):
    os.remove(import_path)
    import_directory_ls = os.listdir(import_directory)
    if len(import_directory_ls) == 0:
        os.removedirs(import_directory)
    elif len(import_directory_ls) == 1 and \
            import_directory_ls[0] == "__pycache__":
        pycache_directory = f"{import_directory}/__pycache__"
        for file_name in os.listdir(pycache_directory):
            os.remove(f"{pycache_directory}/{file_name}")
        os.removedirs(pycache_directory)


def _make_superoperators(
        ham_quiescent: list[np.ndarray], ham_time_dependent: list[np.ndarray],
        jump_quiescent: list[np.ndarray],
        jump_time_dependent: list[np.ndarray], allowed: np.ndarray,
        use_rotating: bool):
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
        # print(label, operator)
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
        rotating_dict = {
            "vectors_real": vectors_real,
            "inv_vectors_real": inv_vectors_real,
            "doubles": doubles,
            "singles": singles
        }
    else:
        superoperators_time_dependent = \
            [superoperator_quiescent] + superoperators_time_dependent
        rotating_dict = {}

    superoperators_time_dependent = np.array(superoperators_time_dependent)

    return_value = \
        (superoperators_time_dependent, valid_indices, rotating_dict)
    return return_value


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


def simspins(
        coefficients: list[callable], time_start: float, time_end: float,
        time_step: float, spins: list[list[dict]],
        spin_interactions: list[dict], group_interactions: dict,
        density_initial: np.ndarray, number_of_exponentials: int = 5,
        number_of_fine_divisions: int = 1,
        number_of_quadratic_repeats: int = 35, use_rotating: bool = True,
        use_residual: bool = True):
    """Run a simulation based on a description of spins, as described in
    `Spin description syntax`_.

    Parameters
    ----------
    coefficients: list[callable]
        A list of the coefficients of the von Neumann and Dissipator
        generators.
        The first entries are the x, y, and z magnetic fields, in Tesla.
        The final entry is the inceherent optical excitation as fraction of
        saturation.
    time_start: `float`
        The initial time of the simulation.
    time_end: `float`
        The final time of the simulation.
    time_step: `float`
        The time step at which to provide solutions.
        These steps are conducted in parallel.
    spins: list[list[dict]]
        Descriptions of individual "atoms" of spin.
        See `Individual spins`_ .
    spin_interactions: list[dict[tuple[int, int], dict]]
        Descriptions of coherent interactions between spins.
        See `Spin-spin interaction description`_ .
    group_interactions: dict[tuple[tuple[int, int], tuple[int, int]], dict]
        Description of incoherent interactions between spins.
        See `Incoherent interactions between coherent blocks`_ .
    density_initial: numpy.ndarray
        Initial density matrix to evolve through time.
    number_of_exponentials: int (default is 5)
        A shorthand for the kind of commutator-free Magnus integrator method
        being used in the simulator.
        Choose :obj:`1` for the CF2:1 second-order method.
        Choose :obj:`2` for the CF4:2 fourth-order method.
        Choose :obj:`3` for the CF4:3 fourth-order method.
        Choose :obj:`5` for the CF6:5 sixth-order method.
        Choose :obj:`6` for the CF6:6 sixth-order method.
    number_of_fine_divisions: int (default is 1)
        The number of integration steps to conduct per every time step given in
        the solution.
        These steps are not conducted in parallel.
    number_of_quadratic_repeats: int (default is 35)
        Twice the number of repeated squares to use in matrix exponentiation.
    use_rotating: bool (default is True)
        Whether or not to use the generalised rotating frame technique.
    use_residual: bool (default is True)
        Whether or not to use the residual arithmetic technique.

    Returns
    -------
    time: np.ndarray
        Time samples in seconds.
    density: np.ndarray
        Evaluated density matrices at the times given by :obj"`time` .
    """

    generators, vectorisation_map = generate_atoms(
        spins, spin_interactions, group_interactions)
    generators = list(generators["generators"].values())

    if use_rotating:
        generators, vectors_real, inv_vectors_real, doubles, singles = \
            _make_rotating(generators[0], generators[1:])
        rotating_dict = {
            "vectors_real": vectors_real,
            "inv_vectors_real": inv_vectors_real,
            "doubles": doubles,
            "singles": singles
        }
    else:
        rotating_dict = {}

    lindbladian = _generate_lindbladian(coefficients, use_rotating)
    simulator = generate_simulator(
        lindbladian, np.array(generators), vectorisation_map,
        use_residual=use_residual,
        number_of_exponentials=number_of_exponentials,
        number_of_fine_divisions=number_of_fine_divisions,
        number_of_quartic_repeats=number_of_quadratic_repeats,
        use_rotating=use_rotating,
        **rotating_dict
    )
    return_value = simulator(density_initial, time_start, time_end, time_step)
    return return_value
