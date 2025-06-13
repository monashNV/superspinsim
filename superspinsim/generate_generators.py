import numpy as np
import scipy.linalg as spl
import math
import copy

import numba.cuda as nc

import superspinsim.params as s3p
from superspinsim.util import colour_complex_matrix

meta_datatype = np.float64


def generate_atoms(
    description: list[list[dict]], atom_interactions: list[dict],
    block_interactions: dict, verbose=False
) -> [dict, dict, list[dict], dict]:
    """
        Define a system of multiple spins.
    """

    # We're going to add to these structures, so it's best to deep copy.
    description = copy.deepcopy(description)
    atom_interactions = copy.deepcopy(atom_interactions)
    block_interactions = copy.deepcopy(block_interactions)

    hilbert_space_shape = []
    operator_labels = set()
    tensor_labels = set()
    zero_field_labels = set()
    field_labels = set()
    projector_labels = set()
    dissipator_labels = set()
    label_sets = {
        "operator_labels": operator_labels,
        "tensor_labels": tensor_labels,
        "zero_field_labels": zero_field_labels,
        "field_labels": field_labels,
        "projector_labels": projector_labels,
        "dissipator_labels": dissipator_labels
    }
    previous_identity = None

    for block, atom_interaction in zip(description, atom_interactions):

        # Individual atoms
        for atom_index, atom in enumerate(block):
            previous_identity = _add_atom(
                block, atom, label_sets,
                previous_identity, hilbert_space_shape
            )

        # Spin-spin interactions and molecular hyperfine
        for (index_a, index_b), j_description in atom_interaction.items():
            _add_spin_spin_coupling(
                block, j_description, index_a, index_b, label_sets)

        previous_identity = None

    # Make traceless
    _remove_trace(description, operator_labels)

    # Combine all atoms in blocks
    coherent_atoms = _combine_coherent_atoms(
        description, atom_interactions, field_labels)

    allowed = _combine_blocks(description, coherent_atoms, label_sets)

    coherent_blocks = _combine_coherent_blocks(coherent_atoms)

    for ((block_a, atom_a), (block_b, atom_b)), interaction in \
            block_interactions.items():
        _add_block_interaction(
            block_a, atom_a, block_b, atom_b, interaction, description,
            label_sets
        )

    # Create lists of operators
    operator_dicts = _list_operators(
        description, atom_interactions, block_interactions, label_sets)
    operator_dicts["coherent_atoms"] = coherent_atoms
    operator_dicts["coherent"] = coherent_blocks

    # Generate superoperators
    operators_to_super = {**operator_dicts["jump"], **coherent_blocks}
    valid_indices = _generate_valid_indices(allowed)
    superoperators = _generate_superoperators(
        operators_to_super, valid_indices)
    dissipator_dict = _combine_superoperators(superoperators)

    operator_dicts["super_all"] = superoperators
    operator_dicts["dissipators"] = dissipator_dict

    generators_dict = {}
    for label in ["L0", "Gx", "Gy", "Gz", "Gr"]:
        if label in superoperators.keys():
            generators_dict[label] = superoperators[label]

    operator_dicts["generators"] = generators_dict

    return operator_dicts, valid_indices


def _add_atom(
        block: list[dict], atom: dict, label_sets: dict[str, set[str]],
        previous_identity: np.ndarray, hilbert_space_shape: list[int]):
    """
        Define an atom.
    """

    electron_spin, previous_identity = _add_electron(
        block, atom, label_sets, previous_identity, hilbert_space_shape)

    # Nuclear spin
    nuclear_spin, previous_identity = _add_nucleus(
        block, atom, label_sets, previous_identity, hilbert_space_shape)

    # Hyperfine
    _add_hyperfine(electron_spin, nuclear_spin, atom, label_sets)

    # Generators
    _combine_in_atom(electron_spin, nuclear_spin, atom, label_sets)

    return previous_identity


def _add_electron(
            block: list[dict], atom: dict, label_sets: dict[str, set[str]],
            previous_identity: np.ndarray, hilbert_space_shape: list[int]
        ) -> [int, np.ndarray]:
    """
        Defines the electron spin system.
    """

    operator_labels = label_sets["operator_labels"]
    zero_field_labels = label_sets["zero_field_labels"]
    tensor_labels = label_sets["tensor_labels"]
    projector_labels = label_sets["projector_labels"]
    dissipator_labels = label_sets["dissipator_labels"]

    temp_labels = set()

    # Electron spin
    if "S" in atom.keys():
        spin = atom["S"]
    else:
        spin = 0

    spin_vec, projectors, spin_identity = add_spin(
        spin, hilbert_space_shape)
    for key, projector in projectors.items():
        _record_operator(
            f"S{key}", projector, atom,
            [temp_labels, operator_labels, projector_labels]
        )
    if spin > 0:
        _record_spin_vec(
            "S", spin_vec, atom, [temp_labels, operator_labels])

    # Electron ZFS
    if spin > 1/2:
        zfs = np.zeros((3, 3), dtype=meta_datatype)
        zfs_longitudinal = 0
        zfs_transverse = 0

        if "D" in atom.keys():
            zfs_longitudinal = atom["D"]
        if "E" in atom.keys():
            zfs_transverse = atom["E"]
        zfs[0, 0] = zfs_transverse - zfs_longitudinal/3
        zfs[1, 1] = -zfs_transverse - zfs_longitudinal/3
        zfs[2, 2] = zfs_longitudinal*2/3

        quadratic = _quadratic_outer(spin_vec, spin_vec)
        _record_spin_quadratic(
            "S", "S", quadratic, atom, [temp_labels, operator_labels])

        zfs_generator = _quadratic_transform(zfs, quadratic)
        _record_operator(
            "HD", zfs_generator, atom,
            [temp_labels, operator_labels, zero_field_labels]
        )

        atom["Dten"] = zfs
        tensor_labels.add("Dten")

    # Electron Zeeman
    if spin > 0:
        if "g" in atom.keys():
            g_iso = atom["g"]
            if "g_dipole" in atom.keys():
                g_dipole = atom["g_dipole"]
            elif "g_perp" in atom.keys():
                g_perp = atom["g_perp"]
                g_dipole = (g_iso - g_perp)/3
                g_iso -= 2*g_dipole
            else:
                g_dipole = 0
        else:
            g_iso = 2
            g_dipole = 0

        g_spin = np.zeros((3, 3), dtype=meta_datatype)
        g_spin[0, 0] = g_iso - g_dipole
        g_spin[1, 1] = g_iso - g_dipole
        g_spin[2, 2] = g_iso + 2*g_dipole

        electron_gyro = g_spin*math.tau*s3p.fundamental.bohr_gyro
        atom["gyroS"] = electron_gyro
        tensor_labels.add("gyroS")

        generator_vec = _linear_transform(electron_gyro, spin_vec)
        _record_spin_vec(
            "GS", generator_vec, atom, [temp_labels, operator_labels])

        quiescent_magnetic_generator = None
        if "B0" in atom.keys():
            quiescent_magnetic_field = atom["B0"]
            quiescent_magnetic_generator = \
                quiescent_magnetic_field[0]*generator_vec[0] \
                + quiescent_magnetic_field[1]*generator_vec[1] \
                + quiescent_magnetic_field[2]*generator_vec[2]

            # Dephasing:
            # Dephasing modelled as fluctuations of magnetic field in its
            # current direction, meaning that only changes due to the
            # Zeeman effect matter.
            # See 1D from sup mat of [Hapuarachchi et al. Opt. Express 32,
            # 22352-22361 (2024)]
            if "TS2" in atom.keys():
                dephasing_time = atom["TS2"]
                if np.sum(quiescent_magnetic_generator) > 0:
                    dephasing_operator = quiescent_magnetic_generator
                else:
                    dephasing_operator = generator_vec[2]
                l2 = np.max(
                    np.linalg.eigvalsh(
                        dephasing_operator[:, :, 0]
                        + 1j*dephasing_operator[:, :, 1]))
                jump_dephasing = (spin/(l2*math.sqrt(dephasing_time))) \
                    * dephasing_operator
                _record_operator(
                    "LS2", jump_dephasing, atom,
                    [temp_labels, operator_labels, dissipator_labels]
                )

        temp_labels |= _add_thermalisation(
            "S", atom, quiescent_magnetic_generator, zfs_generator,
            label_sets
        )

    previous_identity = _product_spin_state(
        previous_identity, spin_identity, block, atom, temp_labels,
        operator_labels, dissipator_labels
    )

    return spin, previous_identity


def _add_nucleus(
            block: list[dict], atom: dict, label_sets: dict[str, set[str]],
            previous_identity: np.ndarray, hilbert_space_shape: list[int]
        ) -> [int, np.ndarray]:
    """
        Generates a nucleus.
    """

    operator_labels = label_sets["operator_labels"]
    zero_field_labels = label_sets["zero_field_labels"]
    tensor_labels = label_sets["tensor_labels"]
    dissipator_labels = label_sets["dissipator_labels"]

    if "I" in atom.keys():
        temp_labels = set()

        spin = atom["I"]

        spin_vec, projectors, spin_identity = add_spin(
            spin, hilbert_space_shape)

        _record_spin_vec(
            "I", spin_vec, atom, [temp_labels, operator_labels])

        # Nuclear quadrupole
        zfs = np.zeros((3, 3), dtype=meta_datatype)
        zfs_longitudinal = 0
        zfs_transverse = 0
        if spin > 1/2:
            if "P" in atom.keys():
                zfs_longitudinal = atom["P"]
            if "Pt" in atom.keys():
                zfs_transverse = atom["Pt"]
            zfs[0, 0] = zfs_transverse - zfs_longitudinal/3
            zfs[1, 1] = -zfs_transverse - zfs_longitudinal/3
            zfs[2, 2] = zfs_longitudinal*2/3

        if np.sum(np.abs(zfs)) > 0:
            quadratic = _quadratic_outer(spin_vec, spin_vec)
            _record_spin_quadratic(
                "I", "I", quadratic, atom, [temp_labels, operator_labels])

            zfs_generator = _quadratic_transform(zfs, quadratic)
            _record_operator(
                "HP", zfs_generator, atom,
                [temp_labels, operator_labels, zero_field_labels]
            )

            atom["Pten"] = zfs
            tensor_labels.add("Pten")

    else:
        spin = 0

    # Nuclear Zeeman
    if spin > 0:
        if "gN" in atom.keys():
            g_iso = atom["gN"]
            if "gN_dipole" in atom.keys():
                g_dipole = atom["gN_dipole"]
            elif "gN_perp" in atom.keys():
                g_perp = atom["gN_perp"]
                g_dipole = (g_iso - g_perp)/3
                g_iso -= 2*g_dipole
            else:
                g_dipole = 0
        else:
            g_iso = 2
            g_dipole = 0

        g_spin = np.zeros((3, 3), dtype=meta_datatype)
        g_spin[0, 0] = g_iso - g_dipole
        g_spin[1, 1] = g_iso - g_dipole
        g_spin[2, 2] = g_iso + 2*g_dipole

        atom["gyroI"] = -g_spin*math.tau*s3p.fundamental.nuclear_gyro
        tensor_labels |= {"gyroI"}

        generator_vec = _linear_transform(g_spin, spin_vec)
        _record_spin_vec(
            "GI", generator_vec, atom, [temp_labels, operator_labels])

        quiescent_magnetic_generator = None
        if "B0" in atom.keys():
            quiescent_magnetic_field = atom["B0"]
            quiescent_magnetic_generator = \
                quiescent_magnetic_field[0]*generator_vec[0] \
                + quiescent_magnetic_field[1]*generator_vec[1] \
                + quiescent_magnetic_field[2]*generator_vec[2]

            # Dephasing:
            # Dephasing modelled as fluctuations of magnetic field in its
            # current direction, meaning that only changes due to the
            # Zeeman effect matter.
            # See 1D from sup mat of [Hapuarachchi et al. Opt. Express 32,
            # 22352-22361 (2024)]
            if "TI2" in atom.keys():
                dephasing_time = atom["TI2"]
                if np.sum(quiescent_magnetic_generator) > 0:
                    dephasing_operator = quiescent_magnetic_generator
                else:
                    dephasing_operator = generator_vec[2]
                l2 = np.max(
                    np.linalg.eigvalsh(
                        dephasing_operator[:, :, 0]
                        + 1j*dephasing_operator[:, :, 1]))
                jump_dephasing = (spin/(l2*math.sqrt(dephasing_time))) \
                    * dephasing_operator
                _record_operator(
                    "LI2", jump_dephasing, atom,
                    [temp_labels, operator_labels, dissipator_labels]
                )

        temp_labels |= _add_thermalisation(
            "I", atom, quiescent_magnetic_generator, zfs_generator, label_sets)

        previous_identity = _product_spin_state(
            previous_identity, spin_identity, block, atom,
            temp_labels, operator_labels, dissipator_labels
        )

    return spin, previous_identity


def _add_thermalisation(
    spin_label: str, atom: dict, quiescent_magnetic_generator: np.ndarray,
    zfs_generator: np.ndarray, label_sets: dict[str, set[str]]
) -> set[str]:
    """
        Themalisation (T1 time):
        Model: The rate of flow into an energy eigenstate is proportional
        to its Boltzmann factor.
        See Equation (VIII.9) from [Abragam "The Principles of Nuclear
        Magnetism", 1961, ISBN 0198512368]
    """

    operator_labels = label_sets["operator_labels"]
    dissipator_labels = label_sets["dissipator_labels"]
    temp_labels = set()

    if f"T{spin_label}1" in atom.keys():
        thermalisation_time = atom[f"T{spin_label}1"]

        # Find quiescent energy eigstates and values
        if quiescent_magnetic_generator is None:
            quiescent_generator = zfs_generator.copy()
        else:
            quiescent_generator = quiescent_magnetic_generator + zfs_generator
        quiescent_energies, quiescent_states = np.linalg.eigh(
            quiescent_generator[:, :, 0]
            + 1j*quiescent_generator[:, :, 1]
        )

        # Find Boltzmann factors
        if "T" in atom.keys():
            temperature = atom["T"]
        else:
            temperature = s3p.standards.lab.ntp.temperature
        if temperature > 0:
            boltzmann_factors = np.exp(-quiescent_energies/(
                math.tau*s3p.fundamental.boltzmann_gyro
            ))
        else:
            boltzmann_factors = np.zeros_like(
                quiescent_energies, dtype=meta_datatype)
            energy_min = np.min(quiescent_energies)
            for state_index, energy in enumerate(quiescent_energies):
                if energy == energy_min:
                    boltzmann_factors[state_index] = 1

        # Normalise rate by T1
        markov_matrix = np.empty(
            quiescent_states.shape, dtype=meta_datatype)
        for state_index, boltzmann_factor in enumerate(boltzmann_factors):
            markov_matrix[state_index, :] = boltzmann_factors
            markov_matrix[state_index, state_index] -= \
                np.sum(boltzmann_factors)
        norm = np.linalg.norm(markov_matrix, ord=2)
        boltzmann_factors /= norm*thermalisation_time
        boltzmann_factors = np.sqrt(boltzmann_factors)

        # Generate the jump operators
        for state_index_init in range(len(boltzmann_factors)):
            for state_index_final, boltzmann_factor in \
                    enumerate(boltzmann_factors):
                jump_temp = boltzmann_factor*np.outer(
                    quiescent_states[:, state_index_final],
                    np.conj(quiescent_states[:, state_index_init]),
                )
                jump = np.empty(
                    (jump_temp.shape[0], jump_temp.shape[1], 2),
                    dtype=meta_datatype
                )
                jump[:, :, 0] = np.real(jump_temp)
                jump[:, :, 1] = np.imag(jump_temp)

                _record_operator(
                    f"L{spin_label}1"
                    + f"{state_index_final} {state_index_init}", jump,
                    atom, [temp_labels, operator_labels, dissipator_labels]
                )
    return temp_labels


def _add_hyperfine(
        electron_spin: float, nuclear_spin: float, atom: dict,
        label_sets: dict[str, set[str]]):
    """
        Add hyperfine interactions to the atom.
    """

    operator_labels = label_sets["operator_labels"]
    zero_field_labels = label_sets["zero_field_labels"]
    tensor_labels = label_sets["tensor_labels"]

    if electron_spin > 0 and nuclear_spin > 0:
        a_hyperfine = np.zeros((3, 3), dtype=meta_datatype)
        if electron_spin > 0:
            if "A" in atom.keys():
                a_iso = atom["A"]
                if "A_dipole" in atom.keys():
                    a_dipole = atom["A_dipole"]
                elif "A_perp" in atom.keys():
                    a_perp = atom["A_perp"]
                    a_dipole = (a_iso - a_perp)/3
                    a_iso -= 2*a_dipole
                else:
                    a_dipole = 0
            else:
                a_iso = 2
                a_dipole = 0

            a_hyperfine[0, 0] = a_iso - a_dipole
            a_hyperfine[1, 1] = a_iso - a_dipole
            a_hyperfine[2, 2] = a_iso + 2*a_dipole

            atom["Aten"] = a_hyperfine
            tensor_labels.add("Aten")

            electron_spin_vec = _read_spin_vec("S", atom)
            nuclear_spin_vec = _read_spin_vec("I", atom)
            quadratic = _quadratic_outer(
                electron_spin_vec, nuclear_spin_vec)
            _record_spin_quadratic(
                "S", "I", quadratic, atom, [operator_labels])

            (
                (hyperfine_xx, hyperfine_xy, hyperfine_xz),
                (hyperfine_yx, hyperfine_yy, hyperfine_yz),
                (hyperfine_zx, hyperfine_zy, hyperfine_zz)
            ) = quadratic

            hyperfine_vec = _add_vec(electron_spin_vec, nuclear_spin_vec)
            _record_spin_vec(
                "F", hyperfine_vec, atom, [operator_labels])

            hyperfine_generator = _quadratic_transform(
                a_hyperfine, quadratic)
            _record_operator(
                "HA", hyperfine_generator, atom,
                [operator_labels, zero_field_labels]
            )


def _combine_in_atom(
        electron_spin: float, nuclear_spin: float, atom: dict,
        label_sets: dict[str, set[str]]):
    """
        Zeeman and ZFS.
    """

    operator_labels = label_sets["operator_labels"]
    field_labels = label_sets["field_labels"]
    zero_field_labels = label_sets["zero_field_labels"]

    # Magnetic generators
    if electron_spin > 0:
        spin_gyro = atom["gyroS"]

        spin_vec = _read_spin_vec("S", atom)
        electron_generator_vec = _linear_transform(
            spin_gyro, spin_vec)

    if nuclear_spin > 0:
        spin_gyro = atom["gyroI"]

        spin_vec = _read_spin_vec("I", atom)
        nuclear_generator_vec = _linear_transform(
            spin_gyro, spin_vec)

    if nuclear_spin > 0:
        if electron_spin > 0:
            generator_vec = _add_vec(
                electron_generator_vec, nuclear_generator_vec)
        else:
            generator_vec = nuclear_generator_vec
    elif electron_spin > 0:
        generator_vec = electron_generator_vec
    else:
        generator_vec = None

    if generator_vec is not None:
        _record_spin_vec(
            "G", generator_vec, atom,
            [operator_labels, field_labels]
        )

    # Zero-field total
    zfs_generator = None
    for label in zero_field_labels:
        if label in atom.keys():
            if zfs_generator is None:
                zfs_generator = atom[label].copy()
            else:
                zfs_generator += atom[label]
    if zfs_generator is not None:
        atom["Z"] = zfs_generator
        operator_labels.add("Z")

    # Quiescent
    if zfs_generator is not None:
        quiescent_generator = zfs_generator.copy()
    else:
        quiescent_generator = None
    if "B0" in atom.keys() and generator_vec is not None:
        quiescent_magnetic_field = atom["B0"]
        quiescent_magnetic_generator = \
            quiescent_magnetic_field[0]*generator_vec[0] \
            + quiescent_magnetic_field[1]*generator_vec[1] \
            + quiescent_magnetic_field[2]*generator_vec[2]
        if quiescent_generator is None:
            quiescent_generator = quiescent_magnetic_generator
        else:
            quiescent_generator += quiescent_magnetic_generator
    if quiescent_generator is not None:
        atom["H0"] = quiescent_generator
        operator_labels.add("H0")


def _add_spin_spin_coupling(
        block: list[dict], j_description: dict, index_a: int, index_b: int,
        label_sets: dict[str, set[str]]):
    """
    Adds spin-spin coupling to the model.
    """

    operator_labels = label_sets["operator_labels"]
    tensor_labels = label_sets["tensor_labels"]

    j_tensor = np.zeros((3, 3), dtype=meta_datatype)
    if "J" in j_description.keys():
        j_iso = j_description["J"]
        if "J_dipole" in j_description.keys():
            j_dipole = j_description["J_dipole"]
        elif "J_perp" in j_description.keys():
            j_perp = j_description["J_perp"]
            j_dipole = (j_iso - j_perp)/3
            j_iso -= 2*j_dipole
        else:
            j_dipole = 0
    else:
        j_iso = 2
        j_dipole = 0

    j_tensor[0, 0] = j_iso - j_dipole
    j_tensor[1, 1] = j_iso - j_dipole
    j_tensor[2, 2] = j_iso + 2*j_dipole

    j_description["Jten"] = j_tensor
    tensor_labels.add("Jten")

    atom_a = block[index_a]
    atom_b = block[index_b]

    for label in ["S", "I"]:
        if label in atom_a.keys():
            label_a = label
            break
    for label in ["S", "I"]:
        if label in atom_b.keys():
            label_b = label
            break

    spin_a_vec = _read_spin_vec(label_a, atom_a)
    spin_b_vec = _read_spin_vec(label_b, atom_b)

    quadratic = _quadratic_outer(spin_a_vec, spin_b_vec)
    _record_spin_quadratic(
        label_a, label_b, quadratic,
        j_description, [operator_labels]
    )

    spin_generator = _quadratic_transform(j_tensor, quadratic)
    _record_operator(
        "HJ", spin_generator, j_description, [operator_labels])


def _remove_trace(description: list[dict], operator_labels: set[str]):
    """
        Remove trace from all operators.
    """

    for block in description:
        for atom in block:
            for operator_label in operator_labels:
                if "L" not in operator_label and "|" not in operator_label:
                    if operator_label in atom.keys():
                        operator = atom[operator_label]
                        trace = np.trace(operator[:, :, 0])
                        operator_new = operator.copy()
                        operator_new[:, :, 0] -= \
                            trace*np.eye(operator.shape[0])/operator.shape[0]
                        atom[operator_label] = operator_new


def _combine_coherent_atoms(
        description: list[dict], atom_interactions: list[dict],
        field_labels: set[str]) -> list:
    """
        Combine all operators in block.
    """

    block_operator_list = []
    for block_index, (block, atom_interaction) \
            in enumerate(zip(description, atom_interactions)):
        block_operator_dict = {}
        for atom_index, atom in enumerate(block):
            for operator_label in field_labels:
                if operator_label in atom.keys():
                    if operator_label not in block_operator_dict.keys():
                        block_operator_dict[operator_label] = \
                            atom[operator_label].copy()
                    else:
                        block_operator_dict[operator_label] += \
                            atom[operator_label]
            for operator_label in ["Z", "H0"]:
                if operator_label in atom.keys():
                    if operator_label not in block_operator_dict.keys():
                        block_operator_dict[operator_label] = \
                            atom[operator_label].copy()
                    else:
                        block_operator_dict[operator_label] += \
                            atom[operator_label]

        for (index_a, index_b), j_description in atom_interaction.items():
            if "HJ" in j_description.keys():
                for operator_label in ["Z", "H0"]:
                    if operator_label not in block_operator_dict.keys():
                        block_operator_dict[operator_label] = \
                            j_description["HJ"].copy()
                    else:
                        block_operator_dict[operator_label] += \
                            j_description["HJ"]
        block_operator_list.append(block_operator_dict)

    return block_operator_list


def _combine_coherent_blocks(coherent_atoms: list[dict]):

    coherent_block = {}
    for coherent_atom in coherent_atoms:
        for label, operator in coherent_atom.items():
            if label not in coherent_block.keys():
                coherent_block[label] = operator.copy()
            else:
                coherent_block[label] += operator
    return coherent_block


def _list_operators(
    description: list[list[dict]], atom_interactions: list[dict],
    block_interactions: dict, label_sets: dict[str, set[str]]
) -> dict[str, dict[str, np.ndarray]]:
    """
        Put all generated operators into global lists.
    """

    operator_labels = label_sets["operator_labels"]
    field_labels = label_sets["field_labels"]
    zero_field_labels = label_sets["zero_field_labels"]
    tensor_labels = label_sets["tensor_labels"]
    dissipator_labels = label_sets["dissipator_labels"]

    operator_dict = {}
    jump_dict = {}
    composite_operator_dict = {}
    tensor_dict = {}
    for block_index, (block, atom_interaction) in \
            enumerate(zip(description, atom_interactions)):
        for atom_index, atom in enumerate(block):
            # All
            for operator_label in operator_labels:
                if operator_label in atom.keys():
                    operator_name = f"[{block_index}, {atom_index}]" \
                                    + f" {operator_label}"
                    operator_dict[operator_name] = atom[operator_label]

            # Jump
            for operator_label in dissipator_labels:
                if operator_label in atom.keys():
                    operator_name = f"[{block_index}, {atom_index}]" \
                                    + f" {operator_label}"
                    jump_dict[operator_name] = atom[operator_label]

            # Composite
            for operator_label in zero_field_labels:
                if operator_label in atom.keys():
                    operator_name = f"[{block_index}, {atom_index}]" \
                                    + f" {operator_label}"
                    composite_operator_dict[operator_name] = \
                        atom[operator_label]
            for operator_label in field_labels:
                if operator_label in atom.keys():
                    operator_name = f"[{block_index}, {atom_index}]" \
                                    + f" {operator_label}"
                    composite_operator_dict[operator_name] = \
                        atom[operator_label]
            for operator_label in ["Z", "H0"]:
                if operator_label in atom.keys():
                    operator_name = f"[{block_index}, {atom_index}]" \
                                    + f" {operator_label}"
                    composite_operator_dict[operator_name] = \
                        atom[operator_label]

            # Tensors
            for tensor_label in tensor_labels:
                if tensor_label in atom.keys():
                    tensor_name = f"[{block_index}, {atom_index}]" \
                                    + f" {tensor_label}"
                    tensor_dict[tensor_name] = atom[tensor_label]

        for (index_a, index_b), j_description in atom_interaction.items():
            # Operators
            for operator_label in operator_labels:
                if operator_label in j_description.keys():
                    operator_name = f"[{block_index}, {index_a}, {index_b}]" \
                        + f" {operator_label}"
                    operator_dict[operator_name] = \
                        j_description[operator_label]

            # Tensor
            for tensor_label in tensor_labels:
                if tensor_label in j_description.keys():
                    tensor_name = f"[{block_index}, {index_a}, {index_b}]" \
                        + f" {tensor_label}"
                    tensor_dict[tensor_name] = j_description[tensor_label]

    for (
        (block_index_a, atom_index_a),
        (block_index_b, atom_index_b)
    ), block_interaction in block_interactions.items():
        # Operators
        for operator_label in operator_labels:
            if operator_label in block_interaction.keys():
                operator_name = f"[{block_index_a}, {atom_index_a}," \
                    + f" {block_index_b}, {atom_index_b}] {operator_label}"
                operator_dict[operator_name] = \
                    block_interaction[operator_label]

        # Operators
        for operator_label in dissipator_labels:
            if operator_label in block_interaction.keys():
                operator_name = \
                    f"[{block_index_a}, {atom_index_a}," \
                    + f" {block_index_b}, {atom_index_b}] {operator_label}"
                jump_dict[operator_name] = block_interaction[operator_label]

    operator_dicts = {
            "all": operator_dict,
            "jump": jump_dict,
            "composite": composite_operator_dict,
            "tensor": tensor_dict
    }
    return operator_dicts


def _mult(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
        Multiply two complex matrices.
    """

    operator_out = np.empty_like(left)
    operator_out[:, :, 0] = left[:, :, 0]@right[:, :, 0] \
        - left[:, :, 1]@right[:, :, 1]
    operator_out[:, :, 1] = left[:, :, 0]@right[:, :, 1] \
        + left[:, :, 1]@right[:, :, 0]
    return operator_out


def add_spin(spin: int, hilbert_space_shape: tuple) -> tuple[np.ndarray]:
    """
        Add a new spin system to the Hilbert/operator space.
        See Sakurai 3ed Section 3.5.3.
    """

    spin_dimension = int(2*spin + 1)
    hilbert_space_shape.append(spin_dimension)

    magnetic = np.linspace(
        spin, -spin, spin_dimension,
        dtype=meta_datatype
    )
    diag_p = np.sqrt(
        (spin - magnetic)*(1 + spin + magnetic))[1:]
    diag_m = np.sqrt(
        (spin + magnetic)*(1 + spin - magnetic))[:-1]
    spin_p = np.diag(diag_p, 1)
    spin_m = np.diag(diag_m, -1)

    spin_x = np.zeros(
        (spin_p.shape[0], spin_p.shape[1], 2),
        dtype=meta_datatype
    )
    spin_x[:, :, 0] = (spin_p + spin_m)/2

    spin_y = np.zeros_like(spin_x)
    spin_y[:, :, 1] = (spin_m - spin_p)/2

    spin_z = np.zeros_like(spin_x)
    spin_z[:, :, 0] = np.diag(magnetic)

    spin_identity = np.zeros_like(spin_x)
    spin_identity[:, :, 0] = np.eye(spin_x.shape[0])

    projectors = {}
    for magnetic_index, magnetic_number in \
            enumerate(np.arange(-spin, spin + 0.1)):
        projector = np.zeros_like(spin_identity)
        projector[magnetic_index, magnetic_index, 0] = 1
        if magnetic_number != 0:
            magnetic_number *= -1
        if np.isclose(np.fmod(spin, 1), 0):
            key = f"|{magnetic_number:.0f})({magnetic_number:.0f}|"
        else:
            key = f"|{magnetic_number:.1f})({magnetic_number:.1f}|"
        projectors[key] = projector

    return (spin_x, spin_y, spin_z), projectors, spin_identity


def _product_spin_state(
        previous_identity: np.ndarray, spin_identity: np.ndarray,
        block: list, atom_current: dict, temp_labels: set,
        operator_labels: set, dissipator_labels: set) -> np.ndarray:
    if previous_identity is not None:
        operator_labels_add = set()
        for atom in block:
            ignore = []
            for operator_label in operator_labels:
                if operator_label in atom.keys() \
                        and operator_label not in ignore:
                    operator = atom[operator_label]
                    if operator_label in dissipator_labels:
                        atom.pop(operator_label)
                        if operator_label in temp_labels and \
                                atom is atom_current:
                            operators_new = kroneker_jump_inner(
                                operator, previous_identity)
                        else:
                            operators_new = kroneker_jump_outer(
                                spin_identity, operator)
                        for index, operator_new in enumerate(operators_new):
                            label_extend = f"{operator_label} {index}"
                            atom[label_extend] = operator_new
                            operator_labels_add.add(label_extend)
                            ignore.append(label_extend)

                    else:
                        if operator_label in temp_labels and \
                                atom is atom_current:
                            operator_new = kroneker_product(
                                operator, previous_identity)
                        else:
                            operator_new = kroneker_product(
                                spin_identity, operator)
                        atom[operator_label] = operator_new

        operator_labels |= operator_labels_add
        dissipator_labels |= operator_labels_add

        spin_identity = kroneker_product(spin_identity, previous_identity)

    return spin_identity


def kroneker_product(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """
        Take the Kroneker product between two operators.
    """

    product = np.empty(
        (
            outer.shape[0]*inner.shape[0], outer.shape[1]*inner.shape[1],
            outer.shape[2]
        ),
        dtype=meta_datatype
    )

    for outer_index_y in range(outer.shape[0]):
        for outer_index_x in range(outer.shape[1]):
            product[
                outer_index_y*inner.shape[0]
                :(outer_index_y + 1)*inner.shape[0],
                outer_index_x*inner.shape[1]
                :(outer_index_x + 1)*inner.shape[1], 0
            ] = outer[outer_index_y, outer_index_x, 0]*inner[:, :, 0] \
                - outer[outer_index_y, outer_index_x, 1]*inner[:, :, 1]
            product[
                outer_index_y*inner.shape[0]
                :(outer_index_y + 1)*inner.shape[0],
                outer_index_x*inner.shape[1]
                :(outer_index_x + 1)*inner.shape[1], 1
            ] = outer[outer_index_y, outer_index_x, 0]*inner[:, :, 1] \
                + outer[outer_index_y, outer_index_x, 1]*inner[:, :, 0]
    return product


def kroneker_jump_inner(
        inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    products = []

    for outer_index in range(outer.shape[0]):
        product = np.zeros(
            (
                outer.shape[0]*inner.shape[0],
                outer.shape[1]*inner.shape[1], outer.shape[2]
            ),
            dtype=meta_datatype
        )

        product[
            outer_index*inner.shape[0]
            :(outer_index + 1)*inner.shape[0],
            outer_index*inner.shape[1]
            :(outer_index + 1)*inner.shape[1], :
        ] = inner

        if not np.isclose(np.sum(product**2), 0):
            products.append(product)

    return products


def kroneker_jump_outer(
        inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    products = []

    for inner_index in range(outer.shape[0]):
        product = np.zeros(
            (
                outer.shape[0]*inner.shape[0],
                outer.shape[1]*inner.shape[1], outer.shape[2]
            ),
            dtype=meta_datatype
        )

        product[
            inner_index::inner.shape[0],
            inner_index::inner.shape[1], :
        ] = outer

        if not np.isclose(np.sum(product**2), 0):
            products.append(product)

    return products


def _direct_sum(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """
        Combining incoherent systems
    """

    sum_size = upper.shape[0] + lower.shape[0]
    direct_sum = np.zeros((sum_size, sum_size, 2), dtype=meta_datatype)
    direct_sum[:upper.shape[0], :upper.shape[0], :] = upper
    direct_sum[upper.shape[0]:, upper.shape[0]:] = lower
    return direct_sum


def _add_vec(left: tuple, right: tuple) -> tuple:
    """
        Add two vector operators together.
    """

    out = []
    for left_spin, right_spin in zip(left, right):
        out.append(left_spin + right_spin)
    return tuple(out)


def _linear_transform(trans: np.ndarray, inp: tuple) -> tuple:
    """
        Apply a linear transform to a spin vector.
        The typical case would be the g tensor.
    """

    out = [None]*3
    for y_index in range(3):
        for x_index in range(3):
            if out[y_index] is None:
                out[y_index] = trans[y_index, x_index]*inp[x_index]
            else:
                out[y_index] += trans[y_index, x_index]*inp[x_index]
    return tuple(out)


def _quadratic_outer(left: tuple, right: tuple) -> tuple:
    """
        Generate all products of two vector operators.
    """

    quadratic = []
    for y_index in range(3):
        sub_quadratic = []
        for x_index in range(3):
            sub_quadratic.append(_mult(left[y_index], right[x_index]))
        quadratic.append(tuple(sub_quadratic))
    return tuple(quadratic)


def _quadratic_transform(
        trans: np.ndarray, quadratic: tuple) -> np.ndarray:
    """
        Apply a transform to a quadratic expansion.
        Cases include the D, P, A and J tensors.
    """

    operator = None
    for y_index in range(3):
        for x_index in range(3):
            if operator is None:
                operator = trans[y_index, x_index]*quadratic[y_index][x_index]
            else:
                operator += trans[y_index, x_index]*quadratic[y_index][x_index]
    return operator


def _record_operator(
        label: str, operator: np.ndarray, atom: dict, label_sets: list):
    """
        Write a single operator to a dictionary.
    """

    atom[label] = operator
    for label_set in label_sets:
        label_set.add(label)


def _record_spin_vec(
        label: str, spin_vec: tuple, atom: dict, label_sets: list):
    """
        Write a spin vector to an atom dictionary.
    """

    directions = ("x", "y", "z")
    for spin_operator, direction in zip(spin_vec, directions):
        operator_label = label + direction
        atom[operator_label] = spin_operator
        for label_set in label_sets:
            label_set.add(operator_label)


def _read_spin_vec(label: str, atom: dict) -> tuple:
    """
        Read a spin vector from a dictionary.
    """

    directions = ("x", "y", "z")
    spin_vec = []
    for direction in directions:
        operator_label = label + direction
        if operator_label in atom.keys():
            spin_vec.append(atom[operator_label])
        else:
            raise KeyError(
                f"Label \"{operator_label}\" not found in dict.")
    return tuple(spin_vec)


def _record_spin_quadratic(
        label_left: str, label_right: str,  quadratic: tuple,
        atom: dict, label_sets: list):
    """
        Write a quadratic expansion of a spin vector to an atom dictionary.
    """

    # directions = ("x", "y", "z")
    # for quadratic_vec, direction_left in zip(quadratic, directions):
    #     operator_label_left = label_left + direction_left
    #     for quadratic_operator, direction_right in \
    #             zip(quadratic_vec, directions):

    #         operator_label = operator_label_left + " " \
    #             + label_right + direction_right
    #         atom[operator_label] = quadratic_operator
    #         for label_set in label_sets:
    #             label_set.add(operator_label)

    return


def _combine_blocks(
        description: list[list[dict]], coherent_atoms: list[dict],
        label_sets: dict[str, set[str]]):
    """
        Direct sum the incoherent blocks together.
    """

    projector_labels = label_sets["projector_labels"]
    dissipator_labels = label_sets["dissipator_labels"]

    combined_size = 0
    combined_zero = None
    combined_allowed = None
    for block_index, (block, coherent_atom) in \
            enumerate(zip(description, coherent_atoms)):
        current_size = 0
        current_zero = None
        current_allowed = None
        if combined_size > 0:
            combined_zero = np.zeros(
                (combined_size, combined_size, 2),
                dtype=meta_datatype
            )
        for atom_index, atom in enumerate(block):

            for operator_label in dissipator_labels:
                if operator_label in atom.keys():
                    current_size, current_zero, current_allowed = \
                        _combine_blocks_operator(
                            operator_label, atom, current_size,
                            current_zero, current_allowed, combined_size,
                            combined_zero
                        )

            for operator_label in projector_labels:
                if operator_label in atom.keys():
                    current_size, current_zero, current_allowed = \
                        _combine_blocks_operator(
                            operator_label, atom, current_size,
                            current_zero, current_allowed, combined_size,
                            combined_zero
                        )

        for operator_label in coherent_atom.keys():
            current_size, current_zero, current_allowed = \
                _combine_blocks_operator(
                    operator_label, coherent_atom, current_size,
                    current_zero, current_allowed, combined_size,
                    combined_zero
                )

        if current_size == 0:
            current_size = 1
            current_zero = np.zeros(
                (current_size, current_size, 2), dtype=meta_datatype)
            current_allowed = np.zeros_like(current_zero)
            current_allowed[:, :, 0] = np.ones(
                (current_size, current_size), dtype=meta_datatype)

        for block_previous_index, (block_previous, coherent_atom_previous) \
                in enumerate(zip(description, coherent_atoms)):
            if block_previous_index < block_index:
                for atom_previous in block_previous:

                    for operator_label in dissipator_labels:
                        if operator_label in atom_previous.keys():
                            _combine_blocks_operator_previous(
                                operator_label, atom_previous, current_zero
                            )

                    for operator_label in projector_labels:
                        if operator_label in atom_previous.keys():
                            _combine_blocks_operator_previous(
                                operator_label, atom_previous, current_zero
                            )

                    for operator_label in coherent_atom_previous.keys():
                        _combine_blocks_operator_previous(
                            operator_label, coherent_atom_previous,
                            current_zero
                        )

        if combined_size == 0:
            combined_allowed = current_allowed
        else:
            combined_allowed = _direct_sum(
                combined_allowed, current_allowed)
        combined_size += current_size

    return combined_allowed[:, :, 0]


def _combine_blocks_operator(
        operator_label: str, atom: dict, current_size: int,
        current_zero: np.ndarray, current_allowed: np.ndarray,
        combined_size: int, combined_zero: np.ndarray):

    operator = atom[operator_label]
    if current_size == 0:
        current_size = operator.shape[0]
        current_zero = np.zeros(
            (current_size, current_size, 2),
            dtype=meta_datatype
        )
        current_allowed = np.zeros_like(current_zero)
        current_allowed[:, :, 0] = np.ones(
            (current_size, current_size),
            dtype=meta_datatype
        )

    if combined_size > 0:
        operator = _direct_sum(combined_zero, operator)
    atom[operator_label] = operator

    return current_size, current_zero, current_allowed


def _combine_blocks_operator_previous(
        operator_label: str, atom_previous: dict,
        current_zero: np.ndarray):

    operator = atom_previous[operator_label]
    operator = _direct_sum(operator, current_zero)
    atom_previous[operator_label] = operator


def _add_block_interaction(
        block_a: int, atom_a: int, block_b: int, atom_b: int,
        interaction: dict, description: list[list[dict]],
        label_sets: dict[str, set[str]]):
    """
        Add incoherent interactions between blocks.
    """

    atom_a_dict = description[block_a][atom_a]
    atom_b_dict = description[block_b][atom_b]

    spin_a = None
    spin_b = None
    if "S" in atom_a_dict.keys():
        spin_a = atom_a_dict["S"]
    if "S" in atom_b_dict.keys():
        spin_b = atom_b_dict["S"]

    if spin_a is not None and spin_b is not None:
        if spin_a == spin_b:
            # Optical
            if "rel_n" in interaction.keys():
                relaxation_rate_nonconserve = interaction["rel_n"]
            else:
                relaxation_rate_nonconserve = None
            if "rel" in interaction.keys():
                relaxation_rate = interaction["rel"]
            else:
                return

            _couple_optical(
                atom_a_dict, atom_b_dict, spin_a, spin_b, interaction,
                relaxation_rate, relaxation_rate_nonconserve, label_sets
            )

        elif spin_a == 1 and spin_b == 0:
            # ISC excited
            if "s_gets_0" in interaction.keys():
                relaxation_rate_0 = interaction["s_gets_0"]
            if "s_gets_1" in interaction.keys():
                relaxation_rate_1 = interaction["s_gets_1"]
            if relaxation_rate_0 is not None or relaxation_rate_1 is not None:
                _couple_isc_excited(
                    atom_a_dict, atom_b_dict, relaxation_rate_0,
                    relaxation_rate_1, interaction, label_sets
                )

        elif spin_a == 0 and spin_b == 1:
            # ISC ground
            if "0_gets_s" in interaction.keys():
                relaxation_rate_0 = interaction["0_gets_s"]
            if "1_gets_s" in interaction.keys():
                relaxation_rate_1 = interaction["1_gets_s"]
            if relaxation_rate_0 is not None or relaxation_rate_1 is not None:
                _couple_isc_ground(
                    atom_a_dict, atom_b_dict, relaxation_rate_0,
                    relaxation_rate_1, interaction, label_sets
                )


def _couple_optical(
        atom_a_dict: dict, atom_b_dict: dict, spin_a: float, spin_b: float,
        interaction: dict, relaxation_rate: float,
        relaxation_rate_nonconserve: float,
        label_sets: dict[str, set[str]]):
    """
        Add NV laser excitations and decay.
    """

    operator_labels = label_sets["operator_labels"]
    dissipator_labels = label_sets["dissipator_labels"]

    spin = spin_a
    for magnetic_number in np.arange(-spin, spin + 0.1):
        magnetic_number, magnetic_label, operator_label_partial = \
            _get_spin_labels(spin, magnetic_number)
        operator_label = f"{operator_label_partial} {operator_label_partial}"

        raise_conserves = _couple_incoherent(
            atom_a_dict[magnetic_label],
            atom_b_dict[magnetic_label],
            np.zeros_like(atom_a_dict[magnetic_label])
        )
        for index, raise_conserve in enumerate(raise_conserves):
            raise_conserve *= math.sqrt(relaxation_rate)
            _record_operator(
                f"Lrc{operator_label} {index}", raise_conserve,
                interaction, [operator_labels, dissipator_labels]
            )

        lower_conserves = _couple_incoherent(
            atom_b_dict[magnetic_label],
            atom_a_dict[magnetic_label],
            np.zeros_like(atom_a_dict[magnetic_label])
        )
        for index, lower_conserve in enumerate(lower_conserves):
            lower_conserve *= math.sqrt(relaxation_rate)
            _record_operator(
                f"Llc{operator_label} {index}", lower_conserve,
                interaction, [operator_labels, dissipator_labels]
            )

        if relaxation_rate_nonconserve is None:
            continue

        for magnetic_number_excited in np.arange(-spin, spin + 0.1):

            if np.isclose(
                    abs(magnetic_number - magnetic_number_excited), 1):
                magnetic_number_excited, magnetic_label_excited, \
                    operator_label_partial_excited = _get_spin_labels(
                        spin, magnetic_number_excited
                    )
                operator_label_raise = operator_label_partial_excited + " " \
                    + operator_label_partial
                operator_label_lower = operator_label_partial + " " \
                    + operator_label_partial_excited

                raise_nonconserves = _couple_incoherent(
                    atom_a_dict[magnetic_label],
                    atom_b_dict[magnetic_label_excited],
                    np.zeros_like(atom_a_dict[magnetic_label])
                )
                for index, raise_nonconserve in enumerate(
                        raise_nonconserves):
                    if np.isclose(abs(magnetic_number), spin):
                        raise_nonconserve *= math.sqrt(
                            relaxation_rate_nonconserve/2)
                    else:
                        raise_nonconserve *= math.sqrt(
                            relaxation_rate_nonconserve)
                    _record_operator(
                        f"Lrn{operator_label_raise} {index}",
                        raise_nonconserve, interaction,
                        [operator_labels, dissipator_labels]
                    )

                lower_nonconserves = _couple_incoherent(
                    atom_b_dict[magnetic_label_excited],
                    atom_a_dict[magnetic_label],
                    np.zeros_like(atom_a_dict[magnetic_label])
                )
                for index, lower_nonconserve in enumerate(
                        lower_nonconserves):
                    if np.isclose(
                            abs(magnetic_number_excited), spin):
                        lower_nonconserve *= math.sqrt(
                            relaxation_rate_nonconserve/2)
                    else:
                        lower_nonconserve *= math.sqrt(
                            relaxation_rate_nonconserve)
                    _record_operator(
                        f"Lln{operator_label_lower} {index}",
                        lower_nonconserve, interaction,
                        [operator_labels, dissipator_labels]
                    )


def _couple_isc_excited(
        atom_a_dict: dict, atom_b_dict: dict, relaxation_rate_0: float,
        relaxation_rate_1: float, interaction: dict,
        label_sets: dict[str, dict[str, np.ndarray]]):

    operator_labels = label_sets["operator_labels"]
    dissipator_labels = label_sets["dissipator_labels"]

    spin = 1
    for magnetic_number in np.arange(-spin, spin + 0.1):
        magnetic_number, magnetic_label, operator_label_triplet = \
            _get_spin_labels(spin, magnetic_number)
        _, magnetic_label_singlet, _ = _get_spin_labels(
            0, 0)
        operator_label = f"s {operator_label_triplet}"

        dissipators = _couple_incoherent(
            atom_a_dict[magnetic_label],
            atom_b_dict[magnetic_label_singlet],
            np.zeros_like(atom_a_dict[magnetic_label])
        )

        for index, dissipator in enumerate(dissipators):
            if np.isclose(magnetic_number, 0):
                if relaxation_rate_0 is None:
                    break
                dissipator *= np.sqrt(relaxation_rate_0)
            else:
                if relaxation_rate_1 is None:
                    break
                dissipator *= np.sqrt(relaxation_rate_1)
            _record_operator(
                f"Lisc{operator_label} {index}",
                dissipator, interaction,
                [operator_labels, dissipator_labels]
            )


def _couple_isc_ground(
        atom_a_dict: dict, atom_b_dict: dict, relaxation_rate_0: float,
        relaxation_rate_1: float, interaction: dict,
        label_sets: dict[str, dict[str, np.ndarray]]):

    operator_labels = label_sets["operator_labels"]
    dissipator_labels = label_sets["dissipator_labels"]

    spin = 1
    for magnetic_number in np.arange(-spin, spin + 0.1):
        magnetic_number, magnetic_label, operator_label_triplet = \
            _get_spin_labels(spin, magnetic_number)
        _, magnetic_label_singlet, _ = _get_spin_labels(
            0, 0)
        operator_label = f"{operator_label_triplet} s"

        dissipators = _couple_incoherent(
            atom_a_dict[magnetic_label_singlet],
            atom_b_dict[magnetic_label],
            np.zeros_like(atom_b_dict[magnetic_label])
        )

        for index, dissipator in enumerate(dissipators):
            if np.isclose(magnetic_number, 0):
                if relaxation_rate_0 is None:
                    break
                dissipator *= np.sqrt(relaxation_rate_0)
            else:
                if relaxation_rate_1 is None:
                    break
                dissipator *= np.sqrt(relaxation_rate_1)
            _record_operator(
                f"Lisc{operator_label} {index}",
                dissipator, interaction,
                [operator_labels, dissipator_labels]
            )


def _couple_incoherent(
        initial_projector: np.ndarray, final_projector: np.ndarray,
        template: np.ndarray) -> list[np.ndarray]:
    initial = np.sum(initial_projector, axis=(0, 2))
    final = np.sum(final_projector, axis=(0, 2))

    if np.isclose(np.sum(initial), np.sum(final)):
        out = []
        indices_initial = np.where(initial > 0)[0]
        indices_excited = np.where(final > 0)[0]
        for index_initial, index_final in \
                zip(indices_initial, indices_excited):
            superoperator = np.zeros_like(template)
            superoperator[index_final, index_initial, 0] = 1
            out.append(superoperator)
    else:
        superoperator = np.zeros_like(template)
        superoperator[:, :, 0] = np.outer(final, initial)
        out = [superoperator]
    return out


def _get_spin_labels(
        spin: float, magnetic_number: float) -> [float, str, str]:
    if magnetic_number != 0:
        magnetic_number *= -1
    if np.isclose(math.fmod(spin, 1), 0):
        magnetic_label = f"S|{magnetic_number:.0f})({magnetic_number:.0f}|"
        operator_label = f"{magnetic_number:.0f}"
    else:
        magnetic_label = f"S|{magnetic_number:.1f})({magnetic_number:.1f}|"
        operator_label = f"{magnetic_number:.1f}"

    return magnetic_number, magnetic_label, operator_label


def _plot_operators(operator_dicts: dict[str, dict[str, np.ndarray]]):
    """
        Generates heat maps of all the operators and tensors generated.
    """

    print("Plotting")

    interesting_operators = ["dissipators", "coherent", "generators"]
    for interesting_operator in interesting_operators:
        operator_dict = operator_dicts[interesting_operator]
        if interesting_operator in ["dissipators", "generators"]:
            number_of_columns_min = 1
            plot_extension_scale = 1
        else:
            number_of_columns_min = 6
            plot_extension_scale = 12

        plt.figure(
            figsize=
            (6.4, max(len(operator_dict)*4.8/plot_extension_scale, 1)),
            label=interesting_operator
        )
        plot_columns = min(
            number_of_columns_min,
            int(math.ceil(math.sqrt(len(operator_dict))))
        )
        plot_rows = math.ceil(len(operator_dict)/plot_columns)
        for operator_index, (operator_name, operator) in \
                enumerate(operator_dict.items()):
            plt.subplot(plot_rows, plot_columns, operator_index + 1)
            plt.imshow(colour_complex_matrix(
                operator/(2*np.max(np.abs(operator)))),
                       interpolation="nearest")
            plt.title(operator_name)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
        plt.draw()

    # plt.figure(
    #     figsize=(6.4, 24*4.8),
    #     label="operators"
    # )
    # plot_columns = \
    #     min(4, int(math.ceil(math.sqrt(len(operator_dict)))))
    # plot_rows = math.ceil(len(operator_dict)/plot_columns)
    # for operator_index, (operator_name, operator) in \
    #         enumerate(operator_dict.items()):
    #     plt.subplot(plot_rows, plot_columns, operator_index + 1)
    #     plt.imshow(colour_complex_matrix(
    #         operator/(2*np.max(np.abs(operator)))))
    #     plt.title(operator_name)
    #     plt.xticks([], [])
    #     plt.yticks([], [])
    #     plt.tight_layout()
    # plt.draw()

    # if superoperator_dict is not None:
    #     plt.figure(
    #         figsize=(6.4, 96*4.8),
    #         label="superoperators"
    #     )
    #     plot_columns = \
    #         min(2, int(math.ceil(math.sqrt(len(superoperator_dict)))))
    #     plot_rows = math.ceil(len(superoperator_dict)/plot_columns)
    #     for operator_index, (operator_name, operator) in \
    #             enumerate(superoperator_dict.items()):
    #         plt.subplot(plot_rows, plot_columns, operator_index + 1)
    #         plt.imshow(colour_complex_matrix(
    #             operator/(np.max(np.abs(operator)))))
    #         plt.title(operator_name)
    #         plt.xticks([], [])
    #         plt.yticks([], [])
    #         plt.tight_layout()
    #     plt.draw()

    # # plt.figure(
    # #     figsize=(6.4, 6.4),
    # #     label="composite_operators"
    # # )
    # # plot_columns = \
    # #     min(4, int(math.ceil(math.sqrt(len(composite_operator_dict)))))
    # # plot_rows = math.ceil(len(composite_operator_dict)/plot_columns)
    # # for operator_index, (operator_name, operator) in \
    # #         enumerate(composite_operator_dict.items()):
    # #     plt.subplot(plot_rows, plot_columns, operator_index + 1)
    # #     plt.imshow(colour_complex_matrix(
    # #         operator/(2*np.max(np.abs(operator)))))
    # #     plt.title(operator_name)
    # #     plt.xticks([], [])
    # #     plt.yticks([], [])
    # # plt.draw()

    # # plt.figure(
    # #     figsize=(6.4, 6.4),
    # #     label="blocks"
    # # )
    # # plot_columns = \
    # #     min(5, int(math.ceil(math.sqrt(len(block_operator_list[0])))))
    # # plot_rows = math.ceil(len(block_operator_list[0])/plot_columns)
    # # for operator_index, (operator_name, operator) in \
    # #         enumerate(block_operator_list[0].items()):
    # #     plt.subplot(plot_rows, plot_columns, operator_index + 1)
    # #     plt.imshow(colour_complex_matrix(
    # #         operator/(2*np.max(np.abs(operator)))))
    # #     plt.title(operator_name)
    # #     plt.xticks([], [])
    # #     plt.yticks([], [])
    # # plt.draw()

    # plt.figure(
    #     figsize=(6.4, 4.8),
    #     label="tensors"
    # )
    # plot_columns = int(math.ceil(math.sqrt(len(tensor_dict))))
    # plot_rows = math.ceil(len(tensor_dict)/plot_columns)
    # for tensor_index, (tensor_name, tensor) in \
    #         enumerate(tensor_dict.items()):
    #     plt.subplot(plot_rows, plot_columns, tensor_index + 1)
    #     plt.imshow(colour_complex_matrix(
    #         tensor/(2*np.max(np.abs(tensor)))))
    #     plt.title(tensor_name)
    #     plt.xticks([], [])
    #     plt.yticks([], [])
    # plt.draw()

    copy_dict = copy.copy(operator_dicts)
    copy_dict.pop("coherent_atoms")

    return copy_dict


def _generate_superoperators(
        operator_dict: dict, valid_indices: np.ndarray) -> dict:
    superoperator_dict = {}
    for label, operator in operator_dict.items():
        if "|" not in label:
            if "L" in label:
                superoperator = _generate_dissipator(
                    operator, valid_indices)
            else:
                superoperator = _generate_von_neumann(
                    operator, valid_indices)
            superoperator_dict[label] = superoperator
    return superoperator_dict


def _combine_superoperators(superoperator_dict: dict):

    dissipator_dict = {}
    superoperator_combine_labels = \
        {"LS1", "LI1", "Lrc", "Llc", "Lrn", "Lln", "Lisc"}
    superoperator_dict_add = {}
    for label, superoperator in superoperator_dict.items():
        if "]" not in label:
            continue
        atom_label, operator_label = label.split("] ", 1)
        atom_label += "] "
        for combine_label in superoperator_combine_labels:
            if len(operator_label) > len(combine_label):
                if operator_label[:len(combine_label)] == combine_label:
                    atom_combine_label = atom_label + combine_label
                    if atom_combine_label in superoperator_dict_add.keys():
                        superoperator_dict_add[atom_combine_label] += \
                            superoperator
                    else:
                        superoperator_dict_add[atom_combine_label] = \
                            superoperator.copy()
    superoperator_dict.update(superoperator_dict_add)
    dissipator_dict.update(superoperator_dict_add)

    superoperator_combine_labels_dict = {
        "D": ["LS1", "LI1", "LS2", "LI2", "Llc", "Lln", "Lisc"],
        "Gr": ["Lrc", "Lrn"]
    }
    superoperator_dict_add = {}
    for combined_label, superoperator_combine_labels in \
            superoperator_combine_labels_dict.items():
        for label, superoperator in superoperator_dict.items():
            if "]" not in label:
                continue
            atom_label, operator_label = label.split("] ", 1)
            atom_label += "] "
            for combine_label in superoperator_combine_labels:
                if operator_label == combine_label:
                    atom_combine_label = atom_label + combined_label
                    if atom_combine_label in superoperator_dict_add.keys():
                        superoperator_dict_add[atom_combine_label] += \
                            superoperator
                    else:
                        superoperator_dict_add[atom_combine_label] = \
                            superoperator.copy()
    superoperator_dict.update(superoperator_dict_add)
    dissipator_dict.update(superoperator_dict_add)

    superoperator_dict_add = {}
    for label, superoperator in dissipator_dict.items():
        if "]" not in label:
            continue
        atom_label, operator_label = label.split("] ", 1)
        if operator_label in superoperator_combine_labels_dict.keys():
            if operator_label in superoperator_dict_add.keys():
                superoperator_dict_add[operator_label] += superoperator
            else:
                superoperator_dict_add[operator_label] = superoperator.copy()
    superoperator_dict.update(superoperator_dict_add)
    dissipator_dict.update(superoperator_dict_add)

    superoperator_combine_labels_dict = {"L0": ["H0", "D"]}
    superoperator_dict_add = {}
    for combined_label, superoperator_combine_labels in \
            superoperator_combine_labels_dict.items():
        for label, superoperator in superoperator_dict.items():
            if label in superoperator_combine_labels:
                if combined_label in superoperator_dict_add.keys():
                    superoperator_dict_add[combined_label] += superoperator
                else:
                    superoperator_dict_add[combined_label] = \
                        superoperator.copy()
    superoperator_dict.update(superoperator_dict_add)
    dissipator_dict.update(superoperator_dict_add)

    return dissipator_dict


def _generate_lindbladian(coefficient_functions: list[callable]):
    coefficient_x = nc.jit(device=True)(coefficient_functions[0])
    coefficient_y = nc.jit(device=True)(coefficient_functions[1])
    coefficient_z = nc.jit(device=True)(coefficient_functions[2])
    coefficient_r = nc.jit(device=True)(coefficient_functions[3])

    def lindbladian(time, coefficient):
        coefficient[0] = 1.0
        coefficient[1] = coefficient_x(time)
        coefficient[2] = coefficient_y(time)
        coefficient[3] = coefficient_z(time)
        coefficient[4] = coefficient_r(time)

    return lindbladian


def generate_7(
        coefficient_functions: list[callable],
        quiescent_magnetic_field: np.ndarray,
        use_rotating: bool = False):

    lindbladian = _generate_lindbladian(coefficient_functions)

    nv_ground = {
        "S": 1,
        "g": s3p.nv.room.ground.g_longitudinal,
        "g_perp": s3p.nv.room.ground.g_transverse,
        "D": math.tau*s3p.nv.room.ground.zfs_longitudinal,
        "TS1": s3p.nv.room.ground.thermalisation_time,
        "TS2": s3p.nv.room.ground.dephasing_time,

        "B0": quiescent_magnetic_field,
        "T": s3p.standards.lab.ntp.temperature
    }

    nv_excited = {
        "S": 1,
        "g": s3p.nv.room.excited.g_longitudinal,
        "D": math.tau*s3p.nv.room.excited.zfs_longitudinal,
        "TS1": s3p.nv.room.excited.thermalisation_time,
        "TS2": s3p.nv.room.excited.dephasing_time,

        "B0": quiescent_magnetic_field,
        "T": s3p.standards.lab.ntp.temperature
    }

    nv_singlet = {
        "S": 0,
    }

    nv_orbitals = {
        # Optical transitions
        ((0, 0), (1, 0)): {
            "rel": s3p.nv.room.optical.conserving,
            "rel_n": s3p.nv.room.optical.nonconserving
        },

        # ISC excited
        ((1, 0), (2, 0)): {
            "s_gets_0": s3p.nv.room.isc.s_gets_z,
            "s_gets_1": s3p.nv.room.isc.s_gets_pm
        },

        # ISC ground
        ((2, 0), (0, 0)): {
            "0_gets_s": s3p.nv.room.isc.z_gets_s,
            "1_gets_s": s3p.nv.room.isc.pm_gets_s
        }
    }

    generators, vectorisation_map = generate_atoms(
        [[nv_ground], [nv_excited], [nv_singlet]], [{}, {}, {}], nv_orbitals
    )

    generators_list = list(generators["generators"].values())

    if use_rotating:
        vectors_real, inv_vectors_real, doubles, singles = \
            real_eig(generators_list[0])

        generators_list_real = [
            inv_vectors_real@generator@vectors_real
            for generator in generators_list
        ]

        return (
            lindbladian, generators_list_real, vectorisation_map, vectors_real,
            inv_vectors_real, doubles, singles
        )

    return lindbladian, generators_list, vectorisation_map


def generate_21(
        coefficient_functions: list[callable],
        quiescent_magnetic_field: np.ndarray):

    lindbladian = _generate_lindbladian(coefficient_functions)

    nv_ground = {
        "S": 1,
        "g": s3p.nv.room.ground.g_longitudinal,
        "g_perp": s3p.nv.room.ground.g_transverse,
        "D": math.tau*s3p.nv.room.ground.zfs_longitudinal,
        "TS1": s3p.nv.room.ground.thermalisation_time,
        "TS2": s3p.nv.room.ground.dephasing_time,

        "I": 1,
        "gN": s3p.nv.room.ground.g_N_14N_longitudinal,
        "P": math.tau*s3p.nv.room.ground.nuclear_quadrupole_14N_longitudinal,
        "TI1": s3p.nv.room.ground.thermalisation_time_14N,
        "TI2": s3p.nv.room.ground.dephasing_time_14N,
        "A": math.tau*s3p.nv.room.ground.hyperfine_14N_longitudinal,
        "A_perp": math.tau*s3p.nv.room.ground.hyperfine_14N_transverse,

        "B0": quiescent_magnetic_field,
        "T": s3p.standards.lab.ntp.temperature
    }

    nv_excited = {
        "S": 1,
        "g": s3p.nv.room.excited.g_longitudinal,
        "D": math.tau*s3p.nv.room.excited.zfs_longitudinal,
        "TS1": s3p.nv.room.excited.thermalisation_time,
        "TS2": s3p.nv.room.excited.dephasing_time,

        "I": 1,
        "gN": s3p.nv.room.excited.g_N_14N_longitudinal,
        "P": math.tau*s3p.nv.room.excited.nuclear_quadrupole_14N_longitudinal,
        "TI1": s3p.nv.room.excited.thermalisation_time_14N,
        "TI2": s3p.nv.room.excited.dephasing_time_14N,
        "A": math.tau*s3p.nv.room.excited.hyperfine_14N_longitudinal,
        "A_perp": math.tau*s3p.nv.room.excited.hyperfine_14N_transverse,

        "B0": quiescent_magnetic_field,
        "T": s3p.standards.lab.ntp.temperature
    }

    nv_singlet = {
        "S": 0,

        "I": 1,
        "gN": s3p.nv.room.ground.g_N_14N_longitudinal,
        "P": math.tau*s3p.nv.room.ground.nuclear_quadrupole_14N_longitudinal,
        "TI1": s3p.nv.room.ground.thermalisation_time_14N,
        "TI2": s3p.nv.room.ground.dephasing_time_14N,

        "B0": quiescent_magnetic_field,
        "T": s3p.standards.lab.ntp.temperature
    }

    nv_orbitals = {
        # Optical transitions
        ((0, 0), (1, 0)): {
            "rel": s3p.nv.room.optical.conserving,
            "rel_n": s3p.nv.room.optical.nonconserving
        },

        # ISC excited
        ((1, 0), (2, 0)): {
            "s_gets_0": s3p.nv.room.isc.s_gets_z,
            "s_gets_1": s3p.nv.room.isc.s_gets_pm
        },

        # ISC ground
        ((2, 0), (0, 0)): {
            "0_gets_s": s3p.nv.room.isc.z_gets_s,
            "1_gets_s": s3p.nv.room.isc.pm_gets_s
        }
    }

    generators, vectorisation_map = generate_atoms(
        [[nv_ground], [nv_excited], [nv_singlet]], [{}, {}, {}], nv_orbitals
    )

    generators_list = list(generators["generators"].values())

    return lindbladian, generators_list, vectorisation_map


def real_eig(generator):
    values, vectors = np.linalg.eig(generator)

    vectors_complex = np.empty(
        (vectors.shape[0], vectors.shape[1], 2), dtype=np.float64)
    vectors_complex[:, :, 0] = np.real(vectors)
    vectors_complex[:, :, 1] = np.imag(vectors)

    vectors_list_real = []
    values_list_real = []
    vectors_list_real_single = []
    vectors_list_real_double = []
    values_list_real_single = []
    values_list_real_double = []
    rank_best = 0
    for index in range(vectors.shape[1]):
        vector_real = np.real(vectors[:, index])
        vectors_list = [
            *vectors_list_real, vector_real]
        vectors_array = np.array(vectors_list).T
        rank = np.linalg.matrix_rank(vectors_array)
        if rank > rank_best:
            rank_best = rank
            vector_real /= math.sqrt(np.sum(vector_real**2))
            vectors_list_real = vectors_list
            values_list_real.append(np.real(values[index]))
        else:
            continue
        vector_imag = np.imag(vectors[:, index])
        vectors_list = [
            *vectors_list_real, vector_imag]
        vectors_array = np.array(vectors_list).T
        rank = np.linalg.matrix_rank(vectors_array)
        if rank > rank_best:
            rank_best = rank
            vector_imag /= math.sqrt(np.sum(vector_imag**2))
            vectors_list_real = vectors_list
            values_list_real.append(np.imag(values[index]))
            vectors_list_real_double.append([vector_real, vector_imag])
            values_list_real_double.append(
                [np.real(values[index]), np.imag(values[index])])
        else:
            vectors_list_real_single.append(vector_real)
            values_list_real_single.append(np.real(values[index]))

    values_list_real_double, vectors_list_real_double = \
        zip(*sorted(
            zip(values_list_real_double, vectors_list_real_double),
            key=(lambda x: -(x[0][0]**2 + x[0][1]**2))
        ))
    values_real_double = np.array(values_list_real_double)
    vectors_list_real_double_new = []
    for vector_double in vectors_list_real_double:
        vectors_list_real_double_new += vector_double
    vectors_list_real_double = vectors_list_real_double_new

    values_list_real_single, vectors_list_real_single = \
        zip(*sorted(
            zip(values_list_real_single, vectors_list_real_single),
            key=(lambda x: -abs(x[0]))
        ))
    values_real_single = np.array(values_list_real_single)

    vectors_list_real = \
        [*vectors_list_real_double, *vectors_list_real_single]
    vectors_real = np.array(vectors_list_real).T
    inv_vectors_real = np.linalg.inv(vectors_real)

    return (
        vectors_real, inv_vectors_real, values_real_double,
        values_real_single
    )


# Legacy code start ===========================================================

def _generate_valid_indices(valid_mask: np.ndarray = None):
    if valid_mask is None:
        valid_mask = np.zeros((7, 7), dtype=meta_datatype)
        valid_mask[:3, :3] = 1
        valid_mask[3:6, 3:6] = 1
        valid_mask[6, 6] = 1

    valid_indices = []

    hilbert_size = valid_mask.shape[0]
    for y_index in range(hilbert_size):
        if valid_mask[y_index, y_index]:
            valid_indices.append([y_index, y_index, 0])

    for y_index in range(hilbert_size - 1):
        for x_index in range(y_index + 1, hilbert_size):
            if valid_mask[y_index, x_index]:
                valid_indices.append([y_index, x_index, 0])
                valid_indices.append([y_index, x_index, 1])

    valid_indices = np.array(valid_indices, dtype=np.int32)
    return valid_indices


def _generate_von_neumann(operator: np.ndarray, valid_indices: np.ndarray):
    operator_dimension = valid_indices.shape[0]
    hilbert_size = operator.shape[0]
    superoperator = np.empty(
        (operator_dimension, operator_dimension),
        dtype=meta_datatype
    )

    for in_index in range(operator_dimension):
        y_in_index = valid_indices[in_index, 0]
        x_in_index = valid_indices[in_index, 1]
        c_in_index = valid_indices[in_index, 2]

        density_matrix = np.zeros(
            (hilbert_size, hilbert_size, 2), dtype=meta_datatype)
        density_matrix[y_in_index, x_in_index, c_in_index] = 1
        if y_in_index != x_in_index:
            if c_in_index:
                density_matrix[x_in_index, y_in_index, c_in_index] = -1
            else:
                density_matrix[x_in_index, y_in_index, c_in_index] = 1

        scratch = _mult(operator, density_matrix) \
            - _mult(density_matrix, operator)
        operator_out = np.empty_like(scratch)
        operator_out[:, :, 0] = scratch[:, :, 1]
        operator_out[:, :, 1] = -scratch[:, :, 0]

        for out_index in range(operator_dimension):
            y_out_index = valid_indices[out_index, 0]
            x_out_index = valid_indices[out_index, 1]
            c_out_index = valid_indices[out_index, 2]
            superoperator[out_index, in_index] = \
                operator_out[y_out_index, x_out_index, c_out_index]
    return superoperator


def _generate_dissipator(operator: np.ndarray, valid_indices: np.ndarray):
    operator_dimension = valid_indices.shape[0]
    hilbert_size = operator.shape[0]
    superoperator = np.empty(
        (operator_dimension, operator_dimension),
        dtype=meta_datatype
    )

    for in_index in range(operator_dimension):
        y_in_index = valid_indices[in_index, 0]
        x_in_index = valid_indices[in_index, 1]
        c_in_index = valid_indices[in_index, 2]

        density_matrix = np.zeros(
            (hilbert_size, hilbert_size, 2), dtype=meta_datatype)
        density_matrix[y_in_index, x_in_index, c_in_index] = 1
        if y_in_index != x_in_index:
            if c_in_index:
                density_matrix[x_in_index, y_in_index, c_in_index] = -1
            else:
                density_matrix[x_in_index, y_in_index, c_in_index] = 1

        operator_transpose = np.transpose(operator.copy(), axes=(1, 0, 2))
        operator_transpose[:, :, 1] = -operator_transpose[:, :, 1]
        operator_out = _mult(_mult(operator, density_matrix),
                             operator_transpose)
        proj = _mult(operator_transpose, operator)
        operator_out -= 0.5*_mult(proj, density_matrix)
        operator_out -= 0.5*_mult(density_matrix, proj)

        for out_index in range(operator_dimension):
            y_out_index = valid_indices[out_index, 0]
            x_out_index = valid_indices[out_index, 1]
            c_out_index = valid_indices[out_index, 2]
            superoperator[out_index, in_index] = \
                operator_out[y_out_index, x_out_index, c_out_index]
    return superoperator


# Main ========================================================================


def main():
    import matplotlib.pyplot as plt
    from cmcrameri import cm

    from pogger import Pogger as Logger

    with Logger("superspinsim-generate") as logger:

        quiescent_magnetic_field = np.array([0, 0, 0])*1e-12

        nv_ground = {
            "S": 1,
            "g": s3p.nv.room.ground.g_longitudinal,
            "g_perp": s3p.nv.room.ground.g_transverse,
            "D": math.tau*s3p.nv.room.ground.zfs_longitudinal,
            "TS1": s3p.nv.room.ground.thermalisation_time,
            "TS2": s3p.nv.room.ground.dephasing_time,

            # "I": 1,
            # "gN": s3p.nv.room.ground.g_N_14N_longitudinal,
            # "P": math.tau*s3p.nv.room.ground.nuclear_quadrupole_14N_longitudinal,
            # "TI1": s3p.nv.room.ground.thermalisation_time_14N,
            # "TI2": s3p.nv.room.ground.dephasing_time_14N,
            # "A": math.tau*s3p.nv.room.ground.hyperfine_14N_longitudinal,
            # "A_perp": math.tau*s3p.nv.room.ground.hyperfine_14N_transverse,

            "B0": quiescent_magnetic_field,
            "T": 0 #s3p.standards.lab.ntp.temperature
        }

        nv_excited = {
            "S": 1,
            "g": s3p.nv.room.excited.g_longitudinal,
            "D": math.tau*s3p.nv.room.excited.zfs_longitudinal,
            "TS1": s3p.nv.room.excited.thermalisation_time,
            "TS2": s3p.nv.room.excited.dephasing_time,

            # "I": 1,
            # "gN": s3p.nv.room.excited.g_N_14N_longitudinal,
            # "P": math.tau*s3p.nv.room.excited.nuclear_quadrupole_14N_longitudinal,
            # "TI1": s3p.nv.room.excited.thermalisation_time_14N,
            # "TI2": s3p.nv.room.excited.dephasing_time_14N,
            # "A": math.tau*s3p.nv.room.excited.hyperfine_14N_longitudinal,
            # "A_perp": math.tau*s3p.nv.room.excited.hyperfine_14N_transverse,

            "B0": quiescent_magnetic_field,
            "T": 0 #s3p.standards.lab.ntp.temperature
        }

        nv_singlet = {
            "S": 0,

            # "I": 1,
            # "gN": s3p.nv.room.ground.g_N_14N_longitudinal,
            # "P": math.tau*s3p.nv.room.ground.nuclear_quadrupole_14N_longitudinal,
            # "TI1": s3p.nv.room.ground.thermalisation_time_14N,
            # "TI2": s3p.nv.room.ground.dephasing_time_14N,

            # "B0": quiescent_magnetic_field,
            # "T": 0 #s3p.standards.lab.ntp.temperature
        }

        nv_orbitals = {
            # Optical transitions
            ((0, 0), (1, 0)): {
                "rel": s3p.nv.room.optical.conserving,
                "rel_n": s3p.nv.room.optical.nonconserving
            },

            # ISC excited
            ((1, 0), (2, 0)): {
                "s_gets_0": s3p.nv.room.isc.s_gets_z,
                "s_gets_1": s3p.nv.room.isc.s_gets_pm
            },

            # ISC ground
            ((2, 0), (0, 0)): {
                "0_gets_s": s3p.nv.room.isc.z_gets_s,
                "1_gets_s": s3p.nv.room.isc.pm_gets_s
            }
        }

        generators, vectorisation_map = generate_atoms(
            [[nv_ground], [nv_excited], [nv_singlet]], [{}, {}, {}],
            nv_orbitals
        )

        generators_list = list(generators["generators"].values())

        eig = logger.record(
            ("vectors", "inv_vectors", "values_double", "values_single"))(
            real_eig
        )

        @logger.record()
        def schur():
            quiescent = generators_list[0]
            quiescent /= np.linalg.matrix_norm(quiescent)

            operator = quiescent@quiescent.T - quiescent.T@quiescent

            print(np.linalg.matrix_norm(quiescent))
            print(np.linalg.matrix_norm(operator))

            # operator = generators["super_all"]["[0, 0] LS1"]

            # schur = spl.schur(quiescent, "real")[0]
            # schur_max = np.max(np.abs(schur))
            # schur = schur*(np.abs(schur) > schur_max*1e-3)

            # operator = schur

            plt.figure(label="schur")
            operator = np.sign(operator)*(
                np.log10(np.abs(operator) + 1e-20) + 20)
            plt.imshow(colour_complex_matrix(
                operator/(np.max(np.abs(operator)))),
                       interpolation="nearest")
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            plt.draw()

        # schur()

        # plt.show()

        @logger.record()
        def plot(
                vectors_real: np.ndarray, inv_vectors_real: np.ndarray,
                generators: list):
            plt.figure(label="real_eigenvectors")
            plt.imshow(
                colour_complex_matrix(
                    vectors_real/np.max(np.abs(vectors_real))
                )
            )
            plt.draw()

            plt.figure(label="inv_real_eigenvectors")
            plt.imshow(
                colour_complex_matrix(
                    inv_vectors_real
                    / np.max(np.abs(inv_vectors_real))
                )
            )
            plt.draw()

            label_list = ["Lq", "Gx", "Gy", "Gz", "Go"]
            for generator, label in zip(generators, label_list):
                generator_transform = inv_vectors_real@generator@vectors_real
                plt.figure(label=label, figsize=(6.4, 4.8*2))
                plt.suptitle(label)
                plt.subplot(2, 1, 1)
                plt.title("Real basis")
                plt.imshow(
                    colour_complex_matrix(
                        generator
                        / np.max(np.abs(generator))
                    )
                )

                plt.subplot(2, 1, 2)
                plt.title("Real eigenbasis")
                plt.imshow(
                    colour_complex_matrix(
                        generator_transform
                        / np.max(np.abs(generator_transform))
                    )
                )
                plt.draw()

        logger.set_context("real_diagonalisation")
        vectors_real, inv_vectors_real, _, _ = eig(generators_list[0])
        plot(vectors_real, inv_vectors_real, generators_list)

        plt.show()

    # logger.set_context("operator_generation")
    # plot_operators = logger.record(("operators"))(_plot_operators)
    # plot_operators(generators)

    # plt.draw()


if __name__ == "__main__":
    main()
