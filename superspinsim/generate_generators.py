import numpy as np
import math

import matplotlib.pyplot as plt
from util import colour_complex_matrix
from pogger import Pogger as Logger

meta_datatype = np.float64

constants = {
    "bohr_gyro": {
        "value": math.tau*13.9962449171e9,
        "units": "rad/s/T",
        "citation": "CODATA Recommended Values of the " \
                    + "Fundamental Physical Constants: 2022"
    },

    "nuclear_gyro": {
        "value": math.tau*7.6225932188e6,
        "units": "rad/s/T",
        "citation": "CODATA Recommended Values of the " \
                    + "Fundamental Physical Constants: 2022"
    }
}

with Logger("superspinsim-generate") as logger:
    @logger.record(("operators", "tensors"))
    def generate_atoms(description: list, atom_interactions, verbose=False):
        hilbert_space_shape = []
        operator_labels = set()
        tensor_labels = set()
        zero_field_labels = set()
        field_labels = set()
        previous_identity = None

        for block, atom_interaction in \
                zip(description, atom_interactions):
            for atom_index, atom in enumerate(block):
                # Electron spin
                if "S" in atom.keys():
                    spin = atom["S"]
                else:
                    spin = 0

                if spin > 0:
                    spin_x, spin_y, spin_z, previous_identity = add_spin(
                        spin, hilbert_space_shape, previous_identity,
                        description, operator_labels
                    )

                    atom["Sx"] = spin_x
                    atom["Sy"] = spin_y
                    atom["Sz"] = spin_z
                    operator_labels |= {"Sx", "Sy", "Sz"}

                # Electron ZFS
                zfs = np.zeros((3, 3), dtype=meta_datatype)
                zfs_longitudinal = 0
                zfs_transverse = 0
                if spin > 1/2:
                    if "D" in atom.keys():
                        zfs_longitudinal = atom["D"]
                    if "E" in atom.keys():
                        zfs_transverse = atom["E"]
                    zfs[0, 0] = zfs_transverse - zfs_longitudinal/3
                    zfs[1, 1] = -zfs_transverse - zfs_longitudinal/3
                    zfs[2, 2] = zfs_longitudinal*2/3

                    spin_xx = _mult(spin_x, spin_x)
                    spin_xy = _mult(spin_x, spin_y)
                    spin_xz = _mult(spin_x, spin_z)
                    spin_yx = _mult(spin_y, spin_x)
                    spin_yy = _mult(spin_y, spin_y)
                    spin_yz = _mult(spin_y, spin_z)
                    spin_zx = _mult(spin_z, spin_x)
                    spin_zy = _mult(spin_z, spin_y)
                    spin_zz = _mult(spin_z, spin_z)

                    atom["Sx Sx"] = spin_xx
                    atom["Sx Sy"] = spin_xy
                    atom["Sx Sz"] = spin_xz
                    atom["Sy Sx"] = spin_yx
                    atom["Sy Sy"] = spin_yy
                    atom["Sy Sz"] = spin_yz
                    atom["Sz Sx"] = spin_zx
                    atom["Sz Sy"] = spin_zy
                    atom["Sz Sz"] = spin_zz
                    operator_labels |= {"Sx Sx", "Sx Sy", "Sx Sz"}
                    operator_labels |= {"Sy Sx", "Sy Sy", "Sy Sz"}
                    operator_labels |= {"Sz Sx", "Sz Sy", "Sz Sz"}

                    zfs_generator = zfs[0, 0]*spin_xx + zfs[0, 1]*spin_xy \
                        + zfs[0, 2]*spin_xz + zfs[1, 0]*spin_yx \
                        + zfs[1, 1]*spin_yy + zfs[1, 2]*spin_yz \
                        + zfs[2, 0]*spin_zx + zfs[2, 1]*spin_zy \
                        + zfs[2, 2]*spin_zz
                    atom["Dten"] = zfs
                    atom["Dgen"] = zfs_generator
                    tensor_labels.add("Dten")
                    operator_labels.add("Dgen")
                    zero_field_labels.add("Dgen")

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

                    atom["gyroS"] = g_spin*constants["bohr_gyro"]["value"]
                    tensor_labels |= {"gyroS"}

                # Nuclear spin
                if "I" in atom.keys():
                    electron_spin = spin

                    spin = atom["I"]
                    nuclear_spin = spin

                    spin_x, spin_y, spin_z, previous_identity = add_spin(
                        spin, hilbert_space_shape, previous_identity,
                        description, operator_labels
                    )

                    atom["Ix"] = spin_x
                    atom["Iy"] = spin_y
                    atom["Iz"] = spin_z
                    operator_labels |= {"Ix", "Iy", "Iz"}

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
                        spin_xx = _mult(spin_x, spin_x)
                        spin_xy = _mult(spin_x, spin_y)
                        spin_xz = _mult(spin_x, spin_z)
                        spin_yx = _mult(spin_y, spin_x)
                        spin_yy = _mult(spin_y, spin_y)
                        spin_yz = _mult(spin_y, spin_z)
                        spin_zx = _mult(spin_z, spin_x)
                        spin_zy = _mult(spin_z, spin_y)
                        spin_zz = _mult(spin_z, spin_z)

                        atom["Ix Ix"] = spin_xx
                        atom["Ix Iy"] = spin_xy
                        atom["Ix Iz"] = spin_xz
                        atom["Iy Ix"] = spin_yx
                        atom["Iy Iy"] = spin_yy
                        atom["Iy Iz"] = spin_yz
                        atom["Iz Ix"] = spin_zx
                        atom["Iz Iy"] = spin_zy
                        atom["Iz Iz"] = spin_zz
                        operator_labels |= {"Ix Ix", "Ix Iy", "Ix Iz"}
                        operator_labels |= {"Iy Ix", "Iy Iy", "Iy Iz"}
                        operator_labels |= {"Iz Ix", "Iz Iy", "Iz Iz"}

                        zfs_generator = zfs[0, 0]*spin_xx + zfs[0, 1]*spin_xy \
                            + zfs[0, 2]*spin_xz + zfs[1, 0]*spin_yx \
                            + zfs[1, 1]*spin_yy + zfs[1, 2]*spin_yz \
                            + zfs[2, 0]*spin_zx + zfs[2, 1]*spin_zy \
                            + zfs[2, 2]*spin_zz
                        atom["Pten"] = zfs
                        atom["Pgen"] = zfs_generator
                        tensor_labels.add("Pten")
                        operator_labels.add("Pgen")
                        zero_field_labels.add("Pgen")

                    # Nuclear Zeeman
                    if spin > 0:
                        if "gN" in atom.keys():
                            g_iso = atom["g"]
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

                        atom["gyroI"] = -g_spin \
                            * constants["nuclear_gyro"]["value"]
                        tensor_labels |= {"gyroI"}

                    # Hyperfine
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

                        electron_spin_x = atom["Sx"]
                        electron_spin_y = atom["Sy"]
                        electron_spin_z = atom["Sz"]

                        hyperfine_x = spin_x + electron_spin_x
                        hyperfine_y = spin_y + electron_spin_y
                        hyperfine_z = spin_z + electron_spin_z
                        hyperfine_xx = _mult(electron_spin_x, spin_x)
                        hyperfine_xy = _mult(electron_spin_x, spin_y)
                        hyperfine_xz = _mult(electron_spin_x, spin_z)
                        hyperfine_yx = _mult(electron_spin_y, spin_x)
                        hyperfine_yy = _mult(electron_spin_y, spin_y)
                        hyperfine_yz = _mult(electron_spin_y, spin_z)
                        hyperfine_zx = _mult(electron_spin_z, spin_x)
                        hyperfine_zy = _mult(electron_spin_z, spin_y)
                        hyperfine_zz = _mult(electron_spin_z, spin_z)

                        atom["Fx"] = hyperfine_x
                        atom["Fy"] = hyperfine_y
                        atom["Fz"] = hyperfine_z
                        operator_labels |= {"Fx", "Fy", "Fz"}

                        atom["Sx Ix"] = hyperfine_xx
                        atom["Sx Iy"] = hyperfine_xy
                        atom["Sx Iz"] = hyperfine_xz
                        atom["Sy Ix"] = hyperfine_yx
                        atom["Sy Iy"] = hyperfine_yy
                        atom["Sy Iz"] = hyperfine_yz
                        atom["Sz Ix"] = hyperfine_zx
                        atom["Sz Iy"] = hyperfine_zy
                        atom["Sz Iz"] = hyperfine_zz
                        operator_labels |= {"Sx Ix", "Sx Iy", "Sx Iz"}
                        operator_labels |= {"Sy Ix", "Sy Iy", "Sy Iz"}
                        operator_labels |= {"Sz Ix", "Sz Iy", "Sz Iz"}

                        hyperfine_generator = a_hyperfine[0, 0]*hyperfine_xx \
                            + a_hyperfine[0, 1]*hyperfine_xy \
                            + a_hyperfine[0, 2]*hyperfine_xz \
                            + a_hyperfine[1, 0]*hyperfine_yx \
                            + a_hyperfine[1, 1]*hyperfine_yy \
                            + a_hyperfine[1, 2]*hyperfine_yz \
                            + a_hyperfine[2, 0]*hyperfine_zx \
                            + a_hyperfine[2, 1]*hyperfine_zy \
                            + a_hyperfine[2, 2]*hyperfine_zz
                        atom["Agen"] = hyperfine_generator
                        operator_labels.add("Agen")
                        zero_field_labels.add("Agen")

                # Magnetic generators
                generator_x = np.empty_like(previous_identity)
                generator_y = np.empty_like(previous_identity)
                generator_z = np.empty_like(previous_identity)

                if electron_spin > 0:
                    spin_x = atom["Sx"]
                    spin_y = atom["Sy"]
                    spin_z = atom["Sz"]
                    spin_gyro = atom["gyroS"]

                    generator_x += spin_gyro[0, 0]*spin_x \
                        + spin_gyro[0, 1]*spin_y \
                        + spin_gyro[0, 2]*spin_z
                    generator_y += spin_gyro[1, 0]*spin_x \
                        + spin_gyro[1, 1]*spin_y \
                        + spin_gyro[1, 2]*spin_z
                    generator_z += spin_gyro[2, 0]*spin_x \
                        + spin_gyro[2, 1]*spin_y \
                        + spin_gyro[2, 2]*spin_z

                if nuclear_spin > 0:
                    spin_x = atom["Ix"]
                    spin_y = atom["Iy"]
                    spin_z = atom["Iz"]
                    spin_gyro = atom["gyroI"]

                    generator_x += spin_gyro[0, 0]*spin_x \
                        + spin_gyro[0, 1]*spin_y \
                        + spin_gyro[0, 2]*spin_z
                    generator_y += spin_gyro[1, 0]*spin_x \
                        + spin_gyro[1, 1]*spin_y \
                        + spin_gyro[1, 2]*spin_z
                    generator_z += spin_gyro[2, 0]*spin_x \
                        + spin_gyro[2, 1]*spin_y \
                        + spin_gyro[2, 2]*spin_z

                atom["Gx"] = generator_x
                atom["Gy"] = generator_y
                atom["Gz"] = generator_z
                operator_labels |= {"Gx", "Gy", "Gz"}
                field_labels |= {"Gx", "Gy", "Gz"}

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

            # Spin-spin interactions and molecular hyperfine
            for (index_a, index_b), j_description in atom_interaction.items():
                # Spin-spin tensor
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

                if "S" in atom_a:
                    spin_xa = atom_a["Sx"]
                    spin_ya = atom_a["Sy"]
                    spin_za = atom_a["Sz"]
                    label_a = "S"
                elif "I" in atom_a:
                    spin_xa = atom_a["Ix"]
                    spin_ya = atom_a["Iy"]
                    spin_za = atom_a["Iz"]
                    label_a = "I"

                if "S" in atom_b:
                    spin_xb = atom_b["Sx"]
                    spin_yb = atom_b["Sy"]
                    spin_zb = atom_b["Sz"]
                    label_b = "S"
                elif "I" in atom_b:
                    spin_xb = atom_b["Ix"]
                    spin_yb = atom_b["Iy"]
                    spin_zb = atom_b["Iz"]
                    label_b = "I"

                spin_xx = _mult(spin_xa, spin_xb)
                spin_xy = _mult(spin_xa, spin_yb)
                spin_xz = _mult(spin_xa, spin_zb)
                spin_yx = _mult(spin_ya, spin_xb)
                spin_yy = _mult(spin_ya, spin_yb)
                spin_yz = _mult(spin_ya, spin_zb)
                spin_zx = _mult(spin_za, spin_xb)
                spin_zy = _mult(spin_za, spin_yb)
                spin_zz = _mult(spin_za, spin_zb)

                j_description[f"{label_a}x {label_b}x"] = spin_xx
                j_description[f"{label_a}x {label_b}y"] = spin_xy
                j_description[f"{label_a}x {label_b}z"] = spin_xz
                j_description[f"{label_a}y {label_b}x"] = spin_yx
                j_description[f"{label_a}y {label_b}y"] = spin_yy
                j_description[f"{label_a}y {label_b}z"] = spin_yz
                j_description[f"{label_a}z {label_b}x"] = spin_zx
                j_description[f"{label_a}z {label_b}y"] = spin_zy
                j_description[f"{label_a}z {label_b}z"] = spin_zz

                operator_labels.add(f"{label_a}x {label_b}x")
                operator_labels.add(f"{label_a}x {label_b}y")
                operator_labels.add(f"{label_a}x {label_b}z")
                operator_labels.add(f"{label_a}y {label_b}x")
                operator_labels.add(f"{label_a}y {label_b}y")
                operator_labels.add(f"{label_a}y {label_b}z")
                operator_labels.add(f"{label_a}z {label_b}x")
                operator_labels.add(f"{label_a}z {label_b}y")
                operator_labels.add(f"{label_a}z {label_b}z")

                spin_generator = j_tensor[0, 0]*spin_xx \
                    + j_tensor[0, 1]*spin_xy + j_tensor[0, 2]*spin_xz \
                    + j_tensor[1, 0]*spin_yx + j_tensor[1, 1]*spin_yy \
                    + j_tensor[1, 2]*spin_yz + j_tensor[2, 0]*spin_zx \
                    + j_tensor[2, 1]*spin_zy + j_tensor[2, 2]*spin_zz
                j_description["Jgen"] = spin_generator
                operator_labels.add("Jgen")

        # Make traceless
        for block in description:
            for atom in block:
                for operator_label in operator_labels:
                    if operator_label in atom.keys():
                        operator = atom[operator_label]
                        trace = np.trace(operator[:, :, 0])
                        operator_new = operator.copy()
                        operator_new[:, :, 0] -= \
                            trace*np.eye(operator.shape[0])/operator.shape[0]
                        atom[operator_label] = operator_new

        # Combine all atoms in blocks
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
                operator_label = "Z"
                if operator_label in atom.keys():
                    if operator_label not in block_operator_dict.keys():
                        block_operator_dict[operator_label] = \
                            atom[operator_label].copy()
                    else:
                        block_operator_dict[operator_label] += \
                            atom[operator_label]

            for (index_a, index_b), j_description in atom_interaction.items():
                if "Jgen" in j_description.keys():
                    if "Z" not in block_operator_dict.keys():
                        block_operator_dict["Z"] = \
                            j_description["Jgen"].copy()
                    else:
                        block_operator_dict["Z"] += \
                            j_description["Jgen"]
            block_operator_list.append(block_operator_dict)

        # Create lists of operators
        operator_dict = {}
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
                operator_label = "Z"
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
                        operator_name = \
                            f"[{block_index}, {index_a}, {index_b}]" \
                            + f" {operator_label}"
                        operator_dict[operator_name] = \
                            j_description[operator_label]

                # Tensor
                for tensor_label in tensor_labels:
                    if tensor_label in j_description.keys():
                        tensor_name = \
                            f"[{block_index}, {index_a}, {index_b}]" \
                            + f" {tensor_label}"
                        tensor_dict[tensor_name] = \
                            j_description[tensor_label]

        # Plot
        if verbose:
            plt.figure(
                figsize=(6.4, 6*4.8),
                label="operators"
            )
            plot_columns = \
                min(4, int(math.ceil(math.sqrt(len(operator_dict)))))
            plot_rows = math.ceil(len(operator_dict)/plot_columns)
            for operator_index, (operator_name, operator) in \
                    enumerate(operator_dict.items()):
                plt.subplot(plot_rows, plot_columns, operator_index + 1)
                plt.imshow(colour_complex_matrix(
                    operator/(2*np.max(np.abs(operator)))))
                plt.title(operator_name)
                plt.xticks([], [])
                plt.yticks([], [])
                plt.tight_layout()
            plt.draw()

            plt.figure(
                figsize=(6.4, 6.4),
                label="composite_operators"
            )
            plot_columns = \
                min(4, int(math.ceil(math.sqrt(len(composite_operator_dict)))))
            plot_rows = math.ceil(len(composite_operator_dict)/plot_columns)
            for operator_index, (operator_name, operator) in \
                    enumerate(composite_operator_dict.items()):
                plt.subplot(plot_rows, plot_columns, operator_index + 1)
                plt.imshow(colour_complex_matrix(
                    operator/(2*np.max(np.abs(operator)))))
                plt.title(operator_name)
                plt.xticks([], [])
                plt.yticks([], [])
            plt.draw()

            plt.figure(
                figsize=(6.4, 6.4),
                label="blocks"
            )
            plot_columns = \
                min(5, int(math.ceil(math.sqrt(len(block_operator_list[0])))))
            plot_rows = math.ceil(len(block_operator_list[0])/plot_columns)
            for operator_index, (operator_name, operator) in \
                    enumerate(block_operator_list[0].items()):
                plt.subplot(plot_rows, plot_columns, operator_index + 1)
                plt.imshow(colour_complex_matrix(
                    operator/(2*np.max(np.abs(operator)))))
                plt.title(operator_name)
                plt.xticks([], [])
                plt.yticks([], [])
            plt.draw()

            plt.figure(
                figsize=(6.4, 4.8),
                label="tensors"
            )
            plot_columns = int(math.ceil(math.sqrt(len(tensor_dict))))
            plot_rows = math.ceil(len(tensor_dict)/plot_columns)
            for tensor_index, (tensor_name, tensor) in \
                    enumerate(tensor_dict.items()):
                plt.subplot(plot_rows, plot_columns, tensor_index + 1)
                plt.imshow(colour_complex_matrix(
                    tensor/(2*np.max(np.abs(tensor)))))
                plt.title(tensor_name)
                plt.xticks([], [])
                plt.yticks([], [])
            plt.draw()

        return operator_dict, tensor_dict

    def add_spin(
            spin, hilbert_space_shape, previous_identity,
            description, operator_labels):
        spin_dimension = int(2*spin + 1)
        hilbert_space_shape.append(spin_dimension)

        # Spin matrix elements (Sakurai 3ed Section 3.5.3)
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
        spin_identity[:, :, 0] = \
            np.eye(spin_x.shape[0])

        if previous_identity is not None:
            for block in description:
                for atom in block:
                    for operator_label in operator_labels:
                        if operator_label in atom.keys():
                            operator = atom[operator_label]
                            operator_new = kroneker_product(
                                spin_identity, operator)
                            atom[operator_label] = operator_new

            # operator_list_new = []
            # for operator in operator_list:
            #     operator_new = kroneker_product(
            #         operator, spin_identity)
            #     operator_list_new.append(operator_new)
            # operator_list = operator_list_new

            spin_x = kroneker_product(
                spin_x, previous_identity)
            spin_y = kroneker_product(
                spin_y, previous_identity)
            spin_z = kroneker_product(
                spin_z, previous_identity)
            spin_identity = kroneker_product(
                previous_identity, spin_identity)

        previous_identity = spin_identity

        return spin_x, spin_y, spin_z, previous_identity


    def kroneker_product(inner: np.ndarray, outer: np.ndarray):
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

    def _generate_valid_indices():
        valid_mask = np.zeros((7, 7), dtype=meta_datatype)
        valid_mask[:3, :3] = 1
        valid_mask[3:6, 3:6] = 1
        valid_mask[6, 6] = 1

        valid_indices = []

        for y_index in range(7):
            if valid_mask[y_index, y_index]:
                valid_indices.append([y_index, y_index, 0])

        for y_index in range(6):
            for x_index in range(y_index + 1, 7):
                if valid_mask[y_index, x_index]:
                    valid_indices.append([y_index, x_index, 0])
                    valid_indices.append([y_index, x_index, 1])

        valid_indices = np.array(valid_indices, dtype=np.int32)
        return valid_indices


    def _mult(operator_a, operator_b):
        operator_out = np.empty_like(operator_a)
        operator_out[:, :, 0] = operator_a[:, :, 0]@operator_b[:, :, 0] \
            - operator_a[:, :, 1]@operator_b[:, :, 1]
        operator_out[:, :, 1] = operator_a[:, :, 0]@operator_b[:, :, 1] \
            + operator_a[:, :, 1]@operator_b[:, :, 0]
        return operator_out


    def _generate_von_neumann(operator: np.ndarray, valid_indices: np.ndarray):
        operator_dimension = valid_indices.shape[0]
        superoperator = np.empty(
            (operator_dimension, operator_dimension),
            dtype=meta_datatype
        )

        for in_index in range(operator_dimension):
            y_in_index = valid_indices[in_index, 0]
            x_in_index = valid_indices[in_index, 1]
            c_in_index = valid_indices[in_index, 2]

            density_matrix = np.zeros((7, 7, 2), dtype=meta_datatype)
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
        superoperator = np.empty(
            (operator_dimension, operator_dimension),
            dtype=meta_datatype
        )

        for in_index in range(operator_dimension):
            y_in_index = valid_indices[in_index, 0]
            x_in_index = valid_indices[in_index, 1]
            c_in_index = valid_indices[in_index, 2]

            density_matrix = np.zeros((7, 7, 2), dtype=meta_datatype)
            density_matrix[y_in_index, x_in_index, c_in_index] = 1

            operator_transpose = np.transpose(operator, axes=(1, 0, 2))
            operator_out = _mult(_mult(operator, density_matrix),
                                 operator_transpose)
            operator_out -= 0.5*_mult(_mult(operator_transpose, operator),
                                      density_matrix)
            operator_out -= 0.5*_mult(density_matrix,
                                      _mult(operator_transpose, operator))

            for out_index in range(operator_dimension):
                y_out_index = valid_indices[out_index, 0]
                x_out_index = valid_indices[out_index, 1]
                c_out_index = valid_indices[out_index, 2]
                superoperator[out_index, in_index] = \
                    operator_out[y_out_index, x_out_index, c_out_index]
        return superoperator

    logger.set_context("spins")
    generate_atoms([[
        {"S": 1, "g": 2, "g_perp": 2.1, "D": 50, "I": 1, "P": 10, "A": 5},
        {"I": 1/2}
        # {"S": 1/2, "g": 2, "g_perp": 2.1, "I": 3/2}
    ]], [{(0, 1): {"J": 4}}], verbose=True)

    plt.draw()
