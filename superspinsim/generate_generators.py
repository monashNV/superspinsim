import numpy as np
import math

import matplotlib.pyplot as plt
from util import colour_complex_matrix
from pogger import Pogger as Logger

meta_datatype = np.float64

with Logger("superspinsim-generate") as logger:
    @logger.record()
    def generate_atoms(description: list, verbose=False):
        hilbert_space_shape = []
        operator_labels = set()
        previous_identity = None
        for block in description:
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
                if spin > 1/2:
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

                    g_electron = np.zeros((3, 3), dtype=meta_datatype)
                    g_electron[0, 0] = g_iso - g_dipole
                    g_electron[1, 1] = g_iso - g_dipole
                    g_electron[2, 2] = g_iso + 2*g_dipole

                # Nuclear spin
                if "I" in atom.keys():
                    electron_spin = spin

                    spin = atom["I"]

                    spin_x, spin_y, spin_z, previous_identity = add_spin(
                        spin, hilbert_space_shape, previous_identity,
                        description, operator_labels
                    )

                    atom["Ix"] = spin_x
                    atom["Iy"] = spin_y
                    atom["Iz"] = spin_z
                    operator_labels |= {"Ix", "Iy", "Iz"}

                    # Hyperfine
                    if electron_spin > 0:
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
                        atom["Sx Ix"] = hyperfine_xx
                        atom["Sx Iy"] = hyperfine_xy
                        atom["Sx Iz"] = hyperfine_xz
                        atom["Sy Ix"] = hyperfine_yx
                        atom["Sy Iy"] = hyperfine_yy
                        atom["Sy Iz"] = hyperfine_yz
                        atom["Sz Ix"] = hyperfine_zx
                        atom["Sz Iy"] = hyperfine_zy
                        atom["Sz Iz"] = hyperfine_zz
                        operator_labels |= {"Fx", "Fy", "Fz"}
                        operator_labels |= {"Sx Ix", "Sx Iy", "Sx Iz"}
                        operator_labels |= {"Sy Ix", "Sy Iy", "Sy Iz"}
                        operator_labels |= {"Sz Ix", "Sz Iy", "Sz Iz"}

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
        # Plot
        if verbose:
            operator_list = []
            operator_names = []
            for block_index, block in enumerate(description):
                for atom_index, atom in enumerate(block):
                    for operator_label in operator_labels:
                        if operator_label in atom.keys():
                            operator_list.append(atom[operator_label])
                            operator_names.append(
                                f"[{block_index}, {atom_index}]"
                                f" {operator_label}"
                            )

            plt.figure(
                figsize=(6.4, 2*4.8),
                label="spins"            )
            plot_columns = 5  # int(math.ceil(math.sqrt(len(operator_list))))
            plot_rows = math.ceil(len(operator_list)/plot_columns)
            for operator_index, (operator, operator_name) in \
                    enumerate(zip(operator_list, operator_names)):
                plt.subplot(plot_rows, plot_columns, operator_index + 1)
                plt.imshow(colour_complex_matrix(
                    operator/np.max(np.abs(operator))))
                plt.title(operator_name)
                plt.gca().set_axis_off()
            plt.draw()

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
        # {"S": 1, "g": 2, "g_perp": 2.1, "I": 1}, {"I": 1/2}
        {"S": 1/2, "g": 2, "g_perp": 2.1, "I": 3/2}
    ]], verbose=True)

    plt.draw()
