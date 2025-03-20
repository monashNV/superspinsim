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
        operator_list = []
        operator_names = []
        previous_spin_identity = None
        for block in description:
            for atom_index, atom in enumerate(block):
                # Electron spin
                if "S" in atom.keys():
                    electron_spin = atom["S"]
                else:
                    electron_spin = 1/2

                electron_spin_dimension = int(2*electron_spin + 1)
                hilbert_space_shape.append(electron_spin_dimension)

                # Spin matrix elements (Sakurai 3ed Section 3.5.3)
                magnetic = np.linspace(
                    electron_spin, -electron_spin, electron_spin_dimension,
                    dtype=meta_datatype
                )
                diag_p = np.sqrt(
                    (electron_spin - magnetic)*(1 + electron_spin + magnetic))[1:]
                diag_m = np.sqrt(
                    (electron_spin + magnetic)*(1 + electron_spin - magnetic))[:-1]
                electron_spin_p = np.diag(diag_p, 1)
                electron_spin_m = np.diag(diag_m, -1)

                electron_spin_x = np.zeros(
                    (electron_spin_p.shape[0], electron_spin_p.shape[1], 2),
                    dtype=meta_datatype
                )
                electron_spin_x[:, :, 0] = (electron_spin_p + electron_spin_m)/2

                electron_spin_y = np.zeros_like(electron_spin_x)
                electron_spin_y[:, :, 1] = (electron_spin_m - electron_spin_p)/2

                electron_spin_z = np.zeros_like(electron_spin_x)
                electron_spin_z[:, :, 0] = np.diag(magnetic)

                electron_spin_identity = np.zeros_like(electron_spin_x)
                electron_spin_identity[:, :, 0] = \
                    np.eye(electron_spin_x.shape[0])

                if previous_spin_identity is not None:
                    operator_list_new = []
                    for operator in operator_list:
                        operator_new = kroneker_product(
                            operator, electron_spin_identity)
                        operator_list_new.append(operator_new)
                    operator_list = operator_list_new

                    electron_spin_x = kroneker_product(
                        previous_spin_identity, electron_spin_x)
                    electron_spin_y = kroneker_product(
                        previous_spin_identity, electron_spin_y)
                    electron_spin_z = kroneker_product(
                        previous_spin_identity, electron_spin_z)
                    electron_spin_identity = kroneker_product(
                        previous_spin_identity, electron_spin_identity)

                previous_spin_identity = electron_spin_identity

                operator_list.append(electron_spin_x)
                operator_list.append(electron_spin_y)
                operator_list.append(electron_spin_z)

                operator_names.append(f"S{atom_index}x")
                operator_names.append(f"S{atom_index}y")
                operator_names.append(f"S{atom_index}z")

                # Electron Zeeman
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
                    nuclear_spin = atom["I"]

                    nuclear_spin_dimension = int(2*nuclear_spin + 1)
                    hilbert_space_shape.append(nuclear_spin_dimension)

        if verbose:
            plt.figure(label="spins")
            plot_rows = int(math.ceil(math.sqrt(len(operator_list))))
            for operator_index, (operator, operator_name) in \
                    enumerate(zip(operator_list, operator_names)):
                plt.subplot(plot_rows, plot_rows, operator_index + 1)
                plt.imshow(colour_complex_matrix(
                    operator/np.max(operator)))
                plt.title(operator_name)
                plt.gca().set_axis_off()
            plt.draw()


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
    generate_atoms([[{"S": 2, "g": 2, "g_perp": 2.1}, {"S": 1}]], verbose=True)

    plt.draw()
