import math
import numpy as np

from pogger import Pogger


meta_datatype = np.float128


_operator_generators = {}


def _generate_spin_x_g_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[0, 1, 0] = 1
    operator[1, 0, 0] = 1
    operator[1, 2, 0] = 1
    operator[2, 1, 0] = 1
    operator /= math.sqrt(2)
    return operator


_operator_generators["spin_x_g"] = _generate_spin_x_g_operator


def _generate_spin_y_g_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[0, 1, 1] = -1
    operator[1, 0, 1] = 1
    operator[1, 2, 1] = -1
    operator[2, 1, 1] = 1
    operator /= math.sqrt(2)
    return operator


_operator_generators["spin_y_g"] = _generate_spin_y_g_operator


def _generate_spin_z_g_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[0, 0, 0] = 1
    operator[2, 2, 0] = -1
    return operator


_operator_generators["spin_z_g"] = _generate_spin_z_g_operator


def _generate_spin_z2_g_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[0, 0, 0] = 1/3
    operator[1, 1, 0] = -2/3
    operator[2, 2, 0] = 1/3
    return operator


_operator_generators["spin_z2_g"] = _generate_spin_z2_g_operator


def _generate_spin_x_e_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[3, 4, 0] = 1
    operator[4, 3, 0] = 1
    operator[4, 5, 0] = 1
    operator[5, 4, 0] = 1
    operator /= math.sqrt(2)
    return operator


_operator_generators["spin_x_e"] = _generate_spin_x_e_operator


def _generate_spin_y_e_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[3, 4, 1] = -1
    operator[4, 3, 1] = 1
    operator[4, 5, 1] = -1
    operator[5, 4, 1] = 1
    operator /= math.sqrt(2)
    return operator


_operator_generators["spin_y_e"] = _generate_spin_y_e_operator


def _generate_spin_z_e_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[3, 3, 0] = 1
    operator[5, 5, 0] = -1
    return operator


_operator_generators["spin_z_e"] = _generate_spin_z_e_operator


def _generate_spin_z2_e_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[3, 3, 0] = 1/3
    operator[4, 4, 0] = -2/3
    operator[5, 5, 0] = 1/3
    return operator


_operator_generators["spin_z2_e"] = _generate_spin_z2_e_operator


def _generate_relax_p_g_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[1, 0, 0] = 1
    return operator


_operator_generators["relax_p_g"] = _generate_relax_p_g_operator


def _generate_relax_m_g_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[1, 2, 0] = 1
    return operator


_operator_generators["relax_m_g"] = _generate_relax_m_g_operator


def _generate_relax_t_g_operator():
    operator = _operator_generators["spin_z_g"]()
    # operator *= math.sqrt(2)
    return operator


_operator_generators["relax_t_g"] = _generate_relax_t_g_operator


def _generate_relax_p_e_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[4, 3, 0] = 1
    return operator


_operator_generators["relax_p_e"] = _generate_relax_p_e_operator


def _generate_relax_m_e_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[4, 5, 0] = 1
    return operator


_operator_generators["relax_m_e"] = _generate_relax_m_e_operator


def _generate_relax_t_e_operator():
    operator = _operator_generators["spin_z_e"]()
    # operator *= math.sqrt(2)
    return operator


_operator_generators["relax_t_e"] = _generate_relax_t_e_operator


def _generate_raise_p_p_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[3, 0, 0] = 1
    return operator


_operator_generators["raise_p_p"] = _generate_raise_p_p_operator


def _generate_raise_z_z_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[4, 1, 0] = 1
    return operator


_operator_generators["raise_z_z"] = _generate_raise_z_z_operator


def _generate_raise_m_m_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[5, 2, 0] = 1
    return operator


_operator_generators["raise_m_m"] = _generate_raise_m_m_operator


def _generate_raise_p_z_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[4, 0, 0] = 1
    return operator


_operator_generators["raise_p_z"] = _generate_raise_p_z_operator


def _generate_raise_z_p_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[3, 1, 0] = 1
    return operator


_operator_generators["raise_z_p"] = _generate_raise_z_p_operator


def _generate_raise_z_m_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[5, 1, 0] = 1
    return operator


_operator_generators["raise_z_m"] = _generate_raise_z_m_operator


def _generate_raise_m_z_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[4, 2, 0] = 1
    return operator


_operator_generators["raise_m_z"] = _generate_raise_m_z_operator


def _generate_decay_p_p_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[0, 3, 0] = 1
    return operator


_operator_generators["decay_p_p"] = _generate_decay_p_p_operator


def _generate_decay_z_z_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[1, 4, 0] = 1
    return operator


_operator_generators["decay_z_z"] = _generate_decay_z_z_operator


def _generate_decay_m_m_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[2, 5, 0] = 1
    return operator


_operator_generators["decay_m_m"] = _generate_decay_m_m_operator


def _generate_decay_p_z_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[1, 3, 0] = 1
    return operator


_operator_generators["decay_p_z"] = _generate_decay_p_z_operator


def _generate_decay_z_p_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[0, 4, 0] = 1
    return operator


_operator_generators["decay_z_p"] = _generate_decay_z_p_operator


def _generate_decay_z_m_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[2, 4, 0] = 1
    return operator


_operator_generators["decay_z_m"] = _generate_decay_z_m_operator


def _generate_decay_m_z_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[1, 5, 0] = 1
    return operator


_operator_generators["decay_m_z"] = _generate_decay_m_z_operator


def _generate_decay_p_s_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[6, 3, 0] = 1
    return operator


_operator_generators["decay_p_s"] = _generate_decay_p_s_operator


def _generate_decay_z_s_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[6, 4, 0] = 1
    return operator


_operator_generators["decay_z_s"] = _generate_decay_z_s_operator


def _generate_decay_m_s_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[6, 5, 0] = 1
    return operator


_operator_generators["decay_m_s"] = _generate_decay_m_s_operator


def _generate_decay_s_p_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[0, 6, 0] = 1
    return operator


_operator_generators["decay_s_p"] = _generate_decay_s_p_operator


def _generate_decay_s_z_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[1, 6, 0] = 1
    return operator


_operator_generators["decay_s_z"] = _generate_decay_s_z_operator


def _generate_decay_s_m_operator():
    operator = np.zeros((7, 7, 2), dtype=meta_datatype)
    operator[2, 6, 0] = 1
    return operator


_operator_generators["decay_s_m"] = _generate_decay_s_m_operator


def _generate_valid_indices():
    valid_mask = np.zeros((7, 7), dtype=meta_datatype)
    valid_mask[:3, :3] = 1
    valid_mask[3:6, 3:6] = 1
    valid_mask[6, 6] = 1

    valid_indices = []

    for y_index in range(7):
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


with Pogger(project_name="superspinsim-generate") as logger:
    @logger.record(("operators"))
    def _generate_operators(operator_generators):
        operators = {}
        for label, operator_generator in operator_generators.items():
            operators[label] = operator_generator()
        return operators

    @logger.record(("superoperators_all", "superoperators", "valid_indices"))
    def _generate_superoperators(operators, valid_indices):
        superoperators = {}
        for label, operator in operators.items():
            if "spin" in label:
                superoperators[label] = \
                    _generate_von_neumann(operator, valid_indices)
            else:
                superoperators[label] = \
                    _generate_dissipator(operator, valid_indices)

        superoperators_use = {}

        for label in [
                "spin_x_g", "spin_y_g", "spin_z_g", "spin_z2_g",
                "spin_x_e", "spin_y_e", "spin_z_e", "spin_z2_e"
                ]:
            superoperators_use[label] = superoperators[label]

        superoperators_use["relax_l_g"] = superoperators["relax_p_g"] \
            + superoperators["relax_m_g"]

        superoperators_use["relax_t_g"] = superoperators["relax_t_g"]

        superoperators_use["relax_l_e"] = superoperators["relax_p_e"] \
            + superoperators["relax_m_e"]

        superoperators_use["relax_t_e"] = superoperators["relax_t_e"]

        superoperators_use["decay_z_s"] = superoperators["decay_z_s"]

        superoperators_use["decay_pm_s"] = superoperators["decay_p_s"] \
            + superoperators["decay_m_s"]

        superoperators_use["decay_s_z"] = superoperators["decay_s_z"]

        superoperators_use["decay_s_pm"] = superoperators["decay_s_p"] \
            + superoperators["decay_s_m"]

        superoperators_use["decay_conserve"] = superoperators["decay_p_p"] \
            + superoperators["decay_z_z"] + superoperators["decay_m_m"]

        superoperators_use["decay_nonconserve"] = superoperators["decay_p_z"] \
            + 0.5*superoperators["decay_z_p"] + 0.5*superoperators["decay_m_m"] \
            + superoperators["decay_m_z"]

        superoperators_use["raise_conserve"] = superoperators["raise_p_p"] \
            + superoperators["raise_z_z"] + superoperators["raise_m_m"]

        superoperators_use["raise_nonconserve"] = superoperators["raise_p_z"] \
            + 0.5*superoperators["raise_z_p"] + 0.5*superoperators["raise_m_m"] \
            + superoperators["raise_m_z"]

        return superoperators, superoperators_use, valid_indices

    def _write_script(operators, superoperators, valid_indices):
        with open("generators.py", "w") as file:
            file.write("\"\"\"\nScript generated by "
                       "`generate_generators.py`.\n\"\"\"\n\n\n")
            file.write("import numpy as np\n\n\n")

            file.write("operators = {}\n\n")

            for label, operator in operators.items():
                file.write(f"operators[\"{label}\"] = np.array([\n")
                for y_index in range(operator.shape[0]):
                    file.write("[")
                    for x_index in range(operator.shape[1]):
                        file.write(f"[{operator[y_index, x_index, 0]}, ")
                        file.write(f"{operator[y_index, x_index, 1]}], ")
                    file.write("],\n")
                file.write("], dtype=np.float64)\n\n")

            file.write("superoperators = {}\n\n")

            for label, superoperator in superoperators.items():
                file.write(f"superoperators[\"{label}\"] = np.array([\n")
                for y_index in range(superoperator.shape[0]):
                    file.write("[")
                    for x_index in range(superoperator.shape[1]):
                        file.write(f"{superoperator[y_index, x_index]}, ")
                    file.write("],\n")
                file.write("], dtype=np.float64)\n\n")

            file.write("vectorisation_map = np.array([\n")
            for y_index in range(valid_indices.shape[0]):
                file.write("[")
                for x_index in range(valid_indices.shape[1]):
                    file.write(f"{valid_indices[y_index, x_index]}, ")
                file.write("],\n")

            file.write("], dtype=np.int32)\n\n")

    @logger.record()
    def _visualise(operators, superoperators, superoperators_use):
        from util import colour_complex_matrix as _colour_complex_matrix
        from matplotlib import pyplot as plt

        plt.figure(
            label="operators",
            figsize=(10, 8)
        )

        number_of_rows = 5
        for operator_index, (label, operator) in enumerate(
                operators.items()):
            plt.subplot(
                number_of_rows,
                math.ceil(len(operators)/number_of_rows),
                1 + operator_index
            )
            coloured = np.array(_colour_complex_matrix(operator), dtype=np.float64)
            plt.imshow(coloured)
            plt.axis("off")
            plt.title(label)

        plt.draw()

        number_of_rows = 5
        plt.figure(
            label="superoperators_all",
            figsize=(10, 8)
        )

        for operator_index, (label, superoperator) in enumerate(
                superoperators.items()):
            plt.subplot(
                number_of_rows,
                math.ceil(len(superoperators)/number_of_rows),
                1 + operator_index
            )
            coloured = np.array(
                _colour_complex_matrix(superoperator), dtype=np.float64)
            plt.imshow(coloured)
            plt.axis("off")
            plt.title(label)
        plt.draw()

        number_of_rows = 5
        plt.figure(
            label="superoperators",
            figsize=(10, 10)
        )

        for operator_index, (label, superoperator) in enumerate(
                superoperators_use.items()):
            plt.subplot(
                number_of_rows,
                math.ceil(len(superoperators_use)/number_of_rows),
                1 + operator_index
            )
            coloured = np.array(
                _colour_complex_matrix(superoperator), dtype=np.float64)
            plt.imshow(coloured)
            # plt.axis("off")
            plt.xticks([])
            plt.yticks([])
            plt.title(f"{operator_index:2d}:{label}")
            if operator_index == 0:
                plt.ylabel("Ground-state\ncoherent\n\n")
            elif operator_index == 4:
                plt.ylabel("Excited-state\ncoherent\n\n")
            elif operator_index == 8:
                plt.ylabel("Spin relaxation\n\n")
            elif operator_index == 12:
                plt.ylabel("Intersystem\n\n")
            elif operator_index == 16:
                plt.ylabel("Optical\n\n")

        plt.draw()

    logger.set_context("superoperators")
    _operators = _generate_operators(_operator_generators)
    _valid_indices = _generate_valid_indices()
    _superoperators, _superoperators_use, _ = \
        _generate_superoperators(_operators, _valid_indices)
    _visualise(_operators, _superoperators, _superoperators_use)


if __name__ == "__main__":
    _write_script(_operators, _superoperators_use, _valid_indices)
