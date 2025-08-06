import math
import numpy as np

from matplotlib import pyplot as plt
from cmcrameri import cm

from superspinsim import generate_simulator, params as s3p
from superspinsim.generate_generators import generate_7, generate_21

from pogger import Pogger


def main():
    hyperfine = False

    mw_amplitudes = [200e-6]  # np.geomspace(0.01e-6, 100e-6, 11)
    fluorescences = []
    carriers = np.array([
        2.770896e+09, 2.796657e+09, 2.840870e+09, 2.864824e+09, 2.882645e+09,
        2.905514e+09, 2.945065e+09, 2.966598e+09
    ])
    fm_modulation = np.array(
        [6300, 26900, 10300, 113800, 70300, 43500, 16600, 3900])
    fm_deviation = np.array([200e3]*8)

    dipole_moment_strength = 0.77
    dipole_rotation_frequency = 220
    dipole_moment_position = np.array(
        [-0.215, -0.01, -0.033], dtype=np.float64)
    magnetic_permeability_normalised = 0.99999999987*1e-7
    dipole_moment_direction_i = np.array([1, -1, 0], dtype=np.float64)
    dipole_moment_direction_q = np.array([0, 0, 1], dtype=np.float64)
    dipole_moment_i = \
        dipole_moment_strength*dipole_moment_direction_i \
        / np.linalg.norm(dipole_moment_direction_i)
    dipole_moment_q = \
        dipole_moment_strength*dipole_moment_direction_q \
        / np.linalg.norm(dipole_moment_direction_q)
    dipole_magnetic_field_i = magnetic_permeability_normalised*(
        -3*dipole_moment_position
        * np.dot(-dipole_moment_position, dipole_moment_i)
        / np.linalg.norm(dipole_moment_position)**5
        - dipole_moment_i/np.linalg.norm(dipole_moment_position)**3
    )
    dipole_magnetic_field_q = magnetic_permeability_normalised*(
        -3*dipole_moment_position
        * np.dot(-dipole_moment_position, dipole_moment_q)
        / np.linalg.norm(dipole_moment_position)**5
        - dipole_moment_q/np.linalg.norm(dipole_moment_position)**3
    )

    # time_sample = np.linspace(0, 100e-3, int(1e3))
    # magnetic_field_x = \
    #     np.cos(math.tau*dipole_rotation_frequency*time_sample) \
    #     * dipole_magnetic_field_i[0] \
    #     + np.sin(math.tau*dipole_rotation_frequency*time_sample) \
    #     * dipole_magnetic_field_q[0]
    # magnetic_field_y = \
    #     np.cos(math.tau*dipole_rotation_frequency*time_sample) \
    #     * dipole_magnetic_field_i[1] \
    #     + np.sin(math.tau*dipole_rotation_frequency*time_sample) \
    #     * dipole_magnetic_field_q[1]
    # magnetic_field_z = \
    #     np.cos(math.tau*dipole_rotation_frequency*time_sample) \
    #     * dipole_magnetic_field_i[2] \
    #     + np.sin(math.tau*dipole_rotation_frequency*time_sample) \
    #     * dipole_magnetic_field_q[2]

    # with Pogger("superspinsim-generate") as logger:
    #     @logger.record((), ())
    #     def draw_fields():
    #         plt.figure(label="dipole_fields")
    #         plt.plot(
    #             time_sample/1e-3, magnetic_field_x/1e-6, "-",
    #             color=cm.hawaii(0/3), label="x"
    #         )
    #         plt.plot(
    #             time_sample/1e-3, magnetic_field_y/1e-6, "-",
    #             color=cm.hawaii(1/3), label="y"
    #         )
    #         plt.plot(
    #             time_sample/1e-3, magnetic_field_z/1e-6, "-",
    #             color=cm.hawaii(2/3), label="z"
    #         )
    #         plt.xlabel("Time (ms)")
    #         plt.ylabel("Magnetic field (uT)")
    #         plt.legend()
    #         plt.draw()
    #     draw_fields()
    # return

    for mw_amplitude in mw_amplitudes:
        # time_start = 10e-6
        time_start = 0e-6
        # time_end = 100e-3
        time_end = 20e-3

        quiescent_magnetic_field = np.array(
            [3300.91, 723.18, 2026.51], dtype=np.float64)*1e-6

        coefficient_r_max = 0.1
        # quiescent_magnetic_field = np.array(
        #     [3300.91, 723.18, 2026.51], dtype=np.float64)*1e-6
        # quiescent_magnetic_field /= np.linalg.norm(quiescent_magnetic_field)
        # quiescent_magnetic_field *= 7e-3

        def coefficient_z(time):
            signal = 0.0
            signal += \
                np.cos(math.tau*dipole_rotation_frequency*time) \
                * dipole_magnetic_field_i[2] \
                + np.sin(math.tau*dipole_rotation_frequency*time) \
                * dipole_magnetic_field_q[2]
            for index in range(8):
                signal += math.cos(
                    math.tau*carriers[index]*time
                    + (fm_deviation[index]/fm_modulation[index])
                    * math.sin(math.tau*fm_modulation[index]*time)
                )
            signal *= mw_amplitude
            return signal

        def coefficient_y(time):
            signal = 0.0
            signal += \
                np.cos(math.tau*dipole_rotation_frequency*time) \
                * dipole_magnetic_field_i[1] \
                + np.sin(math.tau*dipole_rotation_frequency*time) \
                * dipole_magnetic_field_q[1]
            return signal

        def coefficient_x(time):
            signal = 0.0
            signal += \
                np.cos(math.tau*dipole_rotation_frequency*time) \
                * dipole_magnetic_field_i[0] \
                + np.sin(math.tau*dipole_rotation_frequency*time) \
                * dipole_magnetic_field_q[0]
            return signal

        # def coefficient_r(time):
        #     return 0.01

        t100_from_t111 = s3p.PointGroups.T.Transforms.T100_from_T111
        coefficients = [
            coefficient_x, coefficient_y, coefficient_z]
        orientations = [
             [t100_from_t111@s3p.PointGroups.T.T111.a1p],
             [t100_from_t111@s3p.PointGroups.T.T111.a3p],
             [t100_from_t111@s3p.PointGroups.T.T111.b2p],
             [t100_from_t111@s3p.PointGroups.T.T111.b4p],
        ]
        laser_direction = np.array([3, 2, 1], dtype=np.float64)
        laser_direction /= np.linalg.norm(laser_direction)
        for orientation in orientations:
            factor = 1
            # factor = \
            #     (laser_direction.T@orientations[0][0]@laser_direction)**2 \
            #     + 0.25
            # coefficient_r = \
            #     lambda time: coefficient_r_max*(factor + 1)/2
            coefficient_r = \
                lambda time: coefficient_r_max*factor

            # def coefficient_r(time):
            #     return coefficient_r_max*(factor + 1)/2

            orientation.append(coefficients + [coefficient_r])
            print(orientation)

        fluorescence_total = None
        for (orientation, coefficients) in orientations:
            if hyperfine:
                lindbladian, generators, vectorisation_map = generate_21(
                    coefficients, quiescent_magnetic_field)
            else:
                lindbladian, generators, vectorisation_map, vectors_real, \
                    inv_vectors_real, doubles, singles = generate_7(
                        coefficients, quiescent_magnetic_field,
                        use_rotating=True,
                        rotation_magnetic_from_atom=orientation
                    )

            fine_step = 200e-12
            if hyperfine:
                coarse_step = 150e-9
            else:
                coarse_step = 150e-9
            number_of_divisions = int(round(coarse_step/fine_step))
            coarse_step = number_of_divisions*fine_step

            simulator = generate_simulator(
                lindbladian, np.array(generators), vectorisation_map,
                number_of_fine_divisions=number_of_divisions,
                number_of_exponentials=5, use_rotating=True,
                vectors_real=vectors_real, inv_vectors_real=inv_vectors_real,
                doubles=doubles, singles=singles
            )
            if hyperfine:
                density_operator_initial = np.zeros((21, 21))
                density_operator_initial[:9, :9] = 1/9*np.eye(9)
            else:
                density_operator_initial = np.zeros((7, 7))
                density_operator_initial[:3, :3] = 1/3*np.eye(3)

            # density_operator_initial[1, 1, 0] = 1/2
            # density_operator_initial[0, 0, 0] = 1/4
            # density_operator_initial[2, 2, 0] = 1/4
            # density_operator_initial[0, 2, 0] = 1/4
            # density_operator_initial[0, 2, 0] = 1/4
            # density_operator_initial[0, 1, 0] = 1/(2*math.sqrt(2))
            # density_operator_initial[1, 0, 0] = 1/(2*math.sqrt(2))
            # density_operator_initial[1, 2, 0] = 1/(2*math.sqrt(2))
            # density_operator_initial[2, 1, 0] = 1/(2*math.sqrt(2))

            time, density = simulator(
                density_operator_initial, 0, time_end, coarse_step)

            if hyperfine:
                fluorescence = \
                    density[:, 9, 9] + density[:, 10, 10] \
                    + density[:, 11, 11] + density[:, 12, 12] \
                    + density[:, 13, 13] + density[:, 14, 14] \
                    + density[:, 15, 15] + density[:, 16, 16] \
                    + density[:, 17, 17]
            else:
                fluorescence = \
                    density[:, 3, 3] + density[:, 4, 4] + density[:, 5, 5]
            fluorescence = np.real(fluorescence)

            if fluorescence_total is None:
                fluorescence_total = fluorescence
            else:
                fluorescence_total += fluorescence

        fluorescences.append(fluorescence_total)

    with Pogger("superspinsim-generate") as logger:
        @logger.record(
            (
                "time", "mw_amplitude", "fluorescences"
            ), ("s", "T", None, "Hz", None))
        def plot():
            plt.figure("odmr")
            for fluorescence_index, (fluorescence, mw_amplitude) \
                    in enumerate(zip(fluorescences, mw_amplitudes)):
                plt.plot(
                    time/1e-6, 100*fluorescence/np.max(fluorescence), "-",
                    color=cm.hawaii(fluorescence_index/len(fluorescences)),
                    label=f"MW: {mw_amplitude/1e-6:.2f} mT"
                )
            plt.xlabel("Time (us)")
            plt.ylabel("Fluorescence (%)")
            plt.legend()
            plt.gca().spines[["top", "right"]].set_visible(False)
            plt.draw()

            return (
                time, np.array(mw_amplitudes), np.array(fluorescences)
            )

        if hyperfine:
            logger.set_context("21-level")
        else:
            logger.set_context("7-level")
        plot()

    plt.show()
