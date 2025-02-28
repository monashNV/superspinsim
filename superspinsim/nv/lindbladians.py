import math
import superspinsim.nv.parameters as parameters


def laser(time, coefficient):
    # Default
    coefficient[3] = parameters.zero_field_splitting_ground
    coefficient[7] = parameters.zero_field_splitting_excited
    coefficient[8] = 1/parameters.spin_lattice_relaxation_time_ground
    coefficient[9] = 1/parameters.spin_spin_relaxation_time_ground
    coefficient[10] = 1/parameters.spin_lattice_relaxation_time_excited
    coefficient[11] = 1/parameters.spin_spin_relaxation_time_excited
    coefficient[12] = parameters.z_to_singlet_decay_rate
    coefficient[13] = parameters.pm_to_singlet_decay_rate
    coefficient[14] = parameters.singlet_to_z_decay_rate
    coefficient[15] = parameters.singlet_to_pm_decay_rate
    coefficient[16] = parameters.spin_conserving_decay_rate
    coefficient[17] = parameters.spin_nonconserving_decay_rate

    # Laser
    coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
    coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate


def continuous_rabi(time, coefficient):
    # Default
    coefficient[3] = parameters.zero_field_splitting_ground
    coefficient[7] = parameters.zero_field_splitting_excited
    coefficient[8] = 1/parameters.spin_lattice_relaxation_time_ground
    coefficient[9] = 1/parameters.spin_spin_relaxation_time_ground
    coefficient[10] = 1/parameters.spin_lattice_relaxation_time_excited
    coefficient[11] = 1/parameters.spin_spin_relaxation_time_excited
    coefficient[12] = parameters.z_to_singlet_decay_rate
    coefficient[13] = parameters.pm_to_singlet_decay_rate
    coefficient[14] = parameters.singlet_to_z_decay_rate
    coefficient[15] = parameters.singlet_to_pm_decay_rate
    coefficient[16] = parameters.spin_conserving_decay_rate
    coefficient[17] = parameters.spin_nonconserving_decay_rate

    # Microwaves
    if time > 200e-6 and time < 900e-6:
        coefficient[0] = 1000e3*math.tau*math.cos(
            parameters.zero_field_splitting_ground*time)

    # Laser
    if time > 100e-6:
        coefficient[16] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[17] = 0.5*parameters.spin_nonconserving_decay_rate


def odmr_block(time, coefficient):
    # Default
    coefficient[3] = parameters.zero_field_splitting_ground
    coefficient[7] = parameters.zero_field_splitting_excited
    coefficient[8] = 1/parameters.spin_lattice_relaxation_time_ground
    coefficient[9] = 1/parameters.spin_spin_relaxation_time_ground
    coefficient[10] = 1/parameters.spin_lattice_relaxation_time_excited
    coefficient[11] = 1/parameters.spin_spin_relaxation_time_excited
    coefficient[12] = parameters.z_to_singlet_decay_rate
    coefficient[13] = parameters.pm_to_singlet_decay_rate
    coefficient[14] = parameters.singlet_to_z_decay_rate
    coefficient[15] = parameters.singlet_to_pm_decay_rate
    coefficient[16] = parameters.spin_conserving_decay_rate
    coefficient[17] = parameters.spin_nonconserving_decay_rate

    # Microwaves
    if time > 200e-6 and time < 400e-6:
        coefficient[0] = 10e3*math.tau*math.cos(
            parameters.zero_field_splitting_ground/2*time)
    elif time < 600e-6:
        coefficient[0] = 10e3*math.tau*math.cos(
            parameters.zero_field_splitting_ground*time)
    elif time < 800e-6:
        coefficient[0] = 10e3*math.tau*math.cos(
            3/2*parameters.zero_field_splitting_ground*time)
    coefficient[4] = coefficient[0]

    # Laser
    if time > 25e-6:
        coefficient[16] = 0.01*parameters.spin_conserving_decay_rate
        coefficient[17] = 0.01*parameters.spin_nonconserving_decay_rate


def odmr(time, coefficient):
    # Default
    coefficient[3] = parameters.zero_field_splitting_ground
    coefficient[7] = parameters.zero_field_splitting_excited
    coefficient[8] = 1/parameters.spin_lattice_relaxation_time_ground
    coefficient[9] = 1/parameters.spin_spin_relaxation_time_ground
    coefficient[10] = 1/parameters.spin_lattice_relaxation_time_excited
    coefficient[11] = 1/parameters.spin_spin_relaxation_time_excited
    coefficient[12] = parameters.z_to_singlet_decay_rate
    coefficient[13] = parameters.pm_to_singlet_decay_rate
    coefficient[14] = parameters.singlet_to_z_decay_rate
    coefficient[15] = parameters.singlet_to_pm_decay_rate
    coefficient[16] = parameters.spin_conserving_decay_rate
    coefficient[17] = parameters.spin_nonconserving_decay_rate

    # coefficient[2] = 0.001*parameters.longitudinal_gyromagnetic_ratio_ground
    # coefficient[6] = 0.001*parameters.gyromagnetic_ratio_excited

    # Polarise
    coefficient[18] = 0.01*parameters.spin_conserving_decay_rate
    coefficient[19] = 0.01*parameters.spin_nonconserving_decay_rate

    if time > 2e-6:
        phase = 2.8e9*(time - 2e-6) + 100e6*(time - 2e-6)**2/100e-6/2
        coefficient[0] = 1e6*math.tau*math.cos(phase)
        coefficient[4] = 1e6*math.tau*math.cos(phase)


def rabi(time, coefficient):
    # Default
    coefficient[3] = parameters.zero_field_splitting_ground
    coefficient[7] = parameters.zero_field_splitting_excited
    coefficient[8] = 1/parameters.spin_lattice_relaxation_time_ground
    coefficient[9] = 1/parameters.spin_spin_relaxation_time_ground
    coefficient[10] = 1/parameters.spin_lattice_relaxation_time_excited
    coefficient[11] = 1/parameters.spin_spin_relaxation_time_excited
    coefficient[12] = parameters.z_to_singlet_decay_rate
    coefficient[13] = parameters.pm_to_singlet_decay_rate
    coefficient[14] = parameters.singlet_to_z_decay_rate
    coefficient[15] = parameters.singlet_to_pm_decay_rate
    coefficient[16] = parameters.spin_conserving_decay_rate
    coefficient[17] = parameters.spin_nonconserving_decay_rate

    # Bias
    coefficient[2] = 0.1*parameters.longitudinal_gyromagnetic_ratio_ground
    coefficient[6] = 0.1*parameters.gyromagnetic_ratio_excited

    # Pulse sequence
    if time < 2e-6:
        # polarise
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate

    elif time < 2.5e-6:
        # Settle
        pass

    elif time < 2.5e-6 + 1e-6/4:
        pass
        # # MWs
        # coefficient[0] = 1e6*math.tau*math.cos(
        #     parameters.zero_field_splitting_ground*time)
        # coefficient[4] = 1e6*math.tau*math.cos(
        #     parameters.zero_field_splitting_ground*time)

    elif time < 3.5e-6:
        # Wait
        pass

    elif time < 5.5e-6:
        # polarise
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate

    elif time < 6e-6:
        # Settle
        pass

    elif time < 6e-6 + 1e-6/2:
        # MWs
        b1_field = 1e6*math.tau/parameters.transverse_gyromagnetic_ratio_ground
        b1_frequency = parameters.zero_field_splitting_ground \
            + 0.1*parameters.longitudinal_gyromagnetic_ratio_ground

        coefficient[0] = \
            1/math.sqrt(2) \
            * b1_field*parameters.transverse_gyromagnetic_ratio_ground \
            * (1 - math.cos(math.tau*(time - 6e-6)/(1e-6/2))) \
            * math.cos(b1_frequency*time)

        coefficient[1] = \
            1/math.sqrt(2) \
            * b1_field*parameters.transverse_gyromagnetic_ratio_ground \
            * (1 - math.cos(math.tau*(time - 6e-6)/(1e-6/2))) \
            * math.cos(b1_frequency*time)

        coefficient[4] = \
            1/math.sqrt(2) \
            * b1_field*parameters.gyromagnetic_ratio_excited \
            * (1 - math.cos(math.tau*(time - 6e-6)/(1e-6/2))) \
            * math.cos(b1_frequency*time)

        coefficient[5] = \
            1/math.sqrt(2) \
            * b1_field*parameters.gyromagnetic_ratio_excited \
            * (1 - math.cos(math.tau*(time - 6e-6)/(1e-6/2))) \
            * math.cos(b1_frequency*time)

    elif time < 7e-6:
        # Wait
        pass

    elif time < 9e-6:
        # polarise
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate


def rabi_excited(time, coefficient):
    # Default
    coefficient[3] = parameters.zero_field_splitting_ground
    coefficient[7] = parameters.zero_field_splitting_excited
    coefficient[8] = 1/parameters.spin_lattice_relaxation_time_ground
    coefficient[9] = 1/parameters.spin_spin_relaxation_time_ground
    coefficient[10] = 1/parameters.spin_lattice_relaxation_time_excited
    coefficient[11] = 1/parameters.spin_spin_relaxation_time_excited
    coefficient[12] = parameters.z_to_singlet_decay_rate
    coefficient[13] = parameters.pm_to_singlet_decay_rate
    coefficient[14] = parameters.singlet_to_z_decay_rate
    coefficient[15] = parameters.singlet_to_pm_decay_rate
    coefficient[16] = parameters.spin_conserving_decay_rate
    coefficient[17] = parameters.spin_nonconserving_decay_rate

    # Bias
    coefficient[2] = 0.1*parameters.longitudinal_gyromagnetic_ratio_ground
    coefficient[6] = 0.1*parameters.gyromagnetic_ratio_excited

    # Pulse sequence
    if time < 2e-6:
        # polarise
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate

    elif time < 2.49e-6:
        # Settle
        pass

    elif time < 2.5e-6:
        # Pump
        coefficient[18] = 4*parameters.spin_conserving_decay_rate
        coefficient[19] = 4*parameters.spin_nonconserving_decay_rate

    elif time < 2.5e-6 + 1e-6/4:
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate
        # Wait
        # pass

    elif time < 3e-6:
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate
        # Wait
        # pass

    elif time < 5e-6:
        # polarise
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate

    elif time < 5.49e-6:
        # Settle
        pass

    elif time < 5.5e-6:
        # Pump
        coefficient[18] = 4*parameters.spin_conserving_decay_rate
        coefficient[19] = 4*parameters.spin_nonconserving_decay_rate

    elif time < 5.5e-6 + 1e-6/200:
        # MWs
        b1_field = 50e6*math.tau/parameters.transverse_gyromagnetic_ratio_ground
        b1_frequency = parameters.zero_field_splitting_excited \
            + 0.1*parameters.gyromagnetic_ratio_excited

        coefficient[0] = \
            1/math.sqrt(2) \
            * b1_field*parameters.transverse_gyromagnetic_ratio_ground \
            * (1 - math.cos(math.tau*(time - 5.5e-6)/(1e-6/200))) \
            * math.cos(b1_frequency*time)

        coefficient[1] = \
            1/math.sqrt(2) \
            * b1_field*parameters.transverse_gyromagnetic_ratio_ground \
            * (1 - math.cos(math.tau*(time - 5.5e-6)/(1e-6/200))) \
            * math.cos(b1_frequency*time)

        coefficient[4] = \
            1/math.sqrt(2) \
            * b1_field*parameters.gyromagnetic_ratio_excited \
            * (1 - math.cos(math.tau*(time - 5.5e-6)/(1e-6/200))) \
            * math.cos(b1_frequency*time)

        coefficient[5] = \
            1/math.sqrt(2) \
            * b1_field*parameters.gyromagnetic_ratio_excited \
            * (1 - math.cos(math.tau*(time - 5.5e-6)/(1e-6/200))) \
            * math.cos(b1_frequency*time)

        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate
        # # MWs
        # coefficient[0] = 1e6*math.tau*math.cos(
        #     parameters.zero_field_splitting_ground*time)
        # coefficient[4] = 1e6*math.tau*math.cos(
        #     parameters.zero_field_splitting_ground*time)

    else:
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate


def rabi_extended(time, coefficient):
    # Default
    coefficient[3] = parameters.zero_field_splitting_ground
    coefficient[7] = parameters.zero_field_splitting_excited
    coefficient[8] = 1/parameters.spin_lattice_relaxation_time_ground
    coefficient[9] = 1/parameters.spin_spin_relaxation_time_ground
    coefficient[10] = 1/parameters.spin_lattice_relaxation_time_excited
    coefficient[11] = 1/parameters.spin_spin_relaxation_time_excited
    coefficient[12] = parameters.z_to_singlet_decay_rate
    coefficient[13] = parameters.pm_to_singlet_decay_rate
    coefficient[14] = parameters.singlet_to_z_decay_rate
    coefficient[15] = parameters.singlet_to_pm_decay_rate
    coefficient[16] = parameters.spin_conserving_decay_rate
    coefficient[17] = parameters.spin_nonconserving_decay_rate

    # Bias
    coefficient[2] = 1*parameters.longitudinal_gyromagnetic_ratio_ground
    coefficient[6] = 1*parameters.gyromagnetic_ratio_excited

    # Pulse sequence
    if time < 2e-6:
        # polarise
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate

    elif time < 2.5e-6:
        # Settle
        pass

    elif time < 2.5e-6 + 1e-6/4:
        pass
        # # MWs
        # coefficient[0] = 1e6*math.tau*math.cos(
        #     parameters.zero_field_splitting_ground*time)
        # coefficient[4] = 1e6*math.tau*math.cos(
        #     parameters.zero_field_splitting_ground*time)

    elif time < 3.5e-6:
        # Wait
        pass

    elif time < 5.5e-6:
        # polarise
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate

    elif time < 6e-6:
        # Settle
        pass

    elif time < 6e-6 + 1e-6/2:
        # MWs
        b1_field = 1e6*math.tau/parameters.transverse_gyromagnetic_ratio_ground
        b1_frequency = parameters.zero_field_splitting_ground \
            + 1*parameters.longitudinal_gyromagnetic_ratio_ground

        coefficient[0] = \
            1/math.sqrt(2) \
            * b1_field*parameters.transverse_gyromagnetic_ratio_ground \
            * (1 - math.cos(math.tau*(time - 6e-6)/(1e-6/2))) \
            * math.cos(b1_frequency*time)

        coefficient[1] = \
            1/math.sqrt(2) \
            * b1_field*parameters.transverse_gyromagnetic_ratio_ground \
            * (1 - math.cos(math.tau*(time - 6e-6)/(1e-6/2))) \
            * math.cos(b1_frequency*time)

        coefficient[4] = \
            1/math.sqrt(2) \
            * b1_field*parameters.gyromagnetic_ratio_excited \
            * (1 - math.cos(math.tau*(time - 6e-6)/(1e-6/2))) \
            * math.cos(b1_frequency*time)

        coefficient[5] = \
            1/math.sqrt(2) \
            * b1_field*parameters.gyromagnetic_ratio_excited \
            * (1 - math.cos(math.tau*(time - 6e-6)/(1e-6/2))) \
            * math.cos(b1_frequency*time)

    elif time < 7e-6:
        # Wait
        pass

    elif time < 9e-6:
        # polarise
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate

    elif time < 10e-6:
        # MWs
        b1_field = 10e6*math.tau \
            / parameters.transverse_gyromagnetic_ratio_ground
        b1_frequency = parameters.zero_field_splitting_excited \
            + 1*parameters.gyromagnetic_ratio_excited

        coefficient[0] = \
            - 1/math.sqrt(2) \
            * b1_field*parameters.transverse_gyromagnetic_ratio_ground \
            * (1 - math.cos(math.tau*(time - 9e-6)/(1e-6))) \
            * math.cos(b1_frequency*time)

        coefficient[1] = \
            1/math.sqrt(2) \
            * b1_field*parameters.transverse_gyromagnetic_ratio_ground \
            * (1 - math.cos(math.tau*(time - 9e-6)/(1e-6))) \
            * math.cos(b1_frequency*time)

        coefficient[4] = \
            - 1/math.sqrt(2) \
            * b1_field*parameters.gyromagnetic_ratio_excited \
            * (1 - math.cos(math.tau*(time - 9e-6)/(1e-6))) \
            * math.cos(b1_frequency*time)

        coefficient[5] = \
            1/math.sqrt(2) \
            * b1_field*parameters.gyromagnetic_ratio_excited \
            * (1 - math.cos(math.tau*(time - 9e-6)/(1e-6))) \
            * math.cos(b1_frequency*time)

        # polarise
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate

    elif time < 11e-6:
        # polarise
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate

    elif time < 12e-6:
        # MWs
        b1_field = 10e6*math.tau \
            / parameters.transverse_gyromagnetic_ratio_ground
        b1_frequency = parameters.zero_field_splitting_excited \
            - 1*parameters.gyromagnetic_ratio_excited

        coefficient[0] = \
            1/math.sqrt(2) \
            * b1_field*parameters.transverse_gyromagnetic_ratio_ground \
            * (1 - math.cos(math.tau*(time - 11e-6)/(1e-6))) \
            * math.cos(b1_frequency*time)

        coefficient[1] = \
            - 1/math.sqrt(2) \
            * b1_field*parameters.transverse_gyromagnetic_ratio_ground \
            * (1 - math.cos(math.tau*(time - 11e-6)/(1e-6))) \
            * math.cos(b1_frequency*time)

        coefficient[4] = \
            1/math.sqrt(2) \
            * b1_field*parameters.gyromagnetic_ratio_excited \
            * (1 - math.cos(math.tau*(time - 11e-6)/(1e-6))) \
            * math.cos(b1_frequency*time)

        coefficient[5] = \
            - 1/math.sqrt(2) \
            * b1_field*parameters.gyromagnetic_ratio_excited \
            * (1 - math.cos(math.tau*(time - 11e-6)/(1e-6))) \
            * math.cos(b1_frequency*time)

        # polarise
        coefficient[18] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.5*parameters.spin_nonconserving_decay_rate


def coupling(time, coefficient):
    coefficient[0] = math.tau*1e5
    # coefficient[4] = math.tau*1e5
