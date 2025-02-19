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

    if time > 4.0e-6:
        # phase = parameters.zero_field_splitting_ground \
        #     * (1 + 0.5*((time - 500e-6)/1e-3)/2)*(time - 500e-6)
        phase = 2.8e9*time + 100e6*time**2/100e-6/2
        coefficient[0] = 1e4*math.tau*math.cos(phase)
        coefficient[4] = 1e4*math.tau*math.cos(phase)

    if time > 2.5e-6:
        coefficient[16] = 0.001*parameters.spin_conserving_decay_rate
        coefficient[17] = 0.001*parameters.spin_nonconserving_decay_rate


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
        coefficient[18] = 0.1*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.1*parameters.spin_nonconserving_decay_rate

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
        coefficient[18] = 0.1*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.1*parameters.spin_nonconserving_decay_rate

    elif time < 6e-6:
        # Settle
        pass

    elif time < 6e-6 + 1e-6/2:
        # MWs
        coefficient[0] = 1e6*math.tau*math.cos(
            (parameters.zero_field_splitting_ground
                + 0.1*parameters.longitudinal_gyromagnetic_ratio_ground)*time)
        coefficient[4] = 1e6*math.tau*math.cos(
            (parameters.zero_field_splitting_ground
                + 0.1*parameters.longitudinal_gyromagnetic_ratio_ground)*time)

    elif time < 7e-6:
        # Wait
        pass

    elif time < 9e-6:
        # polarise
        coefficient[18] = 0.1*parameters.spin_conserving_decay_rate
        coefficient[19] = 0.1*parameters.spin_nonconserving_decay_rate
