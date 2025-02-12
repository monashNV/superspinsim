import math
import superspinsim.nv.parameters as parameters


def continuous_rabi(time, coefficient):
    # Default
    coefficient[3] = parameters.zero_field_splitting_ground
    coefficient[7] = parameters.zero_field_splitting_excited
    coefficient[8] = 1/parameters.spin_spin_relaxation_time_ground
    coefficient[9] = 1/parameters.spin_spin_relaxation_time_excited
    coefficient[10] = 1/parameters.spin_lattice_relaxation_time_ground
    coefficient[11] = parameters.z_to_singlet_decay_rate
    coefficient[12] = parameters.singlet_to_z_decay_rate
    coefficient[13] = parameters.pm_to_singlet_decay_rate
    coefficient[14] = parameters.spin_conserving_decay_rate
    coefficient[15] = parameters.spin_nonconserving_decay_rate

    # Microwaves
    if time > 200e-6 and time < 900e-6:
        coefficient[0] = 1000e3*math.tau*math.cos(parameters.zero_field_splitting_ground*time)

    # Laser
    if time > 100e-6:
        coefficient[16] = 0.5*parameters.spin_conserving_decay_rate
        coefficient[17] = 0.5*parameters.spin_nonconserving_decay_rate


def odmr_block(time, coefficient):
    # Default
    coefficient[3] = parameters.zero_field_splitting_ground
    coefficient[7] = parameters.zero_field_splitting_excited
    coefficient[8] = 1/parameters.spin_spin_relaxation_time_ground
    coefficient[9] = 1/parameters.spin_spin_relaxation_time_excited
    coefficient[10] = 1/parameters.spin_lattice_relaxation_time_ground
    coefficient[11] = parameters.z_to_singlet_decay_rate
    coefficient[12] = parameters.singlet_to_z_decay_rate
    coefficient[13] = parameters.pm_to_singlet_decay_rate
    coefficient[14] = parameters.spin_conserving_decay_rate
    coefficient[15] = parameters.spin_nonconserving_decay_rate

    # Microwaves
    if time > 200e-6 and time < 400e-6:
        coefficient[0] = 10e3*math.tau*math.cos(parameters.zero_field_splitting_ground/2*time)
    elif time < 600e-6:
        coefficient[0] = 10e3*math.tau*math.cos(parameters.zero_field_splitting_ground*time)
    elif time < 800e-6:
        coefficient[0] = 10e3*math.tau*math.cos(3/2*parameters.zero_field_splitting_ground*time)
    coefficient[4] = coefficient[0]

    # Laser
    if time > 25e-6:
        coefficient[16] = 0.01*parameters.spin_conserving_decay_rate
        coefficient[17] = 0.01*parameters.spin_nonconserving_decay_rate


def odmr(time, coefficient):
    # Default
    coefficient[3] = parameters.zero_field_splitting_ground
    coefficient[7] = parameters.zero_field_splitting_excited
    # coefficient[8] = 1/parameters.spin_spin_relaxation_time_ground
    # coefficient[9] = 1/parameters.spin_spin_relaxation_time_excited
    coefficient[8] = 1/600e-6
    coefficient[9] = 1/600e-6
    coefficient[10] = 1/parameters.spin_lattice_relaxation_time_ground
    coefficient[11] = parameters.z_to_singlet_decay_rate
    coefficient[12] = parameters.singlet_to_z_decay_rate
    coefficient[13] = parameters.pm_to_singlet_decay_rate
    coefficient[14] = parameters.spin_conserving_decay_rate
    coefficient[15] = parameters.spin_nonconserving_decay_rate

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
