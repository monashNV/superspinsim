def main():
    import numpy as np
    import simos
    from superspinsim.nv.lindbladians import contrast
    from superspinsim.util import colour_complex_matrix
    from matplotlib import pyplot as plt

    simos_method = "numpy"

    # Get experiment parameters from superspinsim -----------------------------
    contrast_params, coefficients = contrast()
    quiescent_magnetic_field = contrast_params["quiescent_magnetic_field"]

    # Get model from simOS ----------------------------------------------------
    # nv_system = simos.NV.NVSystem(nitrogen=False, method=simos_method)

    # hamiltonians_quiescent = nv_system.field_hamiltonian(
    #     Bvec=quiescent_magnetic_field)
    # hamiltonians_mw = nv_system.field_hamiltonian(
    #     Bvec=quiescent_magnetic_field + np.array([1, 0, 0]))
    # hamiltonians_mw_diff = [
    #     hamiltonian_mw - hamiltonian_quiescent for
    #     hamiltonian_mw, hamiltonian_quiescent
    #     in zip(hamiltonians_mw, hamiltonians_quiescent)
    # ]
    # hamiltonian_quiescent = sum(hamiltonians_quiescent)
    # hamiltonian_mw_diff = sum(hamiltonians_mw_diff)

    # jumps_quiescent, _ = nv_system.transition_operators(
    #     beta=0, Bvec=quiescent_magnetic_field)
    # jumps_laser, _ = nv_system.transition_operators(
    #     beta=1, Bvec=quiescent_magnetic_field)
    # jumps_laser_diff = [
    #     jump_laser - jump_quiescent for jump_laser, jump_quiescent
    #     in zip(jumps_laser, jumps_quiescent)
    # ]

    # # time_step = 100e-12
    # time_step = 1e-9
    # # density_initial = simos.thermal_state(hamiltonian_quiescent, T=300)
    # # density_initial = np.zeros((7, 7))
    # # density_initial[:3, :3] = 1/3*np.eye(3)

    # density = simos.prop(
    #     H0=hamiltonian_quiescent,
    #     H1=hamiltonians_mw_diff,
    #     # rho=density_initial,
    #     carr1=[np.array([1e-3]) for _ in hamiltonians_mw_diff],
    #     c_ops=jumps_quiescent,
    #     c_ops2=jumps_laser_diff,
    #     carr2=[np.array([0.1]) for _ in jumps_laser_diff],
    #     dt=time_step,
    #     # engine="cpu"
    # )

    # print(density.shape)
    # print(density)

    # quantum system of a single electron spin
    parameters = {'name': 'S', 'val': 1}
    system = simos.System([parameters], method="numpy")
    # the initial state, polarized electron spin
    density_initial = np.zeros((3, 3))
    density_initial[0, 0] = 1
    # rotate state with a pi/2 pulse
    # density_initial = simos.rot(system.Sx, np.pi/2, density_initial)
    # evolve the state
    time_step = 1e-6  # time step [s]
    magnetic_field = 10e-3  # magnetic field [mT]
    hamiltonian = magnetic_field*simos.ye*system.Sz  # the Zeeman Hamiltonian
    hamiltonian_1 = magnetic_field*simos.ye*system.Sx  # the Zeeman Hamiltonian

    density = simos.prop(
        H0=hamiltonian,
        H1=[hamiltonian_1],
        carr1=[np.array([0, 1, 0])],
        dt=time_step,
        # rho=density_initial,

        # hamiltonian,
        # time_step,
        # density_initial,
        # c_ops=[hamiltonian/1e3],
        # c_ops2=[hamiltonian/1e3],
        # carr2=[np.array([1, 1, 1])],
        # engine="parament"
    )

    plt.figure(label="simos")
    plt.imshow(colour_complex_matrix(density/np.max(np.abs(density))))
    plt.show()
