def main():
    import numpy as np
    import simos
    from superspinsim.nv.lindbladians import contrast

    # Get experiment parameters from superspinsim -----------------------------
    contrast_params, coefficients = contrast()
    quiescent_magnetic_field = contrast_params["quiescent_magnetic_field"]

    # Get model from simOS ----------------------------------------------------
    nv_system = simos.NV.NVSystem(nitrogen=False)

    hamiltonians_quiescent = nv_system.field_hamiltonian(
        Bvec=quiescent_magnetic_field)
    hamiltonians_mw = nv_system.field_hamiltonian(
        Bvec=quiescent_magnetic_field + np.array([1, 0, 0]))
    hamiltonians_mw_diff = [
        hamiltonian_mw - hamiltonian_quiescent for
        hamiltonian_mw, hamiltonian_quiescent
        in zip(hamiltonians_mw, hamiltonians_quiescent)
    ]
    hamiltonian_quiescent = sum(hamiltonians_quiescent)
    hamiltonian_mw_diff = sum(hamiltonians_mw_diff)

    jumps_quiescent, _ = nv_system.transition_operators(
        beta=0, Bvec=quiescent_magnetic_field)
    jumps_laser, _ = nv_system.transition_operators(
        beta=1, Bvec=quiescent_magnetic_field)
    jumps_laser_diff = [
        jump_laser - jump_quiescent for jump_laser, jump_quiescent
        in zip(jumps_laser, jumps_quiescent)
    ]

    time_step = 100e-12

    density = simos.prop(
        H0=hamiltonian_quiescent,
        H1=hamiltonian_mw_diff,
        carr1=[1],
        c_ops=jumps_quiescent,
        c_ops2=jumps_laser_diff,
        carr2=[1]*len(jumps_laser_diff),
        dt=time_step,
        engine="parament"
    )
    print(density)
