def main():
    from util import save_density, compare_density

    import math
    import numpy as np

    from superspinsim import simspins

    # Define magnetic field (Rabi dressing)
    dressing_frequency = 28e6   # ~ on resonance
    dressing_amplitude = 10e-6  # 10 uT

    def mag_x(time):
        return dressing_amplitude*math.cos(math.tau*dressing_frequency*time)

    def mag_y(time):
        return 0.0

    def mag_z(time):
        return 0.0

    # Relevant only for defects
    def excitation(time):
        return 0.0

    # Define qubit
    spins = [[{
        "S": 1/2, "g": 2.0,           # (spin-half, electron g factor)
        "B0": np.array([0, 0, 1e-3])  # Bias magnetic field of 1 mT along z
    }]]

    # Define density matrix
    density_initial = np.zeros((2, 2), dtype=np.float64)
    density_initial[0, 0] = 1

    # Simulate
    time, density = simspins(
        [mag_x, mag_y, mag_z, excitation],  # Fields
        0, 10e-6, 1e-9,                     # Time start/stop/step
        spins, [{}], {},                    # Spin description
        density_initial,                    # Initial state
        use_rotating=False,
        use_residual=False,
        number_of_exponentials=1,
        number_of_fine_divisions=100,
        verbose=True
    )

    save_density("qubit-couple", density)
    compare_density("qubit-couple", density)


if __name__ == "__main__":
    main()
