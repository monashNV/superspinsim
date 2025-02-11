import benchmark

import superspinsim as s3
import superspinsim.nv.lindbladians as nvl

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # benchmark.basic.main()

    from pogger import Pogger as Logger

    logger = Logger("superspinsim-benchmarks")

    @logger.record(("time", "density_operator", "fluorescence"), ("s", None, "au"))
    def simulate_continuous_rabi(lindbladian):
        simulator = s3.generate_simulator(
            lindbladian,
            number_of_exponentials=2,
            number_of_fine_divisions=100,
            use_cayley=False
        )

        density_operator_initial = np.zeros((7, 7, 2), dtype=np.float64)
        density_operator_initial[0, 0, 0] = 1/3
        density_operator_initial[1, 1, 0] = 1/3
        density_operator_initial[2, 2, 0] = 1/3

        time, density_operator = simulator(density_operator_initial, 0, 1e-3, 100e-9)

        fluorescence = density_operator[:, 3, 3, 0] + \
            density_operator[:, 4, 4, 0] + density_operator[:, 5, 5, 0]

        lowpass_length = 200
        lowpass_filter = np.ones(lowpass_length, np.float64)/lowpass_length
        fluorescence = np.convolve(fluorescence, lowpass_filter, "same")

        fluorescence_max = np.max(fluorescence)
        if fluorescence_max > 0:
            fluorescence /= fluorescence_max

        try:
            plot_window = np.logical_and(time > 50e-6, time < 950e-6)
            plt.figure(label="fluorescence")
            plt.plot(time[plot_window]/1e-6, fluorescence[plot_window]*100, "k-")
            plt.xlabel("Time (us)")
            plt.ylabel("Fluorescence (%)")
            plt.draw()

        finally:
            return time, density_operator, fluorescence

    # logger.set_context("continuous_rabi")
    # simulate_continuous_rabi(nvl.continuous_rabi)

    logger.set_context("odmr")
    simulate_continuous_rabi(nvl.odmr)

    plt.show()
