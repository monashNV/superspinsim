import benchmark

# import superspinsim as s3
# import superspinsim.nv.lindbladians as nvl

# import numpy as np
# import matplotlib.pyplot as plt

if __name__ == "__main__":
    benchmark.basic.main()

    # from pogger import Pogger as Logger

    # logger = Logger("superspinsim-benchmarks")

    # @logger.record(
    #     ("time", "density_operator", "fluorescence"),
    #     ("s", None, "au")
    # )
    # def simulate_continuous_rabi(lindbladian):
    #     simulator = s3.generate_simulator(
    #         lindbladian,
    #         number_of_exponentials=1,
    #         number_of_fine_divisions=100,
    #         use_cayley=False
    #     )

    #     density_operator_initial = np.zeros((7, 7, 2), dtype=np.float64)
    #     # density_operator_initial[0, 0, 0] = 1
    #     density_operator_initial[1, 1, 0] = 1
    #     # density_operator_initial[2, 2, 0] = 1

    #     time_step_coarse = 10e-9
    #     time, density_operator = simulator(
    #         density_operator_initial, 0, 100e-6, time_step_coarse)

    #     fluorescence = density_operator[:, 3, 3, 0] + \
    #         density_operator[:, 4, 4, 0] + density_operator[:, 5, 5, 0]

    #     lowpass_length_time = 10e-6
    #     lowpass_type = None

    #     if lowpass_type is not None:
    #         if lowpass_type == "gaussian":
    #             lowpass_time = np.arange(
    #                 -lowpass_length_time, lowpass_length_time, time_step_coarse)
    #             lowpass_filter = np.exp((lowpass_time/(lowpass_length_time/8))**2/2)
    #         elif lowpass_type == "linear":
    #             lowpass_length = int(lowpass_length_time/time_step_coarse)
    #             lowpass_filter = np.ones(lowpass_length, np.float64)
    #         if np.sum(lowpass_filter) > 0:
    #             lowpass_filter /= np.sum(lowpass_filter)
    #         fluorescence = np.convolve(fluorescence, lowpass_filter, "same")

    #     fluorescence_max = np.max(fluorescence)
    #     if fluorescence_max > 0:
    #         fluorescence /= fluorescence_max

    #     try:
    #         plot_window = np.logical_and(time > 10e-6, time < 90e-6)
    #         plt.figure(label="fluorescence")
    #         plt.plot(
    #             time[plot_window]/1e-6, fluorescence[plot_window]*100, "k-")
    #         plt.xlabel("Time (us)")
    #         plt.ylabel("Fluorescence (%)")
    #         plt.draw()

    #         # mw_frequency = 2.87e9*(1 + 0.5*(time - 5e-3)/10e-3)
    #         mw_frequency = 2.8e9 + 100e6*time/100e-6
    #         plt.figure(label="odmr")
    #         plt.plot(
    #             mw_frequency[plot_window]/1e9, fluorescence[plot_window]*100, "k-")
    #         plt.xlabel("MW frequency (GHz)")
    #         plt.ylabel("Fluorescence (%)")
    #         plt.draw()
    #     finally:
    #         return time, density_operator, fluorescence

    # # logger.set_context("continuous_rabi")
    # # simulate_continuous_rabi(nvl.continuous_rabi)

    # logger.set_context("odmr")
    # simulate_continuous_rabi(nvl.odmr)

    # plt.show()
