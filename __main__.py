def main():
    import superspinsim as s3
    from superspinsim.generate_generators import generate_7

    import math
    import numpy as np
    from scipy import optimize as spo
    from matplotlib import pyplot as plt
    from cmcrameri import cm

    from pogger import Pogger as Logger

    hyperfine = False

    time_start = 1e-6
    time_end = 100e-6
    frequency_start = 2.82e9
    frequency_width = 100e6

    quiescent_magnetic_field = np.array([0, 0, 1])*1e-3

    def coefficient_x(time):
        if time > time_start:
            phase = math.tau*(
                frequency_start*(time - time_start)
                + frequency_width*((time - time_start)**2)
                / (time_end - time_start)/2
            )
            return 100e-6*math.sin(phase)
        else:
            return 0

    def coefficient_y(time):
        return 0

    def coefficient_z(time):
        return 0

    def coefficient_r(time):
        return 0.1

    coefficients = [coefficient_x, coefficient_y, coefficient_z, coefficient_r]
    if hyperfine:
        lindbladian, generators, vectorisation_map = generate_21(
            coefficients, quiescent_magnetic_field)
    else:
        lindbladian, generators, vectorisation_map = generate_7(
            coefficients, quiescent_magnetic_field)

    print(generators[0].shape)

    fine_step = 100e-12
    if hyperfine:
        coarse_step = 100e-9
    else:
        coarse_step = 1e-9
    number_of_divisions = int(round(coarse_step/fine_step))
        
    simulator = s3.generate_simulator(
        lindbladian, np.array(generators), vectorisation_map,
        number_of_fine_divisions=number_of_divisions, number_of_exponentials=3
    )
    if hyperfine:
        density_operator_initial = np.zeros((21, 21, 2))
        density_operator_initial[:9, :9, 0] = 1/9*np.eye(9)
    else:
        density_operator_initial = np.zeros((7, 7, 2))
        density_operator_initial[:3, :3, 0] = 1/3*np.eye(3)

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
        fluorescense = \
            density[:, 9, 9, 0] + density[:, 10, 10, 0] \
            + density[:, 11, 11, 0] + density[:, 12, 12, 0] \
            + density[:, 13, 13, 0] + density[:, 14, 14, 0] \
            + density[:, 15, 15, 0] + density[:, 16, 16, 0] \
            + density[:, 17, 17, 0]
    else:
        fluorescense = \
            density[:, 3, 3, 0] + density[:, 4, 4, 0] + density[:, 5, 5, 0]

    with Logger("superspinsim-generate") as logger:
        # population = np.abs(density[:, 0, 0, 0] - 1/3)
        # log_population = np.log(population)
        # fit = np.polyfit(time, log_population, 1)
        # thermalisation_time = -1/fit[0]
        # print(thermalisation_time)

        # coherence = density[:, 0, 1, 0]

        # def sine_template(t, f, a, h, p):
        #     return a*np.cos(math.tau*f*t - p) + h

        # fit_parameters, fit_cov = spo.curve_fit(
        #     sine_template, time, coherence, (2.9e9, 1, 0, 0))
        # oscillation = fit_parameters[0]
        # field = oscillation/28e9
        # print(f"Frequency: {oscillation}")
        # print(f"Field: {field}")

        # log_coherence = np.log(coherence)
        # fit = np.polyfit(time, log_coherence, 1)
        # dephasing_time = -1/fit[0]
        # print(dephasing_time)

        @logger.record(("time", "fluorescense", "density"), ("s", None, None))
        def plot():
            plt.figure("odmr")
            plt.plot(time/1e-6, 100*fluorescense/np.max(fluorescense), "k-")
            plt.xlabel("Time (us)")
            plt.ylabel("Fluorescence (%)")
            plt.draw()

            plt.figure("odmr-sweep")
            plt.plot(
                frequency_start/1e9 + frequency_width/1e9*(
                    time[time > time_start*2] - time_start)
                    / (time_end - time_start),
                100*fluorescense[time > time_start*2]/np.max(fluorescense),
                "k-"
            )
            plt.xlabel("Microwave frequency (GHz)")
            plt.ylabel("Fluorescence (%)")
            plt.draw()

            # labels = ["+", "0", "-"]
            # plt.figure("states")
            # for magnetic_index in range(3):
            #     plt.plot(
            #         time/1e-6,
            #         100*density[:, magnetic_index, magnetic_index, 0], "-",
            #         color=cm.hawaii(magnetic_index/3),
            #         label=f"g{labels[magnetic_index]}"
            #     )

            #     plt.plot(
            #         time/1e-6,
            #         100*density[:, 3 + magnetic_index, 3 + magnetic_index, 0],
            #         "--", color=cm.hawaii(magnetic_index/3),
            #         label=f"e{labels[magnetic_index]}"
            #     )
            # plt.plot(
            #     time/1e-6, 100*density[:, 6, 6, 0], "--",
            #     color=cm.hawaii(0.99), label="s")
            # plt.legend()
            # plt.xlabel("Time (us)")
            # plt.ylabel("Population (%)")
            # plt.draw()

            # plt.figure("coherences")
            # # plt.plot(time/1e-6, population, "k-", label="Population")
            # plt.plot(time/1e-6, coherence, "k-", label="Coherence")
            # plt.plot(
            #     time/1e-6, sine_template(time, *fit_parameters), "k--",
            #     label="Fit")
            # plt.xlabel("Time (us)")
            # plt.ylabel("Value")
            # plt.legend()
            # plt.draw()

            return time, fluorescense, density

        if hyperfine:
            logger.set_context("21-level")
        else:
            logger.set_context("7-level")
        plot()

    plt.show()


if __name__ == "__main__":
    main()

    # from superspinsim.params import write_values
    # write_values()

    # import benchmark
    # benchmark.basic.main()
    # benchmark.test_lindbladians.main()
    # benchmark.quartics.main()
    # benchmark.development_tests.scipy_exponentiate()
