def main():
    import superspinsim as s3
    from superspinsim.generate_generators import generate_7

    import math
    import numpy as np
    from matplotlib import pyplot as plt

    from pogger import Pogger as Logger

    quiescent_magnetic_field = np.array([0, 0, 0.1])

    def coefficient_x(time):
        if time > 5e-6:
            return 0.01*math.sin(math.tau*2.8e9*time)

    def coefficient_y(time):
        return 0

    def coefficient_z(time):
        return 0

    def coefficient_r(time):
        return 0.5

    coefficients = [coefficient_x, coefficient_y, coefficient_z, coefficient_r]
    lindbladian, generators, vectorisation_map = generate_7(
        coefficients, quiescent_magnetic_field)

    simulator = s3.generate_simulator(
        lindbladian, np.array(generators), vectorisation_map,
        number_of_fine_divisions=10
    )

    density_operator_initial = np.zeros((7, 7, 2))
    density_operator_initial[6, 6, 0] = 1

    time, density = simulator(density_operator_initial, 0, 10e-6, 100e-12)
    fluorescense = \
        density[:, 3, 3, 0] + density[:, 4, 4, 0] + density[:, 5, 5, 0]

    with Logger("superspinsim-generate") as logger:
        @logger.record(("time", "fluorescense", "density"), ("s", None, None))
        def plot():
            plt.figure(label="7-level-odmr")
            plt.plot(time/1e-6, 100*fluorescense/np.max(fluorescense), "k-")
            plt.xlabel("Time (us)")
            plt.ylabel("Fluorescence (%)")
            plt.draw()
            return time, fluorescense, density

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
