if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from pogger import Pogger

    # from tests.odmr import main
    # from tests.contrast import main
    # from superspinsim.generate_generators import main

    # from comparisons.comp_simos import main

    # from comparisons.comp_qutip import main
    # with Pogger("superspinsim-comparisons") as logger:
    #     logger.set_context("qutip")
    #     main = logger.record(
    #         ["time_step", "density", "wall_duration", "error"],
    #         ["s", None, "s", None])(main)
    #     main()

    from comparisons.comp_superspinsim import main
    with Pogger("superspinsim-comparisons") as logger:
        logger.set_context("superspinsim")
        main = logger.record(
            ["fine_division", "density", "wall_duration", "error"],
            [None, None, "s", None])(main)
        main()

    # from comparisons.compilation import main
    # with Pogger("superspinsim-comparisons") as logger:
    #     logger.set_context("compilation")
    #     main = logger.record(["protocols", "pca_data"])(main)
    #     main()

    # from superspinsim.params import write_values
    # write_values()

    # import benchmark
    # benchmark.basic.main()
    # benchmark.test_lindbladians.main()
    # benchmark.quartics.main()
    # benchmark.development_tests.scipy_exponentiate()

    # plt.show()
