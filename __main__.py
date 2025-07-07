if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # from tests.odmr import main
    # from tests.contrast import main
    # from superspinsim.generate_generators import main

    # from comparisons.comp_simos import main
    from comparisons.comp_qutip import main

    from pogger import Pogger
    with Pogger("superspinsim-comparisons") as logger:
        main = logger.record([])(main)
        main()
    plt.show()

    # from superspinsim.params import write_values
    # write_values()

    # import benchmark
    # benchmark.basic.main()
    # benchmark.test_lindbladians.main()
    # benchmark.quartics.main()
    # benchmark.development_tests.scipy_exponentiate()
