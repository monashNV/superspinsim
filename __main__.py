if __name__ == "__main__":
    # from matplotlib import pyplot as plt
    # from pogger import Pogger

    # from tests.odmr import main
    # from tests.contrast import main
    from tests.odmr_4 import main
    # from superspinsim.generate_generators import main
    main()

    # from comparisons.comp_simos import main

    # from comparisons.comp_qutip import main
    # main(use_jax=False)

    # from comparisons.comp_dynamiqs import main
    # main()

    # from comparisons.comp_quantum_toolbox import main
    # main()

    # from comparisons.comp_superspinsim import main
    # main(
    #     use_rotating=True,
    #     number_of_exponentials=5
    # )

    # from comparisons.compilation import main
    # with Pogger("superspinsim-comparisons") as logger:
    #     logger.set_context("compilation")
    #     main = logger.record(["protocols", "pca_data"])(main)
    #     main()
    # plt.show()

    # from superspinsim.params import write_values
    # write_values()

    # import benchmark
    # benchmark.basic.main()
    # benchmark.test_lindbladians.main()
    # benchmark.quartics.main()
    # benchmark.development_tests.scipy_exponentiate()
