if __name__ == "__main__":

    # from comparisons.comp_simos import main

    # from comparisons.comp_qutip import main
    # main(
    #     lindbladian="odmr",
    #     use_jax=False
    # )

    # from comparisons.comp_dynamiqs import main
    # main()

    # from comparisons.comp_quantum_toolbox import main
    # main(
    #     lindbladian="odmr"
    # )

    from comparisons.comp_superspinsim import main
    for number in [5, 2, 1]:
        print("Number of exponentials:", number)
        main(
            lindbladian="odmr",
            use_rotating=False,
            number_of_exponentials=number
        )

    # from comparisons.compilation import main
    # main()

    # from superspinsim.params import write_values
    # write_values()
