########
Examples
########

.. note::
   
    This is thrown together quickly to help people have some idea of how to
    use the work-in-progress release for the QSSC poster.
    More polish to come in the future.


.. note::

    One needs the superspinsim, numpy, and matplotlib packages to use these
    examples.


Qubit
-----

Basic coupling
~~~~~~~~~~~~~~

Our first example simulates a qubit under coupling between its two states.
The coupling is given in actual physical units: it has the electron g factor of
two, and a magnetic field in the x direction of 1 mT.

.. code:: python

    import numpy as np
    from matplotlib import pyplot as plt

    from superspinsim import simspins


    # Define magnetic field (constant 1 mT along x)
    def mag_x(time):
        return 1e-3

    def mag_y(time):
        return 0.0

    def mag_z(time):
        return 0.0

    # Relevent only for defects
    def excitation(time):
        return 0.0

    # Define qubit (spin-half, electron g factor)
    spins = [[{"S": 1/2, "g": 2.0}]]

    # Define density matrix
    density_initial = np.zeros((2, 2), dtype=np.float64)
    density_initial[0, 0] = 1

    # Simulate
    time, density = simspins(
        [mag_x, mag_y, mag_z, excitation],  # Fields
        0, 100e-6, 1e-9,                    # Time start/stop/step
        spins, [{}], {},                    # Spin description
        density_initial                     # Initial state
    )

    # Plot
    plt.figure(label="Lab frame")
    plt.plot(time/1e-9, density[:, 0, 0].real/1e-2, "k-", label="up")
    plt.plot(time/1e-9, density[:, 1, 1].real/1e-2, "k--", label="down")
    plt.xlim(0, 200)
    plt.xlabel("Time (ns)")
    plt.ylabel("Population (%)")
    plt.legend()


Though this is a basic illustrative example, it is not the recommended way of
simulating a constant bias field in SuperSpinsim.
If we know that the dynamics are dominated by (or entirely) a DC bias term,
then it is more efficient to diagonalise the problem and work within a
"generalised rotating frame".
To use this, define a bias magnetic field in the definition of the spin system.

.. note::

    This is not the same as using the rotating wave approximation.
    The simulator makes no approximations here and actually improves precision.


.. code:: python

    import numpy as np
    from matplotlib import pyplot as plt

    from superspinsim import simspins

    # No "interaction" magnetic field
    def mag_x(time):
        return 0.0

    def mag_y(time):
        return 0.0

    def mag_z(time):
        return 0.0

    # Relevent only for defects
    def excitation(time):
        return 0.0

    # Define qubit 
    spins = [[{
        "S": 1/2, "g": 2.0,           # (spin-half, electron g factor)
        "B0": np.array([1e-3, 0, 0])  # Bias magnetic field of 1 mT along x
    }]]

    # Define density matrix
    density_initial = np.zeros((2, 2), dtype=np.float64)
    density_initial[0, 0] = 1

    # Simulate
    time, density = simspins(
        [mag_x, mag_y, mag_z, excitation],  # Fields
        0, 100e-6, 1e-9,                    # Time start/stop/step
        spins, [{}], {},                    # Spin description
        density_initial                     # Initial state
    )

    # Plot
    plt.figure(label="Rotating frame")
    plt.plot(time/1e-9, density[:, 0, 0].real/1e-2, "k-", label="up")
    plt.plot(time/1e-9, density[:, 1, 1].real/1e-2, "k--", label="down")
    plt.xlim(0, 200)
    plt.xlabel("Time (ns)")
    plt.ylabel("Population (%)")
    plt.legend()
