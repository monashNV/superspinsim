# SuperSpinsim

A general, GPU-based simulator, optimised for time-dependent spin systems.
We were motivated by the ability to quickly simulate highly time-dependent
quantum sensing protocols for the seven-level spin and optical model of
the nitrogen-vacancy (NV) colour centre in diamond.
However, the dynamics of general atomic and defect systems are able to be
simulated as well.

A simulated NV contrast experiment (fully-rendered microwave pulse):

![Contrast image](docs/source/_static/2025-07-25T14-40-26_superspinsim-comparisons_superspinsim_trials_23_fluorescence.png)

A simulated NV optically-detected magnetic resonance (ODMR) sweep
(fully-rendered microwave frequency sweep, and fully simulated optical
dynamics):

![ODMR image](docs/source/_static/2025-08-08T16-28-46_superspinsim-comparisons_superspinsim_trials_10_fluorescence.png)

## Version log

### 0.2.0

Implementing equivalence classes.

### 0.1.1

Fix bug in undefined `zfs_generator` in `generate_generators`.

### 0.1.0

First usable; private.
