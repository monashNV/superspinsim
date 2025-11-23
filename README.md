<img src="https://raw.github.com/monashNV/superspinsim/docs/source/_static/logo.png" height="200">

# SuperSpinsim

[Documentation](https://superspinsim.readthedocs.io/en/latest/) -
[GitHub](https://github.com/monashNV/superspinsim) -
[PyPI](https://pypi.org/project/superspinsim/) -
[Poster](https://raw.github.com/monashNV/superspinsim/docs/source/_static/poster.pdf)

A general, GPU-based simulator, optimised for time-dependent spin systems.
We were motivated by the ability to quickly simulate highly time-dependent
quantum sensing protocols for the seven-level spin and optical model of
the nitrogen-vacancy (NV) colour centre in diamond.
However, the dynamics of general atomic and defect systems are able to be
simulated as well.

A simulated NV contrast experiment (fully-rendered microwave pulse):

<img src="https://raw.github.com/monashNV/superspinsim/docs/source/_static/2025-07-25T14-40-26_superspinsim-comparisons_superspinsim_trials_23_fluorescence.png" height="512">

A simulated NV optically-detected magnetic resonance (ODMR) sweep
(fully-rendered microwave frequency sweep, and fully simulated optical
dynamics):

<img src="https://raw.github.com/monashNV/superspinsim/docs/source/_static/2025-08-08T16-28-46_superspinsim-comparisons_superspinsim_trials_10_fluorescence.png" height="512">

## Project status

SuperSpinsim is a **work-in-progress**.

The simulator is currently functional.
There may still be some edge cases that cause errors; please let us know here
if any are found.

There are a few features related to efficient representation of
states/processes that we want to add which are not fully-implemented yet:

- Compressed, equivalence class representations of systems which have subspaces
  of states unaffected by the equations of motion.
  - Experimental feature; implemented but not fully tested.
- Efficient calculation of purely unitary dynamics.

We will submit a manuscript for review when the above are implemented.

There are some further features which are thinking about adding, possibly after
the manuscript is submitted:

- Support to split the problem over multiple GPUs, rather than just one.
- Support for running the algorithm on (SIMD operations on) a CPU.
  

## Installation

SuperSpinsim is built on top of
[number.cuda](https://nvidia.github.io/numba-cuda/), which means it has similar
system requirements:

- An Nvidia GPU capable of cuda.
  - Meaning AMD, Apple, and Intel GPUs are not currently supported.
- Official Nvidia graphics drivers.
- The Nvidia [cuda toolkit](https://developer.nvidia.com/cuda-downloads)
  - An installer can be downloaded directly, or it can be installed from a
    package manager like conda or pacman.
- Python versions 3.11 to 3.13 only.
  - Meaning python version 3.14 is not currently supported.

If all of these are satisfied, then SuperSpinsim can be installed from the
Python Package Index (PyPI),

```
uv add superspinsim       # If using the uv package manager
pip install superspinsim  # If using the pip package manager
```

### From source

SuperSpinsim can be built from source using the
[uv](https://docs.astral.sh/uv/) package manager.

```
git clone https://github.com/monashNV/superspinsim.git
cd superspinsim
uv build
```

SuperSpinsim can also be installed into another project without being built
using the [uv](https://docs.astral.sh/uv/) package manager.

```
mkdir other-project
git clone https://github.com/monashNV/superspinsim.git
cd other-project
uv init
uv add ../superspinsim
```

## Version log

### 0.3.1

README links fixed.

### 0.3.0

Public release.

### 0.2.0

Implementing equivalence classes.

### 0.1.1

Fixed bug in undefined `zfs_generator` in `generate_generators`.

### 0.1.0

First usable; private.
