#########################
Installation and building
#########################

Installation
============

To install from PyPI from within a python environment,

.. code:: bash

   uv add superspinsim       # If using uv as a python package manager
   pip install superspinsim  # If using pip as a python package manager


One must also install the
`Nvidia cuda toolkit <https://developer.nvidia.com/cuda-downloads>`__.
One can install it from the linked website, or with a package manager
(eg, conda or pacman).


Building and installing from source
===================================

These are instructions for building the package from source.

Module
~~~~~~

The package is built using the
`uv <https://docs.astral.sh/uv/getting-started/installation/>`__ tool for python
package management.
See the link for installation instructions.

To build the package with uv, first clone it,

.. code:: bash

   git clone https://github.com/monashNV/superspinsim.git


Then, from the package directory,

.. code:: bash

   uv build
   

This will build python wheels under the `builds` directory.
These wheels can be installed using using `pip install` or `uv add`.

Alternatively, if one is using uv for package management of other projects,
one can simply use,

.. code:: bash

   uv add <SuperSpinsim directory>


to use the package without building it.


Documentation
~~~~~~~~~~~~~

This documentation is built using the sphinx tool.
The easiest way to build the documentation requires the tools cmake and uv.
From the superspinsim source directory,

.. code:: bash

   cd docs
   uv run make html
