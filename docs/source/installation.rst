#########################
Installation and building
#########################

System requirements
===================

SuperSpinsim is build on top of
`numba.cuda <https://nvidia.github.io/numba-cuda/>`__, which is a python
package used to compile python code into LLVM code for Nvidia cuda GPUs.
Follow the link to see the system requirements for numba.cuda.

In particular, one must have an Nvidia GPU installed in the device being used,
as well as the 
`Nvidia cuda toolkit <https://developer.nvidia.com/cuda-downloads>`__ and
official Nvidia graphics drivers.

Right now the package requirements for SuperSpinsim mean that SuperSpinsim can
be used only with python versions 3.11 to 3.13.
Python version 3.14 is not supported yet, due to the requirements of the numba
package.

We have tested SuperSpinsim on various versions of GNU/Linux
(Arch, Manjaro and Ubuntu), as well as on Windows 11.
It will not work on Mac OS, as modern Mac computers are not compatible with
Nvidia GPUs.

Installation
============

First, make sure you have satisfied the the system requirements (above).
Then, to install from PyPI from within a python environment,

.. code:: bash

   uv add superspinsim       # If using uv as a python package manager
   pip install superspinsim  # If using pip as a python package manager


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

Alternatively, to install SuperSpinsim from source to a project that uses uv
as a package manager, use

.. code:: bash

   uv add <SuperSpinsim directory>


to use the package without building it.


Documentation
~~~~~~~~~~~~~

This documentation is built using the sphinx tool.
The easiest way to build the documentation requires the tools
`make <https://www.gnu.org/software/make/>`__ and
`uv <https://docs.astral.sh/uv/getting-started/installation/>`__.
From the SuperSpinsim source directory,

.. code:: bash

   cd docs
   uv run make html
