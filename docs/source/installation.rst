#########################
Installation and building
#########################

Installation
============

To install from PyPI from within a python environment,

.. code:: bash

   pip install superspinsim


One must also install the Nvidia cuda toolkit.

*Eg.* with conda,

.. code:: bash

   conda install cudatoolkit


Or on an Arch linux derivative,

.. code:: bash

   sudo pacman -Sy cuda


Building
========

These are instructions for building the package from source.

Module
~~~~~~

The package is built using the poetry tool.
To build the package, first clone it.
Then, from the package directory,

.. code:: bash

   pip install -r requirements-build.txt
   poetry build


Documentation
~~~~~~~~~~~~~

This documentation is built using sphinx.
To build the documentation, first install the build requirements,

.. code:: bash

   pip install -r requirements-build.txt


Then from the `docs/` directory, use,

.. code:: bash

   make html
