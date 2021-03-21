.. _installation:

Prerequisites
=============

- A `libcommute <https://github.com/krivenko/libcommute>`_ installation.
- pybind11 >= 2.6.0.
- NumPy.
- Sphinx >= 2.0.0 and sphinx-rtd-theme to build documentation.

Installation
============

Installing *pycommute* from sources is as easy as running

.. code::

  LIBCOMMUTE_INCLUDEDIR=<path_to_libcommute_include_dir> python setup.py install

in the unpacked source directory.

If you need to build the documentation locally, you should additionally run

.. code::

  python setup.py build_sphinx
