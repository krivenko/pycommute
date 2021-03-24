.. _installation:

Prerequisites
=============

- `libcommute <https://github.com/krivenko/libcommute>`_ (either installed or
  as a directory with unpacked source code)
- pybind11 >= 2.6.0
- numpy >= 1.12.0
- Sphinx >= 2.0.0
- sphinx-rtd-theme >= 0.5.0

Installation from sources
=========================

Installing *pycommute* from sources is as easy as running

.. code::

  LIBCOMMUTE_INCLUDEDIR="<path_to_libcommute>/include" python setup.py install

in the unpacked source directory. ``<path_to_libcommute>`` must be either
installation or source directory of *libcommute*.

If you need to build the documentation locally, you should additionally run

.. code::

  python setup.py build_sphinx

Automated installation using ``pip``
====================================

.. code::

  LIBCOMMUTE_INCLUDEDIR="<path_to_libcommute>/include" pip install pycommute
