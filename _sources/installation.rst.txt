.. _installation:

Installation
============

Prerequisites
-------------

- `libcommute <https://github.com/krivenko/libcommute>`_ 0.6 or newer (either
  installed or as a directory with unpacked source code)
- packaging >= 17.0
- pybind11 >= 2.6.0
- numpy >= 1.12.0
- Sphinx >= 2.0.0 (optional, to build documentation)
- sphinx-rtd-theme (optional, to build documentation)

Installation from sources
-------------------------

Installing *pycommute* from sources is as easy as running

.. code::

  LIBCOMMUTE_INCLUDEDIR="<path_to_libcommute>/include" python setup.py install

in the unpacked source directory. ``<path_to_libcommute>`` must be either
installation or source directory of *libcommute*.

If you need to build documentation locally, you should additionally run

.. code::

  python setup.py build_sphinx

Automated installation using ``pip``
------------------------------------

.. code::

  LIBCOMMUTE_INCLUDEDIR="<path_to_libcommute>/include" pip install pycommute

Docker images
-------------

Docker images of tagged releases of *pycommute* are available from
`Docker Hub <https://hub.docker.com/repository/docker/ikrivenko/pycommute>`_.

.. code::

  docker run --rm -p 8888:8888 ikrivenko/pycommute:latest

This command will pull the most recent image and launch a
`Jupyter <https://jupyter.org/>`_ notebook server accessible at
`http://127.0.0.1:8888/ <http://127.0.0.1:8888/>`_. The server is run in a
directory with a few interactive example notebooks.
