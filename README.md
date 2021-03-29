# pycommute

[![Package on PyPI](https://img.shields.io/pypi/v/pycommute.svg)](
https://pypi.org/project/pycommute)
[![Build and test](https://github.com/krivenko/pycommute/actions/workflows/build-and-test.yml/badge.svg)](
https://github.com/krivenko/pycommute/actions/workflows/build-and-test.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-red)](
https://krivenko.github.io/pycommute)

*pycommute* is a Python package providing bindings for the
[libcommute](https://github.com/krivenko/libcommute) quantum operator algerba
library.

You can find a few
[usage examples](https://krivenko.github.io/pycommute/examples.html) and an
[API reference](https://krivenko.github.io/pycommute/reference.html) at
*pycommute*'s [documentation website](https://krivenko.github.io/pycommute/).

Installation from sources
-------------------------

* [Download](https://github.com/krivenko/libcommute/releases) source code of
  *libcommute* 0.6 or newer and optionally
  [install](https://krivenko.github.io/libcommute/installation.html) it.

* Install Python prerequisites:

  - [packaging >= 17.0](https://pypi.org/project/packaging/)
  - [pybind11 >= 2.6.0](https://pypi.org/project/pybind11/)
  - [numpy >= 1.12.0](https://pypi.org/project/numpy/)

* Run the following command in the unpacked source archive of *pycommute*,

```
LIBCOMMUTE_INCLUDEDIR="<path_to_libcommute>/include" python setup.py install
```

``<path_to_libcommute>`` must be either installation or source directory of
*libcommute*.

Automated installation using ``pip``
------------------------------------

```
LIBCOMMUTE_INCLUDEDIR="<path_to_libcommute>/include" pip install pycommute
```

License
-------

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
