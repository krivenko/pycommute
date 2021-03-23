# pycommute

[![Build status](https://github.com/krivenko/pycommute/actions/workflows/pycommute.yml/badge.svg)](
https://github.com/krivenko/pycommute/actions/workflows/pycommute.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-red)](
https://krivenko.github.io/pycommute)

*pycommute* is a Python module providing bindings for the
[libcommute](https://github.com/krivenko/libcommute) quantum operator algerba
library.

You can find a few
[usage examples](https://krivenko.github.io/pycommute/examples.html) and an
[API reference](https://krivenko.github.io/pycommute/reference.html) at
*pycommute*'s [documentation website](https://krivenko.github.io/pycommute/).

Installation from sources
-------------------------

* Install prerequisites:

  - [libcommute](https://krivenko.github.io/libcommute/installation.html)
  - [pybind11 >= 2.6.0](https://pypi.org/project/pybind11/)
  - [NumPy](https://pypi.org/project/numpy/)
  - [Sphinx >= 2.0.0](https://pypi.org/project/sphinx/)
  - [sphinx-rtd-theme](https://pypi.org/project/sphinx-rtd-theme/)

* Run the following command in the unpacked source archive of *pycommute*,

```
LIBCOMMUTE_INCLUDEDIR="<path_to_libcommute>/include" python setup.py install
```

License
-------

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
