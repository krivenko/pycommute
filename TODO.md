TODO list
=========

* Remove the workaround for pybind11 issue #2516 when it is fixed.
* Try to allow inheriting from `Generator` in Python if pybind11 issue #1546 and
  related issues are ever resolved.
  See also https://github.com/pybind/pybind11/pull/1566.
* Models to add:
  - ``jaynes_cummings()`` (multi-mode multi-atom Jaynes–Cummings, https://iopscience.iop.org/article/10.1088/0953-4075/46/22/224008)
  - ``hubbard_kanamori_int()``
  - ``slater_int()``
