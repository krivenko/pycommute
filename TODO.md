TODO list
=========

* Remove the lambda function `extract_spin_arg()` from `expression.cpp` when
  https://github.com/pybind/pybind11/issues/5840 is fixed.
* Remove the workaround for pybind11 issue #2516 when it is fixed.
* Try to allow inheriting from `Generator` in Python if pybind11 issue #1546 and
  related issues are ever resolved.
  See also https://github.com/pybind/pybind11/pull/1566.
