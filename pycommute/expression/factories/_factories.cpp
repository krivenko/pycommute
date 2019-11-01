#include <pybind11/pybind11.h>

#include <libcommute/expression/factories_dyn.hpp>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(_factories, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}
