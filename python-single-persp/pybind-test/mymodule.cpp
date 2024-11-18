// mymodule.cpp
#include <pybind11/pybind11.h>

int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(mymodule, m) {
    m.def("add", &add, "A function that adds two numbers");
}
