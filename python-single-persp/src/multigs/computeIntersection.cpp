#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>

namespace py = pybind11;

double intersect(double* x, double* z, int pix, double param) {
    // std::cout<<"intersect mei\n";
    // std::cout<<
    // Intersection Kernel - C++ version
    std::vector<int> symtab(pix+1, 0);
    double accum = 0;

    // Fill the table
    for (int n = 0; n < static_cast<int>(param); n++) {
        symtab[static_cast<int>(x[n])] = 1;
    }

    // Intersect and accumulate
    for (int n = 0; n < static_cast<int>(param); n++) {
        if (symtab[static_cast<int>(z[n])] == 1) {
            accum++;
        }
    }

    // Normalized intersection
    return accum / param;
}

py::array_t<double> compute_intersection(py::array_t<double> A1, py::array_t<double> A2, double H) {
    
    auto buf1 = A1.request();
    auto buf2 = A2.request();

    std::cout << "Shape of A1: (" << buf1.shape[0] << ", " << buf1.shape[1] << ")" << std::endl;
    std::cout << "Shape of A2: (" << buf2.shape[0] << ", " << buf2.shape[1] << ")" << std::endl;

    double* ptrA1 = static_cast<double*>(buf1.ptr);
    double* ptrA2 = static_cast<double*>(buf2.ptr);
    std::cout<< "Checkpoint\n";

    int m1 = buf1.shape[0];
    int n1 = buf1.shape[1];
    int m2 = buf2.shape[0];
    int n2 = buf2.shape[1];

    // Correct the shape of K: (n1, n2)
    py::array_t<double> K = py::array_t<double>({n1, n2});
    py::buffer_info bufK = K.request();
    double* ptrK = static_cast<double*>(bufK.ptr);
    // for (int j = 0; j < n1; j++) {
    //     for (int i = 0; i < n2; i++) {
    //         // Fix indexing: ptrA1 + j * m1 and ptrA2 + i * m2 should be passed correctly
    //         int temp = intersect(ptrA1 + j * m1, ptrA2 + i * m2, m1, H);
    //         std::cout<<"temp done\n";
    //         ptrK[j + i * n1] = temp;
    //     }
    // }
    for (int j = 0; j < n1; j++) {
        for (int i = 0; i < n2; i++) {
            ptrK[i + j * n2] = intersect(ptrA1 + j * m1, ptrA2 + i * m2, m1, H);
        }
    }


    std::cout << "Shape of K: (" << n1 << ", " << n2 << ")" << std::endl;
    return K;
}

PYBIND11_MODULE(computeIntersection, m) {
    m.def("compute_intersection", &compute_intersection, "Compute the intersection between matrices",
          py::arg("A1"), py::arg("A2"), py::arg("H"));
}
