// #include <pybind11/pybind11.h>
// #include <vector>
// #include <string>
// #include <lsd.h>  // Assuming you have the LSD header file

// namespace py = pybind11;

// std::vector<std::vector<double>> lsd_cpp(const cv::Mat& src) {
//     cv::Mat tmp, src_gray;
//     cv::cvtColor(src, tmp, CV_RGB2GRAY);
//     tmp.convertTo(src_gray, CV_64FC1);

//     image_double image = new_image_double(src_gray.cols, src_gray.rows);
//     image->data = src_gray.ptr<double>(0);

//     ntuple_list ntl = lsd(image);

//     std::vector<std::vector<double>> lines;
//     for (int j = 0; j != ntl->size; ++j) {
//         std::vector<double> line = {
//             ntl->values[0 + j * ntl->dim] + 1,
//             ntl->values[2 + j * ntl->dim] + 1,
//             ntl->values[1 + j * ntl->dim] + 1,
//             ntl->values[3 + j * ntl->dim] + 1,
//             ntl->values[4 + j * ntl->dim]  // width
//         };
//         lines.push_back(line);
//     }
//     return lines;
// }


// PYBIND11_MODULE(lsd_module, m) {
//     m.def("lsd_cpp", &lsd_cpp, "Line Segment Detector in C++");
// }
