#ifndef QCSNPE_H
#define QCSNPE_H

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <map>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <DlContainer/IDlContainer.hpp>
#include "DlSystem/RuntimeList.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
//#include "DlSystem/UDLFunc.hpp"
#include <DlSystem/ITensorFactory.hpp>
#include <SNPE/SNPEFactory.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <ctime>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

typedef std::chrono::milliseconds ms; 

class Qcsnpe {
    private:
        std::unique_ptr<zdl::SNPE::SNPE> model_handler;
        std::unique_ptr<zdl::DlContainer::IDlContainer> container;
        zdl::DlSystem::RuntimeList runtime_list;
        zdl::DlSystem::StringList outputs;
        zdl::DlSystem::TensorMap output_tensor_map;
        zdl::DlSystem::StringList out_tensors;

    public:
        Qcsnpe(std::string &dlc, std::vector<std::string> &output_layers, int system_type);
        std::map<std::string, std::vector<float>> predict(py::array_t<uint8_t> img);
        std::vector<float> throughput_vec;
        std::vector<float> fps_vec;

};

PYBIND11_MODULE(qcsnpe, m) {
    py::class_<Qcsnpe>(m, "qcsnpe")
        .def(py::init<std::string &, std::vector<std::string> &, int>())
        .def("predict", &Qcsnpe::predict);
}

#endif
