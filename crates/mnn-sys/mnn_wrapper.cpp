#include "mnn_wrapper.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNForwardType.h>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>

extern "C" {

// Interpreter functions
MNN_Interpreter* MNN_Interpreter_createFromFile(const char* file) {
    return reinterpret_cast<MNN_Interpreter*>(MNN::Interpreter::createFromFile(file));
}

MNN_Interpreter* MNN_Interpreter_createFromBuffer(const void* buffer, int size) {
    return reinterpret_cast<MNN_Interpreter*>(
        MNN::Interpreter::createFromBuffer(buffer, static_cast<size_t>(size)));
}

void MNN_Interpreter_destroy(MNN_Interpreter* net) {
    delete reinterpret_cast<MNN::Interpreter*>(net);
}

MNN_Session* MNN_Interpreter_createSession(MNN_Interpreter* net, const MNN_ScheduleConfig* config_raw) {
    auto* interpreter = reinterpret_cast<MNN::Interpreter*>(net);
    
    MNN::ScheduleConfig config;
    config.type = static_cast<MNNForwardType>(config_raw->type_);
    config.numThread = config_raw->numThread;
    
    // Handle backend config if present
    if (config_raw->backendConfig != nullptr) {
        MNN::BackendConfig backendConfig;
        backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(
            config_raw->backendConfig->precision);
        backendConfig.power = static_cast<MNN::BackendConfig::PowerMode>(
            config_raw->backendConfig->power);
        backendConfig.memory = static_cast<MNN::BackendConfig::MemoryMode>(
            config_raw->backendConfig->memory);
        config.backendConfig = &backendConfig;
        
        return reinterpret_cast<MNN_Session*>(interpreter->createSession(config));
    }
    
    return reinterpret_cast<MNN_Session*>(interpreter->createSession(config));
}

void MNN_Interpreter_releaseSession(MNN_Interpreter* net, MNN_Session* session) {
    auto* interpreter = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s = reinterpret_cast<MNN::Session*>(session);
    interpreter->releaseSession(s);
}

void MNN_Interpreter_resizeSession(MNN_Interpreter* net, MNN_Session* session) {
    auto* interpreter = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s = reinterpret_cast<MNN::Session*>(session);
    interpreter->resizeSession(s);
}

int MNN_Interpreter_runSession(MNN_Interpreter* net, MNN_Session* session) {
    auto* interpreter = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s = reinterpret_cast<MNN::Session*>(session);
    return static_cast<int>(interpreter->runSession(s));
}

MNN_Tensor* MNN_Interpreter_getSessionInput(MNN_Interpreter* net, MNN_Session* session, const char* name) {
    auto* interpreter = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s = reinterpret_cast<MNN::Session*>(session);
    return reinterpret_cast<MNN_Tensor*>(interpreter->getSessionInput(s, name));
}

MNN_Tensor* MNN_Interpreter_getSessionOutput(MNN_Interpreter* net, MNN_Session* session, const char* name) {
    auto* interpreter = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s = reinterpret_cast<MNN::Session*>(session);
    return reinterpret_cast<MNN_Tensor*>(interpreter->getSessionOutput(s, name));
}

MNN_TensorInfo* MNN_Interpreter_getSessionInputAll(MNN_Interpreter* net, MNN_Session* session, int* size) {
    auto* interpreter = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s = reinterpret_cast<MNN::Session*>(session);
    
    auto inputs = interpreter->getSessionInputAll(s);
    *size = static_cast<int>(inputs.size());
    
    if (inputs.empty()) {
        return nullptr;
    }
    
    // Allocate array - caller must manage this memory
    static std::vector<MNN_TensorInfo> info_storage;
    static std::vector<std::string> name_storage;
    
    info_storage.clear();
    name_storage.clear();
    
    for (const auto& pair : inputs) {
        name_storage.push_back(pair.first);
        MNN_TensorInfo info;
        info.name = name_storage.back().c_str();
        info.tensor = reinterpret_cast<MNN_Tensor*>(pair.second);
        info_storage.push_back(info);
    }
    
    return info_storage.data();
}

MNN_TensorInfo* MNN_Interpreter_getSessionOutputAll(MNN_Interpreter* net, MNN_Session* session, int* size) {
    auto* interpreter = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s = reinterpret_cast<MNN::Session*>(session);
    
    auto outputs = interpreter->getSessionOutputAll(s);
    *size = static_cast<int>(outputs.size());
    
    if (outputs.empty()) {
        return nullptr;
    }
    
    // Allocate array - using static storage for simplicity
    static std::vector<MNN_TensorInfo> info_storage;
    static std::vector<std::string> name_storage;
    
    info_storage.clear();
    name_storage.clear();
    
    for (const auto& pair : outputs) {
        name_storage.push_back(pair.first);
        MNN_TensorInfo info;
        info.name = name_storage.back().c_str();
        info.tensor = reinterpret_cast<MNN_Tensor*>(pair.second);
        info_storage.push_back(info);
    }
    
    return info_storage.data();
}

void MNN_Interpreter_resizeTensor(MNN_Interpreter* net, MNN_Tensor* tensor, const int* dims, int size) {
    auto* interpreter = reinterpret_cast<MNN::Interpreter*>(net);
    auto* t = reinterpret_cast<MNN::Tensor*>(tensor);
    
    std::vector<int> shape(dims, dims + size);
    interpreter->resizeTensor(t, shape);
}

// Tensor functions
MNN_Tensor* MNN_Tensor_create(int dims, const int* shape, int /*type*/) {
    std::vector<int> shape_vec(shape, shape + dims);
    halide_type_t halide_type;
    halide_type.code = halide_type_float;
    halide_type.bits = 32;
    halide_type.lanes = 1;
    
    // Use CAFFE (NCHW) layout by default as it's the standard for data exchange
    return reinterpret_cast<MNN_Tensor*>(
        MNN::Tensor::create(shape_vec, halide_type, nullptr, MNN::Tensor::CAFFE));
}

MNN_Tensor* MNN_Tensor_createHostTensorFromDevice(const MNN_Tensor* device, int copy) {
    auto* t = reinterpret_cast<const MNN::Tensor*>(device);
    return reinterpret_cast<MNN_Tensor*>(
        MNN::Tensor::createHostTensorFromDevice(t, copy != 0));
}

void MNN_Tensor_destroy(MNN_Tensor* tensor) {
    delete reinterpret_cast<MNN::Tensor*>(tensor);
}

int MNN_Tensor_copyFromHostTensor(MNN_Tensor* dst, const MNN_Tensor* src) {
    auto* dst_tensor = reinterpret_cast<MNN::Tensor*>(dst);
    auto* src_tensor = reinterpret_cast<const MNN::Tensor*>(src);
    return dst_tensor->copyFromHostTensor(src_tensor) ? 1 : 0;
}

int MNN_Tensor_copyToHostTensor(const MNN_Tensor* src, MNN_Tensor* dst) {
    auto* src_tensor = reinterpret_cast<const MNN::Tensor*>(src);
    auto* dst_tensor = reinterpret_cast<MNN::Tensor*>(dst);
    return src_tensor->copyToHostTensor(dst_tensor) ? 1 : 0;
}

void* MNN_Tensor_getHost(const MNN_Tensor* tensor) {
    auto* t = reinterpret_cast<const MNN::Tensor*>(tensor);
    return t->host<void>();
}

const int* MNN_Tensor_getDimensions(const MNN_Tensor* tensor, int* size) {
    auto* t = reinterpret_cast<const MNN::Tensor*>(tensor);
    static std::vector<int> shape_storage;
    shape_storage = t->shape();
    *size = static_cast<int>(shape_storage.size());
    return shape_storage.data();
}

int MNN_Tensor_wait(MNN_Tensor* tensor, MNN_Tensor_MapType type, int finish) {
    auto* t = reinterpret_cast<MNN::Tensor*>(tensor);
    return t->wait(static_cast<MNN::Tensor::MapType>(type), finish != 0);
}

} // extern "C"
