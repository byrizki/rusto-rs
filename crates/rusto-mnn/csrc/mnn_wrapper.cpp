#include "mnn_wrapper.h"

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <cstring>
#include <string>
#include <vector>

extern "C" {

MNN_Interpreter* MNN_Interpreter_createFromFile(const char* file_path) {
    if (!file_path) {
        return nullptr;
    }
    auto* interpreter = MNN::Interpreter::createFromFile(file_path);
    return reinterpret_cast<MNN_Interpreter*>(interpreter);
}

void MNN_Interpreter_destroy(MNN_Interpreter* net) {
    if (net) {
        auto* net_ptr = reinterpret_cast<MNN::Interpreter*>(net);
        delete net_ptr;
    }
}

MNN_Session* MNN_Interpreter_createSession(MNN_Interpreter* net, int num_threads, int forward_type) {
    if (!net) {
        return nullptr;
    }
    auto* net_ptr = reinterpret_cast<MNN::Interpreter*>(net);
    MNN::ScheduleConfig config;
    config.type = static_cast<MNNForwardType>(forward_type);
    config.numThread = num_threads > 0 ? num_threads : 4;
    auto* session = net_ptr->createSession(config);
    return reinterpret_cast<MNN_Session*>(session);
}

void MNN_Interpreter_releaseSession(MNN_Interpreter* net, MNN_Session* session) {
    if (net && session) {
        auto* net_ptr = reinterpret_cast<MNN::Interpreter*>(net);
        auto* s_ptr = reinterpret_cast<MNN::Session*>(session);
        net_ptr->releaseSession(s_ptr);
    }
}

int MNN_Interpreter_resizeTensor(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name,
    const int* dims,
    int dim_count
) {
    if (!net || !session || !dims || dim_count <= 0) {
        return -1;
    }
    auto* net_ptr = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s_ptr = reinterpret_cast<MNN::Session*>(session);

    auto* tensor = (name && name[0] != '\0')
        ? net_ptr->getSessionInput(s_ptr, name)
        : net_ptr->getSessionInput(s_ptr, nullptr);

    if (!tensor) {
        auto inputs = net_ptr->getSessionInputAll(s_ptr);
        if (inputs.find("x") != inputs.end()) {
            tensor = inputs["x"];
        } else if (!inputs.empty()) {
            tensor = inputs.begin()->second;
        }
    }

    if (!tensor) {
        return -2;
    }

    std::vector<int> shape(dims, dims + dim_count);
    net_ptr->resizeTensor(tensor, shape);
    return 0;
}

void MNN_Interpreter_resizeSession(MNN_Interpreter* net, MNN_Session* session) {
    if (net && session) {
        auto* net_ptr = reinterpret_cast<MNN::Interpreter*>(net);
        auto* s_ptr = reinterpret_cast<MNN::Session*>(session);
        net_ptr->resizeSession(s_ptr);
    }
}

int MNN_Interpreter_runSession(MNN_Interpreter* net, MNN_Session* session) {
    if (!net || !session) {
        return -1;
    }
    auto* net_ptr = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s_ptr = reinterpret_cast<MNN::Session*>(session);
    return static_cast<int>(net_ptr->runSession(s_ptr));
}

int MNN_Session_setHostInputFloat(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name,
    const float* data,
    int count
) {
    if (!net || !session || !data || count <= 0) {
        return -1;
    }
    auto* net_ptr = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s_ptr = reinterpret_cast<MNN::Session*>(session);

    auto* tensor = (name && name[0] != '\0')
        ? net_ptr->getSessionInput(s_ptr, name)
        : net_ptr->getSessionInput(s_ptr, nullptr);

    if (!tensor) {
        auto inputs = net_ptr->getSessionInputAll(s_ptr);
        if (inputs.find("x") != inputs.end()) {
            tensor = inputs["x"];
        } else if (!inputs.empty()) {
            tensor = inputs.begin()->second;
        }
    }

    if (!tensor) {
        return -2;
    }

    MNN::Tensor host_in(tensor, MNN::Tensor::CAFFE);
    if (host_in.elementSize() != count) {
        return -3;
    }

    std::memcpy(host_in.host<float>(), data, count * sizeof(float));
    bool ok = tensor->copyFromHostTensor(&host_in);
    return ok ? 0 : -4;
}

int MNN_Session_setHostInputInt32(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name,
    const int* data,
    int count
) {
    if (!net || !session || !data || count <= 0) {
        return -1;
    }
    auto* net_ptr = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s_ptr = reinterpret_cast<MNN::Session*>(session);

    auto* tensor = (name && name[0] != '\0')
        ? net_ptr->getSessionInput(s_ptr, name)
        : net_ptr->getSessionInput(s_ptr, nullptr);

    if (!tensor) {
        auto inputs = net_ptr->getSessionInputAll(s_ptr);
        if (inputs.find("x") != inputs.end()) {
            tensor = inputs["x"];
        } else if (!inputs.empty()) {
            tensor = inputs.begin()->second;
        }
    }

    if (!tensor) {
        return -2;
    }

    MNN::Tensor host_in(tensor, MNN::Tensor::CAFFE);
    if (host_in.elementSize() != count) {
        return -3;
    }

    std::memcpy(host_in.host<int32_t>(), data, count * sizeof(int32_t));
    bool ok = tensor->copyFromHostTensor(&host_in);
    return ok ? 0 : -4;
}

int MNN_Session_getInputChannels(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name
) {
    if (!net || !session) {
        return -1;
    }
    auto* net_ptr = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s_ptr = reinterpret_cast<MNN::Session*>(session);

    auto inputs = net_ptr->getSessionInputAll(s_ptr);

    // If specific name requested
    if (name && name[0] != '\0') {
        auto it = inputs.find(name);
        if (it != inputs.end() && it->second) {
            const auto& shape = it->second->shape();
            if (shape.size() >= 2 && shape[1] > 0) {
                return shape[1];
            }
        }
    }

    // Try common image input name 'x'
    auto x_it = inputs.find("x");
    if (x_it != inputs.end() && x_it->second) {
        const auto& shape = x_it->second->shape();
        if (shape.size() >= 2 && shape[1] > 0) {
            return shape[1];
        }
    }

    // Look for any input tensor with >= 3 dimensions (like NCHW)
    for (const auto& pair : inputs) {
        if (pair.second) {
            const auto& shape = pair.second->shape();
            if (shape.size() >= 3 && shape[1] > 0) {
                return shape[1];
            }
        }
    }

    // Fallback to default input
    auto* tensor = net_ptr->getSessionInput(s_ptr, nullptr);
    if (tensor) {
        const auto& shape = tensor->shape();
        if (shape.size() >= 2 && shape[1] > 0) {
            return shape[1];
        }
    }

    return 3;
}

int MNN_Session_getOutputShape(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name,
    int* dims_out,
    int max_dims,
    int* dim_count_out
) {
    if (!net || !session || !dims_out || !dim_count_out) {
        return -1;
    }
    auto* net_ptr = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s_ptr = reinterpret_cast<MNN::Session*>(session);

    auto* tensor = (name && name[0] != '\0')
        ? net_ptr->getSessionOutput(s_ptr, name)
        : net_ptr->getSessionOutput(s_ptr, nullptr);

    if (!tensor) {
        return -2;
    }

    const auto& shape = tensor->shape();
    *dim_count_out = static_cast<int>(shape.size());
    for (size_t i = 0; i < shape.size() && static_cast<int>(i) < max_dims; ++i) {
        dims_out[i] = shape[i];
    }
    return 0;
}

int MNN_Session_getOutputDataFloat(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name,
    float* dst,
    int max_elements
) {
    if (!net || !session || !dst) {
        return -1;
    }
    auto* net_ptr = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s_ptr = reinterpret_cast<MNN::Session*>(session);

    auto* tensor = (name && name[0] != '\0')
        ? net_ptr->getSessionOutput(s_ptr, name)
        : net_ptr->getSessionOutput(s_ptr, nullptr);

    if (!tensor) {
        return -2;
    }

    MNN::Tensor host_out(tensor, MNN::Tensor::CAFFE);
    if (!tensor->copyToHostTensor(&host_out)) {
        return -3;
    }

    int count = host_out.elementSize();
    if (count > max_elements) {
        return -4;
    }

    std::memcpy(dst, host_out.host<float>(), count * sizeof(float));
    return count;
}

int MNN_Session_getOutputDataInt32(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name,
    int* dst,
    int max_elements
) {
    if (!net || !session || !dst) {
        return -1;
    }
    auto* net_ptr = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s_ptr = reinterpret_cast<MNN::Session*>(session);

    auto* tensor = (name && name[0] != '\0')
        ? net_ptr->getSessionOutput(s_ptr, name)
        : net_ptr->getSessionOutput(s_ptr, nullptr);

    if (!tensor) {
        return -2;
    }

    MNN::Tensor host_out(tensor, MNN::Tensor::CAFFE);
    if (!tensor->copyToHostTensor(&host_out)) {
        return -3;
    }

    int count = host_out.elementSize();
    if (count > max_elements) {
        return -4;
    }

    std::memcpy(dst, host_out.host<int32_t>(), count * sizeof(int32_t));
    return count;
}

int MNN_Session_getOutputNames(
    MNN_Interpreter* net,
    MNN_Session* session,
    char* names_buffer,
    int buffer_size,
    int* name_offsets,
    int max_names,
    int* name_count_out
) {
    if (!net || !session || !names_buffer || !name_offsets || !name_count_out) {
        return -1;
    }
    auto* net_ptr = reinterpret_cast<MNN::Interpreter*>(net);
    auto* s_ptr = reinterpret_cast<MNN::Session*>(session);

    auto outputs = net_ptr->getSessionOutputAll(s_ptr);
    int count = 0;
    int curr_offset = 0;

    for (const auto& pair : outputs) {
        if (count >= max_names) {
            break;
        }
        int len = static_cast<int>(pair.first.length());
        if (curr_offset + len + 1 > buffer_size) {
            break;
        }
        std::memcpy(names_buffer + curr_offset, pair.first.c_str(), len + 1);
        name_offsets[count] = curr_offset;
        curr_offset += len + 1;
        count++;
    }

    *name_count_out = count;
    return 0;
}

} // extern "C"

