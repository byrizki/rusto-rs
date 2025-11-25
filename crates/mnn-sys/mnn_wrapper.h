#ifndef MNN_WRAPPER_H
#define MNN_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types
typedef struct MNN_Interpreter MNN_Interpreter;
typedef struct MNN_Session MNN_Session;
typedef struct MNN_Tensor MNN_Tensor;

// Config structures
typedef struct {
    unsigned int precision;
    unsigned int power;
    unsigned int memory;
} MNN_BackendConfig;

typedef struct {
    unsigned int type_;
    int numThread;
    MNN_BackendConfig* backendConfig;
} MNN_ScheduleConfig;

// Enums
typedef enum {
    MNN_TENSOR_MAP_READ = 1,
    MNN_TENSOR_MAP_WRITE = 0
} MNN_Tensor_MapType;

// Tensor name-tensor pair for input/output enumeration
typedef struct {
    const char* name;
    MNN_Tensor* tensor;
} MNN_TensorInfo;

// Interpreter functions
MNN_Interpreter* MNN_Interpreter_createFromFile(const char* file);
MNN_Interpreter* MNN_Interpreter_createFromBuffer(const void* buffer, int size);
void MNN_Interpreter_destroy(MNN_Interpreter* net);

// Session management
MNN_Session* MNN_Interpreter_createSession(MNN_Interpreter* net, const MNN_ScheduleConfig* config);
void MNN_Interpreter_releaseSession(MNN_Interpreter* net, MNN_Session* session);
void MNN_Interpreter_resizeSession(MNN_Interpreter* net, MNN_Session* session);
int MNN_Interpreter_runSession(MNN_Interpreter* net, MNN_Session* session);

// Tensor access
MNN_Tensor* MNN_Interpreter_getSessionInput(MNN_Interpreter* net, MNN_Session* session, const char* name);
MNN_Tensor* MNN_Interpreter_getSessionOutput(MNN_Interpreter* net, MNN_Session* session, const char* name);

// Get all inputs/outputs
MNN_TensorInfo* MNN_Interpreter_getSessionInputAll(MNN_Interpreter* net, MNN_Session* session, int* size);
MNN_TensorInfo* MNN_Interpreter_getSessionOutputAll(MNN_Interpreter* net, MNN_Session* session, int* size);

// Tensor resize
void MNN_Interpreter_resizeTensor(MNN_Interpreter* net, MNN_Tensor* tensor, const int* dims, int size);

// Tensor functions
MNN_Tensor* MNN_Tensor_create(int dims, const int* shape, int type);
MNN_Tensor* MNN_Tensor_createHostTensorFromDevice(const MNN_Tensor* device, int copy);
void MNN_Tensor_destroy(MNN_Tensor* tensor);
int MNN_Tensor_copyFromHostTensor(MNN_Tensor* dst, const MNN_Tensor* src);
int MNN_Tensor_copyToHostTensor(const MNN_Tensor* src, MNN_Tensor* dst);
void* MNN_Tensor_getHost(const MNN_Tensor* tensor);
const int* MNN_Tensor_getDimensions(const MNN_Tensor* tensor, int* size);
int MNN_Tensor_wait(MNN_Tensor* tensor, MNN_Tensor_MapType type, int finish);

#ifdef __cplusplus
}
#endif

#endif // MNN_WRAPPER_H
