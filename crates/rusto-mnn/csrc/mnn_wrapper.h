#ifndef RUSTO_MNN_WRAPPER_H
#define RUSTO_MNN_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MNN_Interpreter MNN_Interpreter;
typedef struct MNN_Session MNN_Session;

MNN_Interpreter* MNN_Interpreter_createFromFile(const char* file_path);
void MNN_Interpreter_destroy(MNN_Interpreter* net);

MNN_Session* MNN_Interpreter_createSession(MNN_Interpreter* net, int num_threads, int forward_type);
void MNN_Interpreter_releaseSession(MNN_Interpreter* net, MNN_Session* session);

int MNN_Interpreter_resizeTensor(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name,
    const int* dims,
    int dim_count
);

void MNN_Interpreter_resizeSession(MNN_Interpreter* net, MNN_Session* session);

int MNN_Interpreter_runSession(MNN_Interpreter* net, MNN_Session* session);

int MNN_Session_setHostInputFloat(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name,
    const float* data,
    int count
);

int MNN_Session_setHostInputInt32(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name,
    const int* data,
    int count
);

int MNN_Session_getInputChannels(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name
);

int MNN_Session_getOutputShape(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name,
    int* dims_out,
    int max_dims,
    int* dim_count_out
);

int MNN_Session_getOutputDataFloat(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name,
    float* dst,
    int max_elements
);

int MNN_Session_getOutputDataInt32(
    MNN_Interpreter* net,
    MNN_Session* session,
    const char* name,
    int* dst,
    int max_elements
);

int MNN_Session_getOutputNames(
    MNN_Interpreter* net,
    MNN_Session* session,
    char* names_buffer,
    int buffer_size,
    int* name_offsets,
    int max_names,
    int* name_count_out
);

#ifdef __cplusplus
}
#endif

#endif // RUSTO_MNN_WRAPPER_H

