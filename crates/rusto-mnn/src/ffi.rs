use std::os::raw::{c_char, c_float, c_int};

#[allow(non_camel_case_types)]
pub enum MNN_Interpreter {}
#[allow(non_camel_case_types)]
pub enum MNN_Session {}

extern "C" {
    pub fn MNN_Interpreter_createFromFile(file_path: *const c_char) -> *mut MNN_Interpreter;
    pub fn MNN_Interpreter_destroy(net: *mut MNN_Interpreter);

    pub fn MNN_Interpreter_createSession(
        net: *mut MNN_Interpreter,
        num_threads: c_int,
        forward_type: c_int,
    ) -> *mut MNN_Session;

    pub fn MNN_Interpreter_releaseSession(
        net: *mut MNN_Interpreter,
        session: *mut MNN_Session,
    );

    pub fn MNN_Interpreter_resizeTensor(
        net: *mut MNN_Interpreter,
        session: *mut MNN_Session,
        name: *const c_char,
        dims: *const c_int,
        dim_count: c_int,
    ) -> c_int;

    pub fn MNN_Interpreter_resizeSession(
        net: *mut MNN_Interpreter,
        session: *mut MNN_Session,
    );

    pub fn MNN_Interpreter_runSession(
        net: *mut MNN_Interpreter,
        session: *mut MNN_Session,
    ) -> c_int;

    pub fn MNN_Session_setHostInputFloat(
        net: *mut MNN_Interpreter,
        session: *mut MNN_Session,
        name: *const c_char,
        data: *const c_float,
        count: c_int,
    ) -> c_int;

    pub fn MNN_Session_setHostInputInt32(
        net: *mut MNN_Interpreter,
        session: *mut MNN_Session,
        name: *const c_char,
        data: *const c_int,
        count: c_int,
    ) -> c_int;

    pub fn MNN_Session_getInputChannels(
        net: *mut MNN_Interpreter,
        session: *mut MNN_Session,
        name: *const c_char,
    ) -> c_int;

    pub fn MNN_Session_getOutputShape(
        net: *mut MNN_Interpreter,
        session: *mut MNN_Session,
        name: *const c_char,
        dims_out: *mut c_int,
        max_dims: c_int,
        dim_count_out: *mut c_int,
    ) -> c_int;

    pub fn MNN_Session_getOutputDataFloat(
        net: *mut MNN_Interpreter,
        session: *mut MNN_Session,
        name: *const c_char,
        dst: *mut c_float,
        max_elements: c_int,
    ) -> c_int;

    pub fn MNN_Session_getOutputDataInt32(
        net: *mut MNN_Interpreter,
        session: *mut MNN_Session,
        name: *const c_char,
        dst: *mut c_int,
        max_elements: c_int,
    ) -> c_int;

    pub fn MNN_Session_getOutputNames(
        net: *mut MNN_Interpreter,
        session: *mut MNN_Session,
        names_buffer: *mut c_char,
        buffer_size: c_int,
        name_offsets: *mut c_int,
        max_names: c_int,
        name_count_out: *mut c_int,
    ) -> c_int;
}
