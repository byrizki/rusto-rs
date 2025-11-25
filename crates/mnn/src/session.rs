use std::ptr::NonNull;

pub struct Session {
    inner: NonNull<mnn_sys::MNN_Session>,
    _interpreter: NonNull<mnn_sys::MNN_Interpreter>,
}

impl Session {
    pub(crate) fn new(
        inner: NonNull<mnn_sys::MNN_Session>,
        interpreter: NonNull<mnn_sys::MNN_Interpreter>,
    ) -> Self {
        Self {
            inner,
            _interpreter: interpreter,
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut mnn_sys::MNN_Session {
        self.inner.as_ptr()
    }
}

// Session is released by the Interpreter, so we don't need to implement Drop
unsafe impl Send for Session {}
