use std::ffi::CString;
use std::path::Path;
use std::ptr::NonNull;
use crate::config::ScheduleConfig;
use crate::error::{MNNError, Result};
use crate::session::Session;
use crate::tensor::{Tensor, TensorInfo};

pub struct Interpreter {
    inner: NonNull<mnn_sys::MNN_Interpreter>,
}

impl Interpreter {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_str()
            .ok_or_else(|| MNNError::InterpreterCreation("Invalid path".to_string()))?;
        let c_path = CString::new(path_str)
            .map_err(|e| MNNError::InterpreterCreation(format!("Invalid path: {}", e)))?;

        unsafe {
            let ptr = mnn_sys::MNN_Interpreter_createFromFile(c_path.as_ptr());
            let inner = NonNull::new(ptr)
                .ok_or_else(|| MNNError::InterpreterCreation("Failed to create interpreter".to_string()))?;
            Ok(Self { inner })
        }
    }

    pub fn create_session(&self, config: ScheduleConfig) -> Result<Session> {
        unsafe {
            // Create MNN schedule config
            let mut mnn_config = mnn_sys::MNN_ScheduleConfig {
                type_: config.forward_type() as u32,
                numThread: config.num_thread(),
                backendConfig: std::ptr::null_mut(),
            };

            // Set backend config if provided
            let mut backend_config_ffi = config.backend_config().map(|bc| mnn_sys::MNN_BackendConfig {
                precision: bc.precision() as u32,
                power: bc.power() as u32,
                memory: 0, // Memory_Normal
            });
            
            if let Some(ref mut bc) = backend_config_ffi {
                mnn_config.backendConfig = bc as *mut _;
            }

            let session_ptr = mnn_sys::MNN_Interpreter_createSession(
                self.inner.as_ptr(),
                &mnn_config as *const _,
            );

            let session_inner = NonNull::new(session_ptr)
                .ok_or_else(|| MNNError::SessionCreation("Failed to create session".to_string()))?;

            Ok(Session::new(session_inner, self.inner))
        }
    }

    pub fn inputs(&self, session: &Session) -> Vec<TensorInfo> {
        unsafe {
            let mut size: i32 = 0;
            let tensors_ptr = mnn_sys::MNN_Interpreter_getSessionInputAll(
                self.inner.as_ptr(),
                session.as_ptr(),
                &mut size as *mut i32,
            );

            if tensors_ptr.is_null() || size == 0 {
                return Vec::new();
            }

            let tensors_slice = std::slice::from_raw_parts(tensors_ptr, size as usize);
            tensors_slice.iter()
                .filter_map(|entry| {
                    if entry.tensor.is_null() || entry.name.is_null() {
                        return None;
                    }
                    let name = std::ffi::CStr::from_ptr(entry.name)
                        .to_string_lossy()
                        .into_owned();
                    Some(TensorInfo {
                        name,
                        _tensor_ptr: entry.tensor,
                    })
                })
                .collect()
        }
    }

    pub fn outputs(&self, session: &Session) -> Vec<TensorInfo> {
        unsafe {
            let mut size: i32 = 0;
            let tensors_ptr = mnn_sys::MNN_Interpreter_getSessionOutputAll(
                self.inner.as_ptr(),
                session.as_ptr(),
                &mut size as *mut i32,
            );

            if tensors_ptr.is_null() || size == 0 {
                return Vec::new();
            }

            let tensors_slice = std::slice::from_raw_parts(tensors_ptr, size as usize);
            tensors_slice.iter()
                .filter_map(|entry| {
                    if entry.tensor.is_null() || entry.name.is_null() {
                        return None;
                    }
                    let name = std::ffi::CStr::from_ptr(entry.name)
                        .to_string_lossy()
                        .into_owned();
                    Some(TensorInfo {
                        name,
                        _tensor_ptr: entry.tensor,
                    })
                })
                .collect()
        }
    }

    pub fn input<'a, T>(&self, session: &'a mut Session, name: &str) -> Result<Tensor<'a>> {
        let c_name = CString::new(name)
            .map_err(|e| MNNError::TensorAccess(format!("Invalid name: {}", e)))?;

        unsafe {
            let tensor_ptr = mnn_sys::MNN_Interpreter_getSessionInput(
                self.inner.as_ptr(),
                session.as_ptr(),
                c_name.as_ptr(),
            );

            let inner = NonNull::new(tensor_ptr)
                .ok_or_else(|| MNNError::TensorAccess(format!("Tensor '{}' not found", name)))?;

            Ok(Tensor::new(inner))
        }
    }

    pub unsafe fn input_unresized<'a, T>(&self, session: &'a mut Session, name: &str) -> Result<Tensor<'a>> {
        self.input::<T>(session, name)
    }

    pub fn output<'a, T>(&self, session: &'a Session, name: &str) -> Result<Tensor<'a>> {
        let c_name = CString::new(name)
            .map_err(|e| MNNError::TensorAccess(format!("Invalid name: {}", e)))?;

        unsafe {
            let tensor_ptr = mnn_sys::MNN_Interpreter_getSessionOutput(
                self.inner.as_ptr(),
                session.as_ptr(),
                c_name.as_ptr(),
            );

            let inner = NonNull::new(tensor_ptr)
                .ok_or_else(|| MNNError::TensorAccess(format!("Output tensor '{}' not found", name)))?;

            Ok(Tensor::new(inner))
        }
    }

    pub fn resize_tensor(&self, tensor: &mut Tensor, shape: &[i32]) {
        unsafe {
            let dims = shape.as_ptr();
            mnn_sys::MNN_Interpreter_resizeTensor(
                self.inner.as_ptr(),
                tensor.as_ptr(),
                dims,
                shape.len() as i32,
            );
        }
    }

    pub fn resize_session(&self, session: &mut Session) {
        unsafe {
            mnn_sys::MNN_Interpreter_resizeSession(self.inner.as_ptr(), session.as_ptr());
        }
    }

    pub fn run_session(&self, session: &mut Session) -> Result<()> {
        unsafe {
            let error_code = mnn_sys::MNN_Interpreter_runSession(
                self.inner.as_ptr(),
                session.as_ptr(),
            );

            if error_code != 0 {
                return Err(MNNError::SessionRun(format!("Error code: {}", error_code)));
            }

            Ok(())
        }
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        unsafe {
            mnn_sys::MNN_Interpreter_destroy(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for Interpreter {}
unsafe impl Sync for Interpreter {}
