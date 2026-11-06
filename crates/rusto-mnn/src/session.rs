use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::Path;

use ndarray::{Array4, ArrayD, IxDyn};
use thiserror::Error;

use crate::ffi;

#[derive(Error, Debug)]
pub enum MnnCppError {
    #[error("Failed to load MNN model from path: {0}")]
    Load(String),

    #[error("Failed to create MNN session")]
    SessionCreation,

    #[error("Tensor operation failed: {0}")]
    Tensor(String),

    #[error("Execution failed with code: {0}")]
    Execution(i32),

    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
}

pub struct MnnCppSession {
    net: *mut ffi::MNN_Interpreter,
    session: *mut ffi::MNN_Session,
    channels: usize,
}

unsafe impl Send for MnnCppSession {}

impl MnnCppSession {
    pub fn from_file<P: AsRef<Path>>(path: P, num_threads: usize) -> Result<Self, MnnCppError> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| MnnCppError::Load("Invalid path unicode".into()))?;
        let c_path = CString::new(path_str)
            .map_err(|e| MnnCppError::Load(format!("CString conversion failed: {e}")))?;

        unsafe {
            let net = ffi::MNN_Interpreter_createFromFile(c_path.as_ptr());
            if net.is_null() {
                return Err(MnnCppError::Load(path_str.to_string()));
            }

            let threads = if num_threads > 0 {
                num_threads as i32
            } else {
                num_cpus_count() as i32
            };

            let session = ffi::MNN_Interpreter_createSession(net, threads, 0); // 0 = ForwardType::CPU
            if session.is_null() {
                ffi::MNN_Interpreter_destroy(net);
                return Err(MnnCppError::SessionCreation);
            }

            let ch = ffi::MNN_Session_getInputChannels(net, session, std::ptr::null());
            let channels = if ch > 0 { ch as usize } else { 3 };

            Ok(Self {
                net,
                session,
                channels,
            })
        }
    }

    pub fn input_channels(&self) -> usize {
        self.channels
    }

    pub fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, MnnCppError> {
        let shape: Vec<i32> = input.shape().iter().map(|&x| x as i32).collect();
        if shape.len() >= 2 {
            self.channels = shape[1] as usize;
        }

        let slice = if let Some(s) = input.as_slice() {
            s.to_vec()
        } else {
            input.iter().copied().collect()
        };

        unsafe {
            let res = ffi::MNN_Interpreter_resizeTensor(
                self.net,
                self.session,
                std::ptr::null(),
                shape.as_ptr(),
                shape.len() as i32,
            );
            if res != 0 {
                return Err(MnnCppError::Tensor(format!("Resize input tensor failed: {res}")));
            }

            ffi::MNN_Interpreter_resizeSession(self.net, self.session);

            let set_res = ffi::MNN_Session_setHostInputFloat(
                self.net,
                self.session,
                std::ptr::null(),
                slice.as_ptr(),
                slice.len() as i32,
            );
            if set_res != 0 {
                return Err(MnnCppError::Tensor(format!("Set host input failed: {set_res}")));
            }

            let run_res = ffi::MNN_Interpreter_runSession(self.net, self.session);
            if run_res != 0 {
                return Err(MnnCppError::Execution(run_res));
            }

            let mut out_dims = [0i32; 8];
            let mut dim_count = 0i32;
            let shape_res = ffi::MNN_Session_getOutputShape(
                self.net,
                self.session,
                std::ptr::null(),
                out_dims.as_mut_ptr(),
                8,
                &mut dim_count as *mut i32,
            );
            if shape_res != 0 {
                return Err(MnnCppError::Tensor(format!("Get output shape failed: {shape_res}")));
            }

            let out_shape: Vec<usize> = out_dims[..dim_count as usize]
                .iter()
                .map(|&d| d as usize)
                .collect();
            let total_elements: usize = out_shape.iter().product();

            let mut out_data = vec![0.0f32; total_elements];
            let copy_res = ffi::MNN_Session_getOutputDataFloat(
                self.net,
                self.session,
                std::ptr::null(),
                out_data.as_mut_ptr(),
                total_elements as i32,
            );
            if copy_res < 0 {
                return Err(MnnCppError::Tensor(format!("Get output data failed: {copy_res}")));
            }

            ArrayD::from_shape_vec(IxDyn(&out_shape), out_data).map_err(MnnCppError::Shape)
        }
    }

    pub fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, MnnCppError> {
        let shape: Vec<i32> = input.shape().iter().map(|&x| x as i32).collect();
        if shape.len() >= 2 {
            self.channels = shape[1] as usize;
        }

        let slice = if let Some(s) = input.as_slice() {
            s.to_vec()
        } else {
            input.iter().copied().collect()
        };

        unsafe {
            let res = ffi::MNN_Interpreter_resizeTensor(
                self.net,
                self.session,
                std::ptr::null(),
                shape.as_ptr(),
                shape.len() as i32,
            );
            if res != 0 {
                return Err(MnnCppError::Tensor(format!("Resize input tensor failed: {res}")));
            }

            ffi::MNN_Interpreter_resizeSession(self.net, self.session);

            let set_res = ffi::MNN_Session_setHostInputFloat(
                self.net,
                self.session,
                std::ptr::null(),
                slice.as_ptr(),
                slice.len() as i32,
            );
            if set_res != 0 {
                return Err(MnnCppError::Tensor(format!("Set host input failed: {set_res}")));
            }

            let run_res = ffi::MNN_Interpreter_runSession(self.net, self.session);
            if run_res != 0 {
                return Err(MnnCppError::Execution(run_res));
            }

            self.fetch_all_outputs()
        }
    }

    pub fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, MnnCppError> {
        unsafe {
            for (name, tensor) in &inputs {
                let c_name = CString::new(name.as_str())
                    .map_err(|e| MnnCppError::Tensor(e.to_string()))?;
                let shape: Vec<i32> = tensor.shape().iter().map(|&x| x as i32).collect();

                let res = ffi::MNN_Interpreter_resizeTensor(
                    self.net,
                    self.session,
                    c_name.as_ptr(),
                    shape.as_ptr(),
                    shape.len() as i32,
                );
                if res != 0 {
                    return Err(MnnCppError::Tensor(format!("Resize tensor '{name}' failed: {res}")));
                }
            }

            ffi::MNN_Interpreter_resizeSession(self.net, self.session);

            for (name, tensor) in &inputs {
                let c_name = CString::new(name.as_str())
                    .map_err(|e| MnnCppError::Tensor(e.to_string()))?;
                let slice = if let Some(s) = tensor.as_slice() {
                    s.to_vec()
                } else {
                    tensor.iter().copied().collect()
                };

                let set_res = ffi::MNN_Session_setHostInputFloat(
                    self.net,
                    self.session,
                    c_name.as_ptr(),
                    slice.as_ptr(),
                    slice.len() as i32,
                );
                if set_res != 0 {
                    return Err(MnnCppError::Tensor(format!("Set host input '{name}' failed: {set_res}")));
                }
            }

            let run_res = ffi::MNN_Interpreter_runSession(self.net, self.session);
            if run_res != 0 {
                return Err(MnnCppError::Execution(run_res));
            }

            self.fetch_all_outputs()
        }
    }

    pub fn run_linenet_rec(
        &mut self,
        x: Array4<f32>,
        widths: Vec<i32>,
    ) -> Result<(Vec<Vec<i32>>, Vec<Vec<f32>>), MnnCppError> {
        let x_shape: Vec<i32> = x.shape().iter().map(|&d| d as i32).collect();
        let widths_shape: Vec<i32> = vec![widths.len() as i32];

        let x_slice = if let Some(s) = x.as_slice() {
            s.to_vec()
        } else {
            x.iter().copied().collect()
        };

        unsafe {
            let c_x = CString::new("x").unwrap();
            let c_widths = CString::new("widths").unwrap();

            ffi::MNN_Interpreter_resizeTensor(
                self.net,
                self.session,
                c_x.as_ptr(),
                x_shape.as_ptr(),
                x_shape.len() as i32,
            );
            ffi::MNN_Interpreter_resizeTensor(
                self.net,
                self.session,
                c_widths.as_ptr(),
                widths_shape.as_ptr(),
                widths_shape.len() as i32,
            );

            ffi::MNN_Interpreter_resizeSession(self.net, self.session);

            ffi::MNN_Session_setHostInputFloat(
                self.net,
                self.session,
                c_x.as_ptr(),
                x_slice.as_ptr(),
                x_slice.len() as i32,
            );
            ffi::MNN_Session_setHostInputInt32(
                self.net,
                self.session,
                c_widths.as_ptr(),
                widths.as_ptr(),
                widths.len() as i32,
            );

            let run_res = ffi::MNN_Interpreter_runSession(self.net, self.session);
            if run_res != 0 {
                return Err(MnnCppError::Execution(run_res));
            }

            // Extract ids, conf, lengths
            let c_ids = CString::new("ids").unwrap();
            let c_conf = CString::new("conf").unwrap();
            let c_lengths = CString::new("lengths").unwrap();

            let mut ids_dims = [0i32; 8];
            let mut ids_dim_count = 0i32;
            ffi::MNN_Session_getOutputShape(
                self.net,
                self.session,
                c_ids.as_ptr(),
                ids_dims.as_mut_ptr(),
                8,
                &mut ids_dim_count as *mut i32,
            );

            let ids_shape: Vec<usize> = ids_dims[..ids_dim_count as usize]
                .iter()
                .map(|&d| d as usize)
                .collect();
            let total_ids: usize = ids_shape.iter().product();

            let mut ids_vec = vec![0i32; total_ids];
            ffi::MNN_Session_getOutputDataInt32(
                self.net,
                self.session,
                c_ids.as_ptr(),
                ids_vec.as_mut_ptr(),
                total_ids as i32,
            );

            let mut conf_vec = vec![0.0f32; total_ids];
            ffi::MNN_Session_getOutputDataFloat(
                self.net,
                self.session,
                c_conf.as_ptr(),
                conf_vec.as_mut_ptr(),
                total_ids as i32,
            );

            let batch_size = widths.len();
            let mut lengths_vec = vec![0i32; batch_size];
            ffi::MNN_Session_getOutputDataInt32(
                self.net,
                self.session,
                c_lengths.as_ptr(),
                lengths_vec.as_mut_ptr(),
                batch_size as i32,
            );

            let max_len = if ids_shape.len() >= 2 {
                ids_shape[1]
            } else {
                ids_vec.len() / batch_size.max(1)
            };

            let mut batch_ids = Vec::with_capacity(batch_size);
            let mut batch_confs = Vec::with_capacity(batch_size);

            for (b, &len_val) in lengths_vec.iter().enumerate().take(batch_size) {
                let len = len_val.max(0) as usize;
                let start = b * max_len;
                let actual_len = len.min(max_len);
                batch_ids.push(ids_vec[start..start + actual_len].to_vec());
                batch_confs.push(conf_vec[start..start + actual_len].to_vec());
            }

            Ok((batch_ids, batch_confs))
        }
    }

    unsafe fn fetch_all_outputs(&self) -> Result<HashMap<String, ArrayD<f32>>, MnnCppError> {
        let mut names_buf = vec![0i8; 2048];
        let mut offsets = [0i32; 64];
        let mut name_count = 0i32;

        let res = ffi::MNN_Session_getOutputNames(
            self.net,
            self.session,
            names_buf.as_mut_ptr(),
            names_buf.len() as i32,
            offsets.as_mut_ptr(),
            64,
            &mut name_count as *mut i32,
        );
        if res != 0 {
            return Err(MnnCppError::Tensor(format!("Get output names failed: {res}")));
        }

        let mut map = HashMap::with_capacity(name_count as usize);

        for i in 0..name_count as usize {
            let offset = offsets[i] as usize;
            let c_str = CStr::from_ptr(names_buf.as_ptr().add(offset));
            let name = c_str.to_string_lossy().into_owned();

            let mut dims = [0i32; 8];
            let mut dim_count = 0i32;
            ffi::MNN_Session_getOutputShape(
                self.net,
                self.session,
                c_str.as_ptr(),
                dims.as_mut_ptr(),
                8,
                &mut dim_count as *mut i32,
            );

            let out_shape: Vec<usize> = dims[..dim_count as usize]
                .iter()
                .map(|&d| d as usize)
                .collect();
            let total_elements: usize = out_shape.iter().product();

            let mut out_data = vec![0.0f32; total_elements];
            let copy_res = ffi::MNN_Session_getOutputDataFloat(
                self.net,
                self.session,
                c_str.as_ptr(),
                out_data.as_mut_ptr(),
                total_elements as i32,
            );

            if copy_res >= 0 {
                let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), out_data)
                    .map_err(MnnCppError::Shape)?;
                map.insert(name, arr);
            }
        }

        Ok(map)
    }
}

impl Drop for MnnCppSession {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() && !self.net.is_null() {
                ffi::MNN_Interpreter_releaseSession(self.net, self.session);
                self.session = std::ptr::null_mut();
            }
            if !self.net.is_null() {
                ffi::MNN_Interpreter_destroy(self.net);
                self.net = std::ptr::null_mut();
            }
        }
    }
}

impl rusto_core::engine::InferenceSession for MnnCppSession {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, rusto_core::CoreError> {
        self.run(input).map_err(|e| rusto_core::CoreError::Model(e.to_string()))
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, rusto_core::CoreError> {
        self.run_all(input).map_err(|e| rusto_core::CoreError::Model(e.to_string()))
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, rusto_core::CoreError> {
        self.run_with_inputs(inputs).map_err(|e| rusto_core::CoreError::Model(e.to_string()))
    }

    fn input_channels(&self) -> usize {
        self.input_channels()
    }
}

fn num_cpus_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

