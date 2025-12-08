use std::marker::PhantomData;
use std::ptr::NonNull;
use crate::error::{MNNError, Result};

pub struct TensorInfo {
    pub name: String,
    pub(crate) _tensor_ptr: *mut mnn_sys::MNN_Tensor,
}

impl TensorInfo {
    pub fn name(&self) -> &str {
        &self.name
    }
}

pub struct Tensor<'a> {
    inner: NonNull<mnn_sys::MNN_Tensor>,
    _phantom: PhantomData<&'a ()>,
}

impl<'a> Tensor<'a> {
    pub(crate) fn new(inner: NonNull<mnn_sys::MNN_Tensor>) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut mnn_sys::MNN_Tensor {
        self.inner.as_ptr()
    }

    pub fn shape(&self) -> Vec<i32> {
        unsafe {
            let mut size: i32 = 0;
            let dims_ptr = mnn_sys::MNN_Tensor_getDimensions(self.inner.as_ptr(), &mut size as *mut i32);
            
            if dims_ptr.is_null() || size == 0 {
                return Vec::new();
            }

            std::slice::from_raw_parts(dims_ptr, size as usize).to_vec()
        }
    }

    pub fn create_host_tensor_from_device(&self, copy_data: bool) -> HostTensor {
        unsafe {
            let host_ptr = mnn_sys::MNN_Tensor_createHostTensorFromDevice(
                self.inner.as_ptr(),
                copy_data as i32,
            );

            let inner = NonNull::new(host_ptr)
                .expect("Failed to create host tensor");

            HostTensor { inner }
        }
    }

    pub fn copy_from_host_tensor(&mut self, host: &HostTensor) -> Result<()> {
        unsafe {
            let success = mnn_sys::MNN_Tensor_copyFromHostTensor(
                self.inner.as_ptr(),
                host.inner.as_ptr(),
            );

            if success == 0 {
                Err(MNNError::TensorCopy("Failed to copy from host tensor".to_string()))
            } else {
                Ok(())
            }
        }
    }

    pub fn copy_to_host_tensor(&self, host: &mut HostTensor) -> Result<()> {
        unsafe {
            let success = mnn_sys::MNN_Tensor_copyToHostTensor(
                self.inner.as_ptr(),
                host.inner.as_ptr(),
            );

            if success == 0 {
                Err(MNNError::TensorCopy("Failed to copy to host tensor".to_string()))
            } else {
                Ok(())
            }
        }
    }

    pub fn wait(&self, map_type: crate::ffi::MapType, finish: bool) {
        unsafe {
            let mnn_map_type: u32 = map_type as u32;
            mnn_sys::MNN_Tensor_wait(self.inner.as_ptr(), mnn_map_type.try_into().unwrap(), finish as i32);
        }
    }
}

impl Tensor<'_> {
    /// Create a new host tensor with the given shape and default (NCHW) layout
    pub fn new_host(shape: &[i32]) -> HostTensor {
        unsafe {
            let dims = shape.len() as i32;
            let tensor_ptr = mnn_sys::MNN_Tensor_create(
                dims,
                shape.as_ptr(),
                2, // 2 = float32 (Halide type code)
            );
            
            let inner = NonNull::new(tensor_ptr)
                .expect("Failed to create host tensor");

            HostTensor { inner }
        }
    }
}

pub struct HostTensor {
    inner: NonNull<mnn_sys::MNN_Tensor>,
}

impl HostTensor {
    pub fn host<T>(&self) -> &[T] {
        unsafe {
            let ptr = mnn_sys::MNN_Tensor_getHost(self.inner.as_ptr()) as *const T;
            let mut size: i32 = 0;
            let dims_ptr = mnn_sys::MNN_Tensor_getDimensions(self.inner.as_ptr(), &mut size as *mut i32);
            
            if ptr.is_null() || dims_ptr.is_null() || size == 0 {
                return &[];
            }

            let dims = std::slice::from_raw_parts(dims_ptr, size as usize);
            let total_size: usize = dims.iter().map(|&d| d as usize).product();

            if total_size == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(ptr, total_size)
            }
        }
    }

    pub fn host_mut<T>(&mut self) -> &mut [T] {
        unsafe {
            let ptr = mnn_sys::MNN_Tensor_getHost(self.inner.as_ptr()) as *mut T;
            let mut size: i32 = 0;
            let dims_ptr = mnn_sys::MNN_Tensor_getDimensions(self.inner.as_ptr(), &mut size as *mut i32);
            
            if ptr.is_null() || dims_ptr.is_null() || size == 0 {
                return &mut [];
            }

            let dims = std::slice::from_raw_parts(dims_ptr, size as usize);
            let total_size: usize = dims.iter().map(|&d| d as usize).product();

            if total_size == 0 {
                &mut []
            } else {
                std::slice::from_raw_parts_mut(ptr, total_size)
            }
        }
    }
}

impl Drop for HostTensor {
    fn drop(&mut self) {
        unsafe {
            mnn_sys::MNN_Tensor_destroy(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for HostTensor {}
unsafe impl Sync for HostTensor {}
