mod interpreter;
mod session;
mod tensor;
mod config;
mod error;

pub use interpreter::Interpreter;
pub use session::Session;
pub use tensor::{Tensor, HostTensor, TensorInfo};
pub use config::{ScheduleConfig, BackendConfig, ForwardType, PrecisionMode, PowerMode};
pub use error::{MNNError, Result};

pub mod ffi {
    pub use mnn_sys::*;
    
    // Re-export MapType for compatibility
    #[repr(u32)]
    #[derive(Debug, Clone, Copy)]
    #[allow(non_camel_case_types)]
    pub enum MapType {
        MAP_TENSOR_READ = 1,
        MAP_TENSOR_WRITE = 0,
    }
}
