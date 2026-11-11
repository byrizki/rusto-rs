#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::excessive_precision)]

pub mod backend;
pub mod error;
pub mod graph;
pub mod kernels;
pub mod model;
#[cfg(feature = "parallel")]
pub mod par;
pub mod schema;
pub mod session;
pub mod tensor;
pub mod threads;

pub use backend::{active_backend, clear_backend, register_backend, DeviceBackend};
pub use error::MnnError;
pub use graph::{alloc_tensor_buffer_overwrite, clear_tensor_buffer_pool, recycle_tensor_buffer};
pub use model::Model;
pub use session::Session;
pub use tensor::{DataType, Shape, Tensor, TensorLayout};
pub use threads::init_thread_pool;
