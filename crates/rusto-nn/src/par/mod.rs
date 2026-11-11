//! Data-parallel execution layer backed by a lightweight, dependency-free thread pool.

#[path = "iter.rs"]
pub mod prelude;
mod pool;

pub use self::pool::{current_num_threads, current_thread_index, set_num_threads};
