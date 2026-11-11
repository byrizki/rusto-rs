/// Sizes the global worker pool to `threads` workers before its first use.
///
/// The request is ignored when `threads` is zero, when `RUSTO_NUM_THREADS` (or `RAYON_NUM_THREADS`) is set,
/// or when the pool has already been initialised.
pub fn init_thread_pool(threads: usize) {
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    {
        use std::sync::Once;
        static INIT: Once = Once::new();

        if threads == 0
            || std::env::var_os("RUSTO_NUM_THREADS").is_some()
            || std::env::var_os("RAYON_NUM_THREADS").is_some()
        {
            return;
        }
        INIT.call_once(|| {
            crate::par::set_num_threads(threads);
        });
    }
    #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
    let _ = threads;
}
