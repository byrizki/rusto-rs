use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

thread_local! {
    static IN_PARALLEL: Cell<bool> = const { Cell::new(false) };
}

static CONFIGURED_THREADS: AtomicUsize = AtomicUsize::new(0);
static POOL: OnceLock<Pool> = OnceLock::new();

struct Job {
    task: &'static (dyn Fn(usize) + Sync),
    len: usize,
    next: AtomicUsize,
    done: AtomicUsize,
    finished: Mutex<()>,
    finished_cv: Condvar,
}

impl Job {
    fn drain(&self) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.len {
                return;
            }
            (self.task)(i);
            if self.done.fetch_add(1, Ordering::AcqRel) + 1 == self.len {
                let _guard = self.finished.lock().unwrap();
                self.finished_cv.notify_all();
            }
        }
    }

    fn has_work(&self) -> bool {
        self.next.load(Ordering::Relaxed) < self.len
    }
}

struct Pool {
    threads: usize,
    queue: Mutex<VecDeque<Arc<Job>>>,
    work_cv: Condvar,
}

fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let configured = CONFIGURED_THREADS.load(Ordering::Relaxed);
        let env_threads = std::env::var("RUSTO_NUM_THREADS")
            .or_else(|_| std::env::var("RAYON_NUM_THREADS"))
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let threads = if configured > 0 {
            configured
        } else if env_threads > 0 {
            env_threads
        } else {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        };
        let pool = Pool {
            threads,
            queue: Mutex::new(VecDeque::new()),
            work_cv: Condvar::new(),
        };
        for id in 1..threads {
            let _ = std::thread::Builder::new()
                .name(format!("rusto-nn-{id}"))
                .spawn(worker_loop);
        }
        pool
    })
}

fn worker_loop() {
    IN_PARALLEL.with(|f| f.set(true));
    let pool = pool();
    loop {
        let job = {
            let mut queue = pool.queue.lock().unwrap();
            loop {
                if let Some(job) = queue.iter().find(|j| j.has_work()) {
                    break job.clone();
                }
                queue = pool.work_cv.wait(queue).unwrap();
            }
        };
        job.drain();
    }
}

/// Requests a pool size; only effective before the pool is first used.
pub fn set_num_threads(threads: usize) {
    CONFIGURED_THREADS.store(threads, Ordering::Relaxed);
}

pub fn current_num_threads() -> usize {
    pool().threads
}

/// `Some` when called from inside a parallel region, so callers can avoid nesting.
pub fn current_thread_index() -> Option<usize> {
    IN_PARALLEL.with(|f| f.get()).then_some(0)
}

/// Runs `task(0..len)` across the pool, with the caller participating, and
/// blocks until every index has completed.
pub fn run(len: usize, task: &(dyn Fn(usize) + Sync)) {
    if len == 0 {
        return;
    }
    if len == 1 || current_thread_index().is_some() || current_num_threads() == 1 {
        (0..len).for_each(task);
        return;
    }

    // SAFETY: `run` does not return until `done == len`, and no thread invokes
    // `task` after `next >= len`, so the erased reference never outlives `task`.
    let task: &'static (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(task) };
    let job = Arc::new(Job {
        task,
        len,
        next: AtomicUsize::new(0),
        done: AtomicUsize::new(0),
        finished: Mutex::new(()),
        finished_cv: Condvar::new(),
    });

    let pool = pool();
    pool.queue.lock().unwrap().push_back(job.clone());
    for _ in 0..(len - 1).min(pool.threads - 1) {
        pool.work_cv.notify_one();
    }

    IN_PARALLEL.with(|f| f.set(true));
    job.drain();
    IN_PARALLEL.with(|f| f.set(false));

    let mut guard = job.finished.lock().unwrap();
    while job.done.load(Ordering::Acquire) < len {
        guard = job.finished_cv.wait(guard).unwrap();
    }
    drop(guard);
    pool.queue.lock().unwrap().retain(|j| !Arc::ptr_eq(j, &job));
}

