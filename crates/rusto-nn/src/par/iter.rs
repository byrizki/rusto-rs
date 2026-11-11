//! Minimal indexed parallel iterators mirroring the subset of Rayon's prelude used by the kernels.

use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::Range;

use super::pool::{current_num_threads, run};

pub trait ParIter: Sized + Sync {
    type Item: Send;

    fn len(&self) -> usize;

    /// # Safety
    /// Each index in `0..len()` must be fetched at most once.
    unsafe fn get(&self, i: usize) -> Self::Item;

    fn enumerate(self) -> Enumerate<Self> {
        Enumerate { inner: self }
    }

    fn zip<B: ParIter>(self, other: B) -> Zip<Self, B> {
        Zip { a: self, b: other }
    }

    fn map<R: Send, F: Fn(Self::Item) -> R + Sync + Send>(self, f: F) -> Map<Self, F> {
        Map { inner: self, f }
    }

    fn for_each<F: Fn(Self::Item) + Sync + Send>(self, f: F) {
        let len = self.len();
        let grain = (len / (current_num_threads() * 4)).max(1);
        let tasks = len.div_ceil(grain);
        let this = &self;
        run(tasks, &|t| {
            for i in t * grain..((t + 1) * grain).min(len) {
                f(unsafe { this.get(i) });
            }
        });
    }

    fn collect<C: FromIterator<Self::Item>>(self) -> C {
        let len = self.len();
        let mut slots: Vec<Option<Self::Item>> = Vec::with_capacity(len);
        slots.resize_with(len, || None);
        let out = SendPtr(slots.as_mut_ptr());
        let this = &self;
        run(len, &|i| unsafe {
            *out.get().add(i) = Some(this.get(i));
        });
        slots.into_iter().map(|s| s.expect("parallel slot filled")).collect()
    }
}

struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}

pub struct Enumerate<P> {
    inner: P,
}

impl<P: ParIter> ParIter for Enumerate<P> {
    type Item = (usize, P::Item);
    fn len(&self) -> usize {
        self.inner.len()
    }
    unsafe fn get(&self, i: usize) -> Self::Item {
        (i, self.inner.get(i))
    }
}

pub struct Zip<A, B> {
    a: A,
    b: B,
}

impl<A: ParIter, B: ParIter> ParIter for Zip<A, B> {
    type Item = (A::Item, B::Item);
    fn len(&self) -> usize {
        self.a.len().min(self.b.len())
    }
    unsafe fn get(&self, i: usize) -> Self::Item {
        (self.a.get(i), self.b.get(i))
    }
}

pub struct Map<P, F> {
    inner: P,
    f: F,
}

impl<P: ParIter, R: Send, F: Fn(P::Item) -> R + Sync + Send> ParIter for Map<P, F> {
    type Item = R;
    fn len(&self) -> usize {
        self.inner.len()
    }
    unsafe fn get(&self, i: usize) -> R {
        (self.f)(self.inner.get(i))
    }
}

pub struct RangeIter(Range<usize>);

impl ParIter for RangeIter {
    type Item = usize;
    fn len(&self) -> usize {
        self.0.len()
    }
    unsafe fn get(&self, i: usize) -> usize {
        self.0.start + i
    }
}

pub struct ChunksMut<'a, T> {
    ptr: *mut T,
    len: usize,
    chunk: usize,
    exact: bool,
    _marker: PhantomData<&'a mut [T]>,
}

unsafe impl<T: Send> Send for ChunksMut<'_, T> {}
unsafe impl<T: Send> Sync for ChunksMut<'_, T> {}

impl<'a, T: Send> ParIter for ChunksMut<'a, T> {
    type Item = &'a mut [T];
    fn len(&self) -> usize {
        if self.exact {
            self.len / self.chunk
        } else {
            self.len.div_ceil(self.chunk)
        }
    }
    unsafe fn get(&self, i: usize) -> &'a mut [T] {
        let start = i * self.chunk;
        std::slice::from_raw_parts_mut(self.ptr.add(start), self.chunk.min(self.len - start))
    }
}

pub struct Chunks<'a, T> {
    slice: &'a [T],
    chunk: usize,
}

impl<'a, T: Sync> ParIter for Chunks<'a, T> {
    type Item = &'a [T];
    fn len(&self) -> usize {
        self.slice.len().div_ceil(self.chunk)
    }
    unsafe fn get(&self, i: usize) -> &'a [T] {
        let start = i * self.chunk;
        &self.slice[start..(start + self.chunk).min(self.slice.len())]
    }
}

pub struct SliceIter<'a, T>(&'a [T]);

impl<'a, T: Sync> ParIter for SliceIter<'a, T> {
    type Item = &'a T;
    fn len(&self) -> usize {
        self.0.len()
    }
    unsafe fn get(&self, i: usize) -> &'a T {
        &self.0[i]
    }
}

pub struct VecIter<T> {
    vec: ManuallyDrop<Vec<T>>,
}

unsafe impl<T: Send> Sync for VecIter<T> {}

impl<T: Send> ParIter for VecIter<T> {
    type Item = T;
    fn len(&self) -> usize {
        self.vec.len()
    }
    unsafe fn get(&self, i: usize) -> T {
        std::ptr::read(self.vec.as_ptr().add(i))
    }
}

impl<T> Drop for VecIter<T> {
    fn drop(&mut self) {
        // Items are moved out by `get`; only the allocation remains to free.
        unsafe {
            self.vec.set_len(0);
            ManuallyDrop::drop(&mut self.vec);
        }
    }
}

pub trait IntoParallelIterator {
    type Iter: ParIter;
    fn into_par_iter(self) -> Self::Iter;
}

impl IntoParallelIterator for Range<usize> {
    type Iter = RangeIter;
    fn into_par_iter(self) -> RangeIter {
        RangeIter(self)
    }
}

impl<T: Send> IntoParallelIterator for Vec<T> {
    type Iter = VecIter<T>;
    fn into_par_iter(self) -> VecIter<T> {
        VecIter { vec: ManuallyDrop::new(self) }
    }
}

pub trait ParallelSlice<T: Sync> {
    fn par_chunks(&self, chunk: usize) -> Chunks<'_, T>;
    fn par_iter(&self) -> SliceIter<'_, T>;
}

impl<T: Sync> ParallelSlice<T> for [T] {
    fn par_chunks(&self, chunk: usize) -> Chunks<'_, T> {
        assert!(chunk != 0, "chunk size must be non-zero");
        Chunks { slice: self, chunk }
    }
    fn par_iter(&self) -> SliceIter<'_, T> {
        SliceIter(self)
    }
}

pub trait ParallelSliceMut<T: Send> {
    fn par_chunks_mut(&mut self, chunk: usize) -> ChunksMut<'_, T>;
    fn par_chunks_exact_mut(&mut self, chunk: usize) -> ChunksMut<'_, T>;
}

impl<T: Send> ParallelSliceMut<T> for [T] {
    fn par_chunks_mut(&mut self, chunk: usize) -> ChunksMut<'_, T> {
        assert!(chunk != 0, "chunk size must be non-zero");
        ChunksMut { ptr: self.as_mut_ptr(), len: self.len(), chunk, exact: false, _marker: PhantomData }
    }
    fn par_chunks_exact_mut(&mut self, chunk: usize) -> ChunksMut<'_, T> {
        assert!(chunk != 0, "chunk size must be non-zero");
        ChunksMut { ptr: self.as_mut_ptr(), len: self.len(), chunk, exact: true, _marker: PhantomData }
    }
}

