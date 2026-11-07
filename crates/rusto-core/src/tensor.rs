use std::fmt;
use std::ops::{Index, IndexMut};
use crate::error::CoreError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Axis(pub usize);

impl From<usize> for Axis {
    #[inline]
    fn from(val: usize) -> Self {
        Self(val)
    }
}

pub trait IntoShape {
    fn into_shape(self) -> Vec<usize>;
}

impl IntoShape for &[usize] {
    #[inline]
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> IntoShape for [usize; N] {
    #[inline]
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> IntoShape for &[usize; N] {
    #[inline]
    fn into_shape(self) -> Vec<usize> {
        self.as_slice().to_vec()
    }
}

impl IntoShape for Vec<usize> {
    #[inline]
    fn into_shape(self) -> Vec<usize> {
        self
    }
}

impl IntoShape for &Vec<usize> {
    #[inline]
    fn into_shape(self) -> Vec<usize> {
        self.clone()
    }
}

impl IntoShape for usize {
    #[inline]
    fn into_shape(self) -> Vec<usize> {
        vec![self]
    }
}

impl IntoShape for (usize,) {
    #[inline]
    fn into_shape(self) -> Vec<usize> {
        vec![self.0]
    }
}

impl IntoShape for (usize, usize) {
    #[inline]
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}

impl IntoShape for (usize, usize, usize) {
    #[inline]
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
}

impl IntoShape for (usize, usize, usize, usize) {
    #[inline]
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3]
    }
}

pub trait FromShape {
    fn from_shape(shape: &[usize]) -> Self;
}

impl FromShape for (usize, usize) {
    #[inline]
    fn from_shape(s: &[usize]) -> Self {
        (
            s.first().copied().unwrap_or(0),
            s.get(1).copied().unwrap_or(0),
        )
    }
}

impl FromShape for (usize, usize, usize) {
    #[inline]
    fn from_shape(s: &[usize]) -> Self {
        (
            s.first().copied().unwrap_or(0),
            s.get(1).copied().unwrap_or(0),
            s.get(2).copied().unwrap_or(0),
        )
    }
}

impl FromShape for (usize, usize, usize, usize) {
    #[inline]
    fn from_shape(s: &[usize]) -> Self {
        (
            s.first().copied().unwrap_or(0),
            s.get(1).copied().unwrap_or(0),
            s.get(2).copied().unwrap_or(0),
            s.get(3).copied().unwrap_or(0),
        )
    }
}

impl<const N: usize> FromShape for [usize; N] {
    #[inline]
    fn from_shape(s: &[usize]) -> Self {
        let mut out = [0; N];
        for i in 0..N.min(s.len()) {
            out[i] = s[i];
        }
        out
    }
}

pub trait TensorIndex {
    fn offset(&self, shape: &[usize]) -> Option<usize>;
}

impl TensorIndex for [usize; 4] {
    #[inline]
    fn offset(&self, shape: &[usize]) -> Option<usize> {
        if shape.len() == 4
            && self[0] < shape[0]
            && self[1] < shape[1]
            && self[2] < shape[2]
            && self[3] < shape[3]
        {
            Some(((self[0] * shape[1] + self[1]) * shape[2] + self[2]) * shape[3] + self[3])
        } else {
            None
        }
    }
}

impl TensorIndex for [usize; 3] {
    #[inline]
    fn offset(&self, shape: &[usize]) -> Option<usize> {
        if shape.len() == 3 && self[0] < shape[0] && self[1] < shape[1] && self[2] < shape[2] {
            Some((self[0] * shape[1] + self[1]) * shape[2] + self[2])
        } else {
            None
        }
    }
}

impl TensorIndex for [usize; 2] {
    #[inline]
    fn offset(&self, shape: &[usize]) -> Option<usize> {
        if shape.len() == 2 && self[0] < shape[0] && self[1] < shape[1] {
            Some(self[0] * shape[1] + self[1])
        } else {
            None
        }
    }
}

impl TensorIndex for usize {
    #[inline]
    fn offset(&self, shape: &[usize]) -> Option<usize> {
        let total = shape.iter().product::<usize>();
        if *self < total {
            Some(*self)
        } else {
            None
        }
    }
}

/// Contiguous multidimensional tensor with row-major (C-contiguous) layout.
#[derive(Clone, PartialEq)]
pub struct Tensor<T = f32> {
    shape: Vec<usize>,
    data: Vec<T>,
}

impl<T: fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("len", &self.data.len())
            .finish()
    }
}

impl<T: Default + Clone> Tensor<T> {
    /// Create new zero/default-initialized tensor
    pub fn zeros(shape: impl IntoShape) -> Self {
        let shape = shape.into_shape();
        let total = shape.iter().product::<usize>();
        Self {
            shape,
            data: vec![T::default(); total],
        }
    }
}

impl<T> Tensor<T> {
    /// Create tensor from raw vector data
    pub fn from_data(shape: impl IntoShape, data: Vec<T>) -> Self {
        Self {
            shape: shape.into_shape(),
            data,
        }
    }

    /// Create tensor from raw vector data, validating total element count
    pub fn from_shape_vec(shape: impl IntoShape, data: Vec<T>) -> Result<Self, CoreError> {
        let s = shape.into_shape();
        let expected = s.iter().product::<usize>();
        if data.len() != expected {
            return Err(CoreError::InvalidShape(format!(
                "tensor data len {} does not match shape {:?} (expected {})",
                data.len(),
                s,
                expected
            )));
        }
        Ok(Self { shape: s, data })
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn dim<D: FromShape>(&self) -> D {
        D::from_shape(&self.shape)
    }

    #[inline]
    pub fn dim2(&self) -> (usize, usize) {
        self.dim()
    }

    #[inline]
    pub fn dim3(&self) -> (usize, usize, usize) {
        self.dim()
    }

    #[inline]
    pub fn dim4(&self) -> (usize, usize, usize, usize) {
        self.dim()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        Some(&mut self.data)
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    #[inline]
    pub fn into_raw_vec(self) -> Vec<T> {
        self.data
    }

    #[inline]
    pub fn into_raw_vec_and_offset(self) -> (Vec<T>, Option<usize>) {
        (self.data, Some(0))
    }

    #[inline]
    pub fn insert_axis(mut self, axis: impl Into<Axis>) -> Self {
        let ax = axis.into().0;
        let insert_idx = ax.min(self.shape.len());
        self.shape.insert(insert_idx, 1);
        self
    }

    #[inline]
    pub fn into_dyn(self) -> Self {
        self
    }

    #[inline]
    pub fn into_dimensionality(self) -> Result<Self, CoreError> {
        Ok(self)
    }

    #[inline]
    pub fn view(&self) -> &Self {
        self
    }

    pub fn reshape(mut self, new_shape: impl IntoShape) -> Result<Self, CoreError> {
        let s = new_shape.into_shape();
        let expected = s.iter().product::<usize>();
        if expected != self.data.len() {
            return Err(CoreError::InvalidShape(format!(
                "cannot reshape tensor of len {} to shape {:?}",
                self.data.len(),
                s
            )));
        }
        self.shape = s;
        Ok(self)
    }

    /// Iterate over rows along the 0th axis
    pub fn outer_iter(&self) -> impl Iterator<Item = &[T]> {
        let step = if self.shape.len() >= 2 {
            self.shape[1..].iter().product::<usize>().max(1)
        } else {
            1
        };
        self.data.chunks_exact(step)
    }

    /// Get slice for a specific row index along 0th axis
    pub fn row(&self, idx: usize) -> &[T] {
        let step = if self.shape.len() >= 2 {
            self.shape[1..].iter().product::<usize>().max(1)
        } else {
            1
        };
        let start = idx * step;
        &self.data[start..start + step]
    }

    #[inline]
    pub fn get<I: TensorIndex>(&self, idx: I) -> Option<&T> {
        idx.offset(&self.shape).and_then(|off| self.data.get(off))
    }

    #[inline]
    pub fn get_mut<I: TensorIndex>(&mut self, idx: I) -> Option<&mut T> {
        idx.offset(&self.shape).and_then(|off| self.data.get_mut(off))
    }
}

impl<T: Copy> Tensor<T> {
    /// Create tensor by copying slice data
    pub fn from_slice(shape: impl IntoShape, slice: &[T]) -> Self {
        Self {
            shape: shape.into_shape(),
            data: slice.to_vec(),
        }
    }
}

impl<T> Index<[usize; 4]> for Tensor<T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: [usize; 4]) -> &Self::Output {
        let offset =
            ((idx[0] * self.shape[1] + idx[1]) * self.shape[2] + idx[2]) * self.shape[3] + idx[3];
        &self.data[offset]
    }
}

impl<T> IndexMut<[usize; 4]> for Tensor<T> {
    #[inline]
    fn index_mut(&mut self, idx: [usize; 4]) -> &mut Self::Output {
        let offset =
            ((idx[0] * self.shape[1] + idx[1]) * self.shape[2] + idx[2]) * self.shape[3] + idx[3];
        &mut self.data[offset]
    }
}

impl<T> Index<[usize; 3]> for Tensor<T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: [usize; 3]) -> &Self::Output {
        let offset = (idx[0] * self.shape[1] + idx[1]) * self.shape[2] + idx[2];
        &self.data[offset]
    }
}

impl<T> IndexMut<[usize; 3]> for Tensor<T> {
    #[inline]
    fn index_mut(&mut self, idx: [usize; 3]) -> &mut Self::Output {
        let offset = (idx[0] * self.shape[1] + idx[1]) * self.shape[2] + idx[2];
        &mut self.data[offset]
    }
}

impl<T> Index<[usize; 2]> for Tensor<T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: [usize; 2]) -> &Self::Output {
        let offset = idx[0] * self.shape[1] + idx[1];
        &self.data[offset]
    }
}

impl<T> IndexMut<[usize; 2]> for Tensor<T> {
    #[inline]
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output {
        let offset = idx[0] * self.shape[1] + idx[1];
        &mut self.data[offset]
    }
}

impl<T> Index<[usize; 1]> for Tensor<T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: [usize; 1]) -> &Self::Output {
        &self.data[idx[0]]
    }
}

impl<T> IndexMut<[usize; 1]> for Tensor<T> {
    #[inline]
    fn index_mut(&mut self, idx: [usize; 1]) -> &mut Self::Output {
        &mut self.data[idx[0]]
    }
}

impl<T> Index<usize> for Tensor<T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}

impl<T> IndexMut<usize> for Tensor<T> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[idx]
    }
}

pub type Array4<T = f32> = Tensor<T>;
pub type Array3<T = f32> = Tensor<T>;
pub type Array2<T = f32> = Tensor<T>;
pub type ArrayD<T = f32> = Tensor<T>;
