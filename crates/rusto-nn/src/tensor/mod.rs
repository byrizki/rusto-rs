pub mod layout;
pub mod shape;

pub use layout::*;
pub use shape::*;

use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Float,
    Int32,
    UInt8,
    Int8,
}

#[derive(Clone)]
pub enum TensorData {
    Float(Vec<f32>),
    FloatShared(Arc<Vec<f32>>),
    Int32(Vec<i32>),
    Int32Shared(Arc<Vec<i32>>),
}

/// Contiguous multidimensional tensor container
#[derive(Clone)]
pub struct Tensor {
    data: TensorData,
    shape: Shape,
    layout: TensorLayout,
}

impl Tensor {
    /// Create new zero-initialized float tensor
    pub fn zeros(shape: &[usize], layout: TensorLayout) -> Self {
        let count = match layout {
            TensorLayout::NCHW | TensorLayout::NHWC => element_count(shape),
            TensorLayout::NC8HW8 => {
                if shape.len() >= 4 {
                    shape[0] * shape[1].div_ceil(8) * shape[2] * shape[3] * 8
                } else {
                    element_count(shape)
                }
            }
            TensorLayout::NC4HW4 => {
                if shape.len() >= 4 {
                    shape[0] * shape[1].div_ceil(4) * shape[2] * shape[3] * 4
                } else {
                    element_count(shape)
                }
            }
        };

        Self {
            data: TensorData::Float(vec![0.0; count]),
            shape: shape.iter().copied().collect(),
            layout,
        }
    }

    /// Create float tensor from raw vector data
    pub fn from_data(shape: &[usize], data: Vec<f32>, layout: TensorLayout) -> Self {
        Self {
            data: TensorData::Float(data),
            shape: shape.iter().copied().collect(),
            layout,
        }
    }

    /// Create float tensor from raw vector data
    pub fn from_floats(shape: &[usize], data: Vec<f32>, layout: TensorLayout) -> Self {
        Self::from_data(shape, data, layout)
    }

    /// Create int32 tensor from raw vector data
    pub fn from_ints(shape: &[usize], data: Vec<i32>) -> Self {
        Self {
            data: TensorData::Int32(data),
            shape: shape.iter().copied().collect(),
            layout: TensorLayout::NCHW,
        }
    }

    /// Extract underlying owned float buffer if not shared
    pub fn into_owned_floats(self) -> Option<Vec<f32>> {
        match self.data {
            TensorData::Float(v) => Some(v),
            _ => None,
        }
    }

    /// Create a zero-copy shared copy of this tensor without duplicating buffer data
    pub fn share(&mut self) -> Self {
        match &mut self.data {
            TensorData::Float(v) => {
                let arc = Arc::new(std::mem::take(v));
                self.data = TensorData::FloatShared(Arc::clone(&arc));
                Self {
                    data: TensorData::FloatShared(arc),
                    shape: self.shape.clone(),
                    layout: self.layout,
                }
            }
            TensorData::FloatShared(arc) => Self {
                data: TensorData::FloatShared(Arc::clone(arc)),
                shape: self.shape.clone(),
                layout: self.layout,
            },
            TensorData::Int32(v) => {
                let arc = Arc::new(std::mem::take(v));
                self.data = TensorData::Int32Shared(Arc::clone(&arc));
                Self {
                    data: TensorData::Int32Shared(arc),
                    shape: self.shape.clone(),
                    layout: self.layout,
                }
            }
            TensorData::Int32Shared(arc) => Self {
                data: TensorData::Int32Shared(Arc::clone(arc)),
                shape: self.shape.clone(),
                layout: self.layout,
            },
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn set_shape(&mut self, shape: &[usize]) {
        self.shape = shape.iter().copied().collect();
    }

    pub fn layout(&self) -> TensorLayout {
        self.layout
    }

    pub fn set_layout(&mut self, layout: TensorLayout) {
        self.layout = layout;
    }

    pub fn dtype(&self) -> DataType {
        match &self.data {
            TensorData::Float(_) | TensorData::FloatShared(_) => DataType::Float,
            TensorData::Int32(_) | TensorData::Int32Shared(_) => DataType::Int32,
        }
    }

    pub fn as_floats(&self) -> Option<&[f32]> {
        match &self.data {
            TensorData::Float(v) => Some(v.as_slice()),
            TensorData::FloatShared(arc) => Some(arc.as_slice()),
            _ => None,
        }
    }

    pub fn as_floats_mut(&mut self) -> Option<&mut [f32]> {
        match &mut self.data {
            TensorData::Float(v) => Some(v.as_mut_slice()),
            TensorData::FloatShared(arc) => {
                let unique_vec = Arc::make_mut(arc);
                Some(unique_vec.as_mut_slice())
            }
            _ => None,
        }
    }

    pub fn as_ints(&self) -> Option<&[i32]> {
        match &self.data {
            TensorData::Int32(v) => Some(v.as_slice()),
            TensorData::Int32Shared(arc) => Some(arc.as_slice()),
            _ => None,
        }
    }

    pub fn as_ints_mut(&mut self) -> Option<&mut [i32]> {
        match &mut self.data {
            TensorData::Int32(v) => Some(v.as_mut_slice()),
            TensorData::Int32Shared(arc) => {
                let unique_vec = Arc::make_mut(arc);
                Some(unique_vec.as_mut_slice())
            }
            _ => None,
        }
    }

    pub fn raw_floats(&self) -> Option<&[f32]> {
        self.as_floats()
    }

    /// Extract continuous planar NCHW float buffer, unpacking if packed
    pub fn to_nchw_floats(&self) -> Vec<f32> {
        if self.shape.len() == 4 {
            let (b, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
            let total = b * c * h * w;
            match self.layout {
                TensorLayout::NC8HW8 => {
                    let raw = self.as_floats().unwrap_or(&[]);
                    let mut dst = vec![0.0f32; total];
                    layout::unpack_nc8hw8_to_nchw(raw, b, c, h, w, &mut dst);
                    dst
                }
                TensorLayout::NC4HW4 => {
                    let raw = self.as_floats().unwrap_or(&[]);
                    let mut dst = vec![0.0f32; total];
                    layout::unpack_nc4hw4_to_nchw(raw, b, c, h, w, &mut dst);
                    dst
                }
                _ => self.as_floats().map(|s| s.to_vec()).unwrap_or_else(|| {
                    self.as_ints().map(|s| s.iter().map(|&x| x as f32).collect()).unwrap_or_default()
                }),
            }
        } else {
            self.as_floats().map(|s| s.to_vec()).unwrap_or_else(|| {
                self.as_ints().map(|s| s.iter().map(|&x| x as f32).collect()).unwrap_or_default()
            })
        }
    }

    /// View or extract planar NCHW float slice without copying if already planar
    pub fn as_f32_cow(&self) -> std::borrow::Cow<'_, [f32]> {
        if self.layout == TensorLayout::NCHW {
            if let Some(s) = self.as_floats() {
                std::borrow::Cow::Borrowed(s)
            } else if let Some(ints) = self.as_ints() {
                std::borrow::Cow::Owned(ints.iter().map(|&x| x as f32).collect())
            } else {
                std::borrow::Cow::Owned(Vec::new())
            }
        } else {
            std::borrow::Cow::Owned(self.to_nchw_floats())
        }
    }

    /// Convert elements to f32 vector regardless of underlying storage, returning planar layout
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.to_nchw_floats()
    }

    /// Convert elements to i32 vector regardless of underlying storage
    pub fn to_i32_vec(&self) -> Vec<i32> {
        if let Some(ints) = self.as_ints() {
            ints.to_vec()
        } else if let Some(floats) = self.as_floats() {
            floats.iter().map(|&x| x.round() as i32).collect()
        } else {
            Vec::new()
        }
    }

    pub fn len(&self) -> usize {
        match &self.data {
            TensorData::Float(v) => v.len(),
            TensorData::FloatShared(arc) => arc.len(),
            TensorData::Int32(v) => v.len(),
            TensorData::Int32Shared(arc) => arc.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={:?}, layout={:?}, len={})",
            self.shape.as_slice(),
            self.dtype(),
            self.layout,
            self.len()
        )
    }
}
