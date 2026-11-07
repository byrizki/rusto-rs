use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use memmap2::Mmap;

use crate::error::MnnError;
use crate::graph::{load_mnn, Graph};
use crate::schema::{self, Net};

enum Storage {
    #[cfg(not(target_arch = "wasm32"))]
    Mmap(Mmap),
    Bytes(Vec<u8>),
}

impl Storage {
    fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mmap(m) => m.as_ref(),
            Self::Bytes(b) => b.as_slice(),
        }
    }
}

/// Loaded neural network model represented as an in-memory execution Graph
pub struct Model {
    storage: Option<Storage>,
    graph: Graph,
}

impl Model {
    /// Load model from file path using zero-copy memory mapping
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, MnnError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_storage(Storage::Mmap(mmap), true)
    }

    /// Load model from file path in WASM environment
    #[cfg(target_arch = "wasm32")]
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, MnnError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(bytes)
    }

    /// Load model from memory buffer without retaining raw duplicate bytes after graph creation
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, MnnError> {
        Self::from_storage(Storage::Bytes(bytes), false)
    }

    /// Load model from memory buffer while retaining raw bytes for inspectability (e.g. CLI dumpers)
    pub fn from_bytes_retained(bytes: Vec<u8>) -> Result<Self, MnnError> {
        Self::from_storage(Storage::Bytes(bytes), true)
    }

    fn from_storage(storage: Storage, retain_storage: bool) -> Result<Self, MnnError> {
        let slice = storage.as_slice();
        if slice.len() < 4 {
            return Err(MnnError::InvalidModel("Buffer too small for model header".into()));
        }

        // Format Auto-Detection
        let graph = if slice.starts_with(b"RTEN") {
            #[cfg(feature = "rten")]
            {
                crate::graph::load_rten(slice)?
            }
            #[cfg(not(feature = "rten"))]
            {
                return Err(MnnError::InvalidModel(
                    "RTen model format detected, but 'rten' feature is not enabled".into(),
                ));
            }
        } else if slice.first() == Some(&0x08) {
            #[cfg(feature = "onnx")]
            {
                crate::graph::load_onnx(slice)?
            }
            #[cfg(not(feature = "onnx"))]
            {
                return Err(MnnError::InvalidModel(
                    "ONNX model format detected, but 'onnx' feature is not enabled".into(),
                ));
            }
        } else {
            load_mnn(slice)?
        };

        let storage_opt = match &storage {
            #[cfg(not(target_arch = "wasm32"))]
            Storage::Mmap(_) => Some(storage),
            Storage::Bytes(_) => {
                if retain_storage {
                    Some(storage)
                } else {
                    None
                }
            }
        };

        Ok(Self {
            storage: storage_opt,
            graph,
        })
    }

    /// Access underlying Graph IR
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Access underlying FlatBuffers Net representation if loaded from MNN format
    pub fn net(&self) -> Net<'_> {
        if let Some(storage) = &self.storage {
            schema::parse_net(storage.as_slice())
        } else {
            schema::parse_net(&[])
        }
    }

    /// Number of tensors declared in the graph
    pub fn tensor_names(&self) -> &[String] {
        &self.graph.tensor_names
    }

    /// Graph input names
    pub fn input_names(&self) -> &[String] {
        &self.graph.input_names
    }

    /// Graph output names
    pub fn output_names(&self) -> &[String] {
        &self.graph.output_names
    }

    /// Pre-packed static op metadata and weights (deprecated, preserved for compatibility)
    pub fn prepacked_ops(&self) -> &[Option<crate::graph::PrepackedOp>] {
        &[]
    }

    /// Total number of operations in the net
    pub fn op_count(&self) -> usize {
        self.graph.nodes.len()
    }

    /// Primary input channel count (e.g. 1 for grayscale ink map, 3 for RGB)
    pub fn input_channels(&self) -> usize {
        self.graph.input_channels
    }
}
