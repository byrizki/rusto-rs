use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use crate::error::MnnError;
use crate::graph::{execute_node, GraphContext};
use crate::model::Model;
use crate::tensor::Tensor;

pub enum ModelHandle<'a> {
    Ref(&'a Model),
    Arc(Arc<Model>),
}

impl<'a> Deref for ModelHandle<'a> {
    type Target = Model;
    #[inline(always)]
    fn deref(&self) -> &Model {
        match self {
            Self::Ref(m) => m,
            Self::Arc(m) => m.as_ref(),
        }
    }
}

/// Inference Session executing forward pass for a loaded neural network model
pub struct Session<'a> {
    model: ModelHandle<'a>,
    input_map: HashMap<String, usize>,
    output_indices: Vec<(String, usize)>,
    ctx: RefCell<GraphContext>,
}

impl<'a> Session<'a> {
    /// Create a new inference session for the given model
    pub fn new(model: &'a Model) -> Result<Self, MnnError> {
        let tensor_names = model.tensor_names();
        let mut name_to_index: HashMap<&str, usize> = HashMap::with_capacity(tensor_names.len());
        for (i, name) in tensor_names.iter().enumerate() {
            name_to_index.insert(name.as_str(), i);
        }

        let mut input_map = HashMap::new();
        for name in model.input_names() {
            if let Some(&idx) = name_to_index.get(name.as_str()) {
                input_map.insert(name.clone(), idx);
            }
        }

        let mut output_indices = Vec::new();
        for name in model.output_names() {
            if let Some(&idx) = name_to_index.get(name.as_str()) {
                output_indices.push((name.clone(), idx));
            }
        }

        let ctx = RefCell::new(GraphContext::new(tensor_names.len(), tensor_names));

        Ok(Self {
            model: ModelHandle::Ref(model),
            input_map,
            output_indices,
            ctx,
        })
    }

    /// Create an owned session from an Arc<Model> with static lifetime
    pub fn from_arc(model: Arc<Model>) -> Result<Session<'static>, MnnError> {
        let tensor_names = model.tensor_names();
        let mut name_to_index: HashMap<&str, usize> = HashMap::with_capacity(tensor_names.len());
        for (i, name) in tensor_names.iter().enumerate() {
            name_to_index.insert(name.as_str(), i);
        }

        let mut input_map = HashMap::new();
        for name in model.input_names() {
            if let Some(&idx) = name_to_index.get(name.as_str()) {
                input_map.insert(name.clone(), idx);
            }
        }

        let mut output_indices = Vec::new();
        for name in model.output_names() {
            if let Some(&idx) = name_to_index.get(name.as_str()) {
                output_indices.push((name.clone(), idx));
            }
        }

        let ctx = RefCell::new(GraphContext::new(tensor_names.len(), tensor_names));

        Ok(Session {
            model: ModelHandle::Arc(model),
            input_map,
            output_indices,
            ctx,
        })
    }

    /// Execute forward inference with provided input tensors
    pub fn run(
        &self,
        inputs: &[(&str, &Tensor)],
    ) -> Result<HashMap<String, Tensor>, MnnError> {
        let mut ctx = self.ctx.borrow_mut();
        ctx.reset();

        // Bind input tensors
        for &(name, tensor) in inputs {
            let &idx = self.input_map.get(name).ok_or_else(|| {
                MnnError::ExecutionError(format!(
                    "Input name '{}' not found in model inputs {:?}",
                    name,
                    self.model.input_names()
                ))
            })?;
            ctx.set_tensor(idx, tensor.clone());
        }

        // Execute operations sequentially over Graph IR
        for (i, node) in self.model.graph().nodes.iter().enumerate() {
            if let Err(e) = execute_node(node, i, &mut *ctx) {
                return Err(MnnError::ExecutionError(format!(
                    "Op [{:>3}] {} ({}): {}",
                    i,
                    node.op.kind_name(),
                    node.name,
                    e
                )));
            }
        }

        // Collect outputs
        let mut outputs = HashMap::with_capacity(self.output_indices.len());
        for (name, idx) in &self.output_indices {
            let tensor = ctx.get_tensor(*idx)?.clone();
            outputs.insert(name.clone(), tensor);
        }

        Ok(outputs)
    }

    /// Retrieve an intermediate or output tensor by index from the last execution
    pub fn get_tensor(&self, idx: usize) -> Result<Tensor, MnnError> {
        let ctx = self.ctx.borrow();
        ctx.get_tensor(idx).cloned()
    }

    /// Profile forward pass measuring time spent per OpKind
    pub fn profile(
        &self,
        inputs: &[(&str, &Tensor)],
    ) -> Result<HashMap<String, (usize, std::time::Duration)>, MnnError> {
        let mut ctx = self.ctx.borrow_mut();
        ctx.reset();

        for &(name, tensor) in inputs {
            let &idx = self.input_map.get(name).ok_or_else(|| {
                MnnError::ExecutionError(format!("Input '{}' not found", name))
            })?;
            ctx.set_tensor(idx, tensor.clone());
        }

        let mut op_durations: HashMap<String, (usize, std::time::Duration)> = HashMap::new();
        for (i, node) in self.model.graph().nodes.iter().enumerate() {
            let type_name = node.op.kind_name().to_string();
            let t0 = std::time::Instant::now();
            execute_node(node, i, &mut *ctx).map_err(|e| {
                MnnError::ExecutionError(format!("Op [{}] {}: {}", i, node.op.kind_name(), e))
            })?;
            let dt = t0.elapsed();
            let entry = op_durations.entry(type_name).or_insert((0, std::time::Duration::ZERO));
            entry.0 += 1;
            entry.1 += dt;
        }

        Ok(op_durations)
    }

    /// Profile forward pass returning per-operation timings (index, type, name, duration)
    pub fn profile_ops(
        &self,
        inputs: &[(&str, &Tensor)],
    ) -> Result<Vec<(usize, String, String, std::time::Duration)>, MnnError> {
        let mut ctx = self.ctx.borrow_mut();
        ctx.reset();

        for &(name, tensor) in inputs {
            let &idx = self.input_map.get(name).ok_or_else(|| {
                MnnError::ExecutionError(format!("Input '{}' not found", name))
            })?;
            ctx.set_tensor(idx, tensor.clone());
        }

        let mut op_timings = Vec::new();
        for (i, node) in self.model.graph().nodes.iter().enumerate() {
            let type_name = node.op.kind_name().to_string();
            let op_name = node.name.clone();
            let t0 = std::time::Instant::now();
            execute_node(node, i, &mut *ctx).map_err(|e| {
                MnnError::ExecutionError(format!("Op [{}] {}: {}", i, node.op.kind_name(), e))
            })?;
            let dt = t0.elapsed();
            op_timings.push((i, type_name, op_name, dt));
        }

        Ok(op_timings)
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Primary single-tensor run helper
    pub fn run_single(
        &self,
        input_tensor: &Tensor,
    ) -> Result<Tensor, MnnError> {
        let input_names = self.model.input_names();
        if input_names.is_empty() {
            return Err(MnnError::ExecutionError("Model has no input tensors declared".into()));
        }

        let results = self.run(&[(&input_names[0], input_tensor)])?;
        let output_names = self.model.output_names();
        if output_names.is_empty() {
            return Err(MnnError::ExecutionError("Model has no output tensors declared".into()));
        }

        results
            .get(&output_names[0])
            .cloned()
            .ok_or_else(|| MnnError::ExecutionError(format!("Output '{}' not produced", output_names[0])))
    }
}
