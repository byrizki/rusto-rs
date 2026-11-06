use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array, ArrayD};
pub use rten::{Dimension, Model, ModelOptions, NodeId, RunOptions, ThreadPool, Value, ValueOrView};
pub use rten_tensor::AsView;
pub use rten_tensor::Layout;
pub use rten_tensor::Tensor as RtenTensor;
pub use rten_tensor::TensorView;

#[derive(thiserror::Error, Debug)]
pub enum RtenError {
    #[error("RTen load error: {0}")]
    Load(#[from] rten::LoadError),

    #[error("RTen run error: {0}")]
    Run(#[from] rten::RunError),

    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("Output error: {0}")]
    Output(String),

    #[error("Engine error: {0}")]
    Engine(String),
}

pub type LineNetRecOutput = (Vec<Vec<i32>>, Vec<Vec<f32>>);

pub fn convert_output_to_array(value: &Value) -> Result<ArrayD<f32>, RtenError> {
    match value {
        Value::FloatTensor(t) => {
            let shape = t.shape().to_vec();
            let data: Vec<f32> = if let Some(slice) = t.data() {
                slice.to_vec()
            } else {
                t.iter().copied().collect()
            };
            let array = Array::from_shape_vec(shape, data)?;
            Ok(array.into_dyn())
        }
        Value::Int32Tensor(t) => {
            let shape = t.shape().to_vec();
            let data: Vec<f32> = t.iter().map(|&x| x as f32).collect();
            let array = Array::from_shape_vec(shape, data)?;
            Ok(array.into_dyn())
        }
        Value::Int8Tensor(t) => {
            let shape = t.shape().to_vec();
            let data: Vec<f32> = t.iter().map(|&x| x as f32).collect();
            let array = Array::from_shape_vec(shape, data)?;
            Ok(array.into_dyn())
        }
        Value::UInt8Tensor(t) => {
            let shape = t.shape().to_vec();
            let data: Vec<f32> = t.iter().map(|&x| x as f32).collect();
            let array = Array::from_shape_vec(shape, data)?;
            Ok(array.into_dyn())
        }
        _ => Err(RtenError::Output("Unsupported output tensor type".into())),
    }
}

pub struct RtenSession {
    pub model: Model,
    thread_pool: Option<Arc<ThreadPool>>,
}

impl RtenSession {
    pub fn from_file(path: &Path, num_threads: usize) -> Result<Self, RtenError> {
        let mut model_opts = ModelOptions::with_all_ops();
        model_opts.prepack_weights(true);
        let model = model_opts.load_file(path)?;

        let thread_pool = if num_threads > 0 {
            Some(Arc::new(ThreadPool::with_num_threads(num_threads)))
        } else {
            None
        };

        Ok(Self {
            model,
            thread_pool,
        })
    }

    pub fn from_static_slice(data: &'static [u8], num_threads: usize) -> Result<Self, RtenError> {
        let mut model_opts = ModelOptions::with_all_ops();
        model_opts.prepack_weights(true);
        let model = model_opts.load_static_slice(data)?;

        let thread_pool = if num_threads > 0 {
            Some(Arc::new(ThreadPool::with_num_threads(num_threads)))
        } else {
            None
        };

        Ok(Self {
            model,
            thread_pool,
        })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    fn run_options(&self) -> Option<RunOptions> {
        let mut opts = RunOptions::default();
        if let Some(ref tp) = self.thread_pool {
            opts = opts.with_thread_pool(Some(tp.clone()));
        }
        Some(opts)
    }

    pub fn input_channels(&self) -> usize {
        self.model
            .input_shape(0)
            .and_then(|dims| dims.get(1).cloned())
            .map(|d| match d {
                Dimension::Fixed(c) => c,
                _ => 3,
            })
            .unwrap_or(3)
    }

    pub fn is_linenet(&self) -> bool {
        self.input_channels() == 1
    }

    pub fn run(&mut self, input: &ArrayD<f32>) -> Result<ArrayD<f32>, RtenError> {
        let shape: Vec<usize> = input.shape().to_vec();
        let in_ids = self.model.input_ids();
        if in_ids.is_empty() {
            return Err(RtenError::Output("Model has no inputs".into()));
        }
        let in_id = in_ids[0];
        let out_ids = self.model.output_ids();
        let run_opts = self.run_options();

        let outputs = if let Some(slice) = input.as_slice() {
            let view = TensorView::from_data(&shape, slice);
            self.model.run(vec![(in_id, view.into())], out_ids, run_opts)?
        } else {
            let data: Vec<f32> = input.iter().copied().collect();
            let tensor = RtenTensor::<f32>::from_data(&shape, data);
            self.model.run(vec![(in_id, tensor.view().into())], out_ids, run_opts)?
        };

        if outputs.is_empty() {
            return Err(RtenError::Output("Model produced no outputs".into()));
        }

        convert_output_to_array(&outputs[0])
    }

    pub fn run_all(&mut self, input: &ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, RtenError> {
        let shape: Vec<usize> = input.shape().to_vec();
        let in_ids = self.model.input_ids();
        if in_ids.is_empty() {
            return Err(RtenError::Output("Model has no inputs".into()));
        }
        let in_id = in_ids[0];
        let out_ids = self.model.output_ids();

        let outputs = if let Some(slice) = input.as_slice() {
            let view = TensorView::from_data(&shape, slice);
            self.model.run(vec![(in_id, view.into())], out_ids, None)?
        } else {
            let data: Vec<f32> = input.iter().copied().collect();
            let tensor = RtenTensor::<f32>::from_data(&shape, data);
            self.model.run(vec![(in_id, tensor.view().into())], out_ids, None)?
        };

        let mut results = HashMap::new();
        for (i, out_id) in out_ids.iter().enumerate() {
            let name = self
                .model
                .node_info(*out_id)
                .and_then(|info| info.name().map(|s| s.to_string()))
                .unwrap_or_else(|| format!("output_{}", i));
            if let Some(out_val) = outputs.get(i) {
                let array = convert_output_to_array(out_val)?;
                results.insert(name, array);
            }
        }

        Ok(results)
    }

    pub fn run_with_inputs(
        &mut self,
        inputs_map: &HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, RtenError> {
        let mut rten_tensors = Vec::new();

        let model_in_ids = self.model.input_ids();
        for (name, input) in inputs_map {
            let node_id = if let Some(id) = self.model.find_node(name) {
                id
            } else if model_in_ids.len() == 1 {
                if name == "image" || inputs_map.len() == 1 {
                    model_in_ids[0]
                } else {
                    continue;
                }
            } else {
                continue;
            };

            let shape: Vec<usize> = input.shape().to_vec();
            let data: Vec<f32> = if let Some(slice) = input.as_slice() {
                slice.to_vec()
            } else {
                input.iter().copied().collect()
            };
            rten_tensors.push((node_id, RtenTensor::<f32>::from_data(&shape, data)));
        }

        let input_pairs: Vec<(NodeId, ValueOrView)> = rten_tensors
            .iter()
            .map(|(node_id, tensor)| (*node_id, tensor.view().into()))
            .collect();

        let out_ids = self.model.output_ids();
        let run_opts = self.run_options();
        let outputs = self.model.run(input_pairs, out_ids, run_opts)?;

        let mut results = HashMap::new();
        for (i, out_id) in out_ids.iter().enumerate() {
            let name = self
                .model
                .node_info(*out_id)
                .and_then(|info| info.name().map(|s| s.to_string()))
                .unwrap_or_else(|| format!("output_{}", i));
            if let Some(out_val) = outputs.get(i) {
                let array = convert_output_to_array(out_val)?;
                results.insert(name, array);
            }
        }

        Ok(results)
    }

    pub fn run_linenet_rec(
        &mut self,
        x: &ndarray::Array4<f32>,
        widths: &[i32],
    ) -> Result<LineNetRecOutput, RtenError> {
        let in_ids = self.model.input_ids();
        if in_ids.len() < 2 {
            return Err(RtenError::Engine(
                "LineNet rec model expects at least 2 inputs (x and widths)".into(),
            ));
        }

        let x_shape: Vec<usize> = x.shape().to_vec();
        let x_data: Vec<f32> = if let Some(slice) = x.as_slice() {
            slice.to_vec()
        } else {
            x.iter().copied().collect()
        };
        let x_tensor = RtenTensor::<f32>::from_data(&x_shape, x_data);
        let widths_tensor = RtenTensor::<i32>::from_data(&[widths.len()], widths.to_vec());

        let out_ids = self.model.output_ids();
        let run_opts = self.run_options();
        let outputs = self.model.run(
            vec![
                (in_ids[0], x_tensor.view().into()),
                (in_ids[1], widths_tensor.view().into()),
            ],
            out_ids,
            run_opts,
        )?;

        if outputs.len() < 3 {
            return Err(RtenError::Output(
                "LineNet rec model produced fewer than 3 outputs".into(),
            ));
        }

        let (ids_vec, ids_shape) = match &outputs[0] {
            Value::Int32Tensor(t) => (t.to_vec(), t.shape().to_vec()),
            _ => return Err(RtenError::Output("Expected Int32Tensor for ids".into())),
        };
        let (conf_vec, _) = match &outputs[1] {
            Value::FloatTensor(t) => (t.to_vec(), t.shape().to_vec()),
            _ => return Err(RtenError::Output("Expected FloatTensor for conf".into())),
        };
        let lengths_vec = match &outputs[2] {
            Value::Int32Tensor(t) => t.to_vec(),
            _ => return Err(RtenError::Output("Expected Int32Tensor for lengths".into())),
        };

        let batch_size = lengths_vec.len();
        let max_len = if ids_shape.len() >= 2 {
            ids_shape[1]
        } else {
            ids_vec.len() / batch_size.max(1)
        };

        let mut batch_ids = Vec::with_capacity(batch_size);
        let mut batch_confs = Vec::with_capacity(batch_size);

        for (b, &len_val) in lengths_vec.iter().enumerate().take(batch_size) {
            let len = len_val.max(0) as usize;
            let start = b * max_len;
            let actual_len = len.min(max_len);
            let b_ids = ids_vec[start..start + actual_len].to_vec();
            let b_confs = conf_vec[start..start + actual_len].to_vec();
            batch_ids.push(b_ids);
            batch_confs.push(b_confs);
        }

        Ok((batch_ids, batch_confs))
    }

    pub fn get_character_list(&self, _key: &str) -> Option<Vec<String>> {
        None
    }

    pub fn have_key(&self, _key: &str) -> bool {
        false
    }
}

impl rusto_core::engine::InferenceSession for RtenSession {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, rusto_core::CoreError> {
        self.run(&input).map_err(|e| rusto_core::CoreError::Model(e.to_string()))
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, rusto_core::CoreError> {
        self.run_all(&input).map_err(|e| rusto_core::CoreError::Model(e.to_string()))
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, rusto_core::CoreError> {
        self.run_with_inputs(&inputs).map_err(|e| rusto_core::CoreError::Model(e.to_string()))
    }

    fn get_character_list(&self, key: &str) -> Option<Vec<String>> {
        self.get_character_list(key)
    }

    fn have_key(&self, key: &str) -> bool {
        self.have_key(key)
    }

    fn input_channels(&self) -> usize {
        self.input_channels()
    }

    fn run_linenet_rec(
        &mut self,
        x: ndarray::Array4<f32>,
        widths: Vec<i32>,
    ) -> Result<rusto_core::engine::LineNetRecOutput, rusto_core::CoreError> {
        self.run_linenet_rec(&x, &widths)
            .map_err(|e| rusto_core::CoreError::Model(e.to_string()))
    }
}

/// RTen session loader callback for `rusto_core::engine::register_session_loader`.
pub fn loader(
    path: &Path,
    cfg: &rusto_core::types::EngineConfig,
) -> Result<Box<dyn rusto_core::engine::InferenceSession>, rusto_core::error::CoreError> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("rten") || ext.eq_ignore_ascii_case("onnx") {
        let session = RtenSession::from_file(path, cfg.intra_op_num_threads as usize)
            .map_err(|e| rusto_core::error::CoreError::Model(e.to_string()))?;
        Ok(Box::new(session))
    } else {
        Err(rusto_core::error::CoreError::Model("UNSUPPORTED_FORMAT".into()))
    }
}

/// Register the RTen backend with `rusto-core`'s engine session loader.
pub fn register() {
    rusto_core::engine::register_session_loader(loader);
}
