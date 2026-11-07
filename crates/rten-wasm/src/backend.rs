use std::sync::Arc;

#[cfg(feature = "rusto-rten")]
use rusto_rten::{AsView, Layout};

pub struct WasmBackendModel {
    #[cfg(feature = "rusto-rten")]
    model: Arc<rusto_rten::Model>,
    #[cfg(not(feature = "rusto-rten"))]
    model: Arc<rusto_nn::Model>,
    #[cfg(not(feature = "rusto-rten"))]
    session: rusto_nn::Session<'static>,
}

impl WasmBackendModel {
    #[cfg(feature = "rusto-rten")]
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, String> {
        let mut model_opts = rusto_rten::ModelOptions::with_all_ops();
        model_opts.prepack_weights(true);
        let model = Arc::new(
            model_opts
                .load(data)
                .map_err(|e| format!("RTen model load error: {:?}", e))?,
        );
        Ok(Self { model })
    }

    #[cfg(not(feature = "rusto-rten"))]
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, String> {
        let model = Arc::new(
            rusto_nn::Model::from_bytes(data)
                .map_err(|e| format!("Model load error: {:?}", e))?,
        );
        let session = rusto_nn::Session::from_arc(Arc::clone(&model))
            .map_err(|e| format!("Session error: {:?}", e))?;
        Ok(Self { model, session })
    }

    #[cfg(feature = "rusto-rten")]
    pub fn input_channels(&self) -> usize {
        self.model
            .input_shape(0)
            .and_then(|dims| dims.get(1).cloned())
            .map(|d| match d {
                rusto_rten::Dimension::Fixed(c) => c,
                _ => 3,
            })
            .unwrap_or(3)
    }

    #[cfg(not(feature = "rusto-rten"))]
    pub fn input_channels(&self) -> usize {
        self.model.input_channels()
    }

    #[allow(dead_code)]
    #[cfg(feature = "rusto-rten")]
    pub fn input_names(&self) -> Vec<String> {
        self.model
            .input_ids()
            .iter()
            .filter_map(|&id| {
                self.model
                    .node_info(id)
                    .and_then(|info| info.name().map(|s| s.to_string()))
            })
            .collect()
    }

    #[allow(dead_code)]
    #[cfg(not(feature = "rusto-rten"))]
    pub fn input_names(&self) -> Vec<String> {
        self.model.input_names().iter().map(|s| s.to_string()).collect()
    }

    #[allow(dead_code)]
    #[cfg(feature = "rusto-rten")]
    pub fn output_names(&self) -> Vec<String> {
        self.model
            .output_ids()
            .iter()
            .filter_map(|&id| {
                self.model
                    .node_info(id)
                    .and_then(|info| info.name().map(|s| s.to_string()))
            })
            .collect()
    }

    #[allow(dead_code)]
    #[cfg(not(feature = "rusto-rten"))]
    pub fn output_names(&self) -> Vec<String> {
        self.model.output_names().iter().map(|s| s.to_string()).collect()
    }

    #[cfg(feature = "rusto-rten")]
    pub fn find_node(&self, name: &str) -> Option<u32> {
        self.model.find_node(name).map(|id| id.as_u32())
    }

    #[cfg(not(feature = "rusto-rten"))]
    pub fn find_node(&self, name: &str) -> Option<u32> {
        self.model
            .tensor_names()
            .iter()
            .position(|n| n == name)
            .map(|id| id as u32)
    }

    #[cfg(feature = "rusto-rten")]
    pub fn input_ids(&self) -> Vec<u32> {
        self.model.input_ids().iter().map(|id| id.as_u32()).collect()
    }

    #[cfg(not(feature = "rusto-rten"))]
    pub fn input_ids(&self) -> Vec<u32> {
        self.model
            .input_names()
            .iter()
            .filter_map(|name| {
                self.model
                    .tensor_names()
                    .iter()
                    .position(|n| n == name)
                    .map(|id| id as u32)
            })
            .collect()
    }

    #[cfg(feature = "rusto-rten")]
    pub fn output_ids(&self) -> Vec<u32> {
        self.model.output_ids().iter().map(|id| id.as_u32()).collect()
    }

    #[cfg(not(feature = "rusto-rten"))]
    pub fn output_ids(&self) -> Vec<u32> {
        self.model
            .output_names()
            .iter()
            .filter_map(|name| {
                self.model
                    .tensor_names()
                    .iter()
                    .position(|n| n == name)
                    .map(|id| id as u32)
            })
            .collect()
    }

    #[cfg(feature = "rusto-rten")]
    pub fn run_single(
        &self,
        shape: &[usize],
        data: &[f32],
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let in_id = self
            .model
            .input_ids()
            .first()
            .copied()
            .ok_or_else(|| "Model has no inputs".to_string())?;
        let tensor = rusto_rten::RtenTensor::<f32>::from_data(shape, data.to_vec());
        let outputs = self
            .model
            .run(
                vec![(in_id, tensor.view().into())],
                self.model.output_ids(),
                None,
            )
            .map_err(|e| format!("RTen inference error: {:?}", e))?;
        let out_val = outputs
            .into_iter()
            .next()
            .ok_or_else(|| "Model produced no outputs".to_string())?;
        match out_val {
            rusto_rten::Value::FloatTensor(t) => Ok((t.shape().to_vec(), t.to_vec())),
            _ => Err("Expected FloatTensor output".to_string()),
        }
    }

    #[cfg(not(feature = "rusto-rten"))]
    pub fn run_single(
        &self,
        shape: &[usize],
        data: &[f32],
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let tensor =
            rusto_nn::Tensor::from_floats(shape, data.to_vec(), rusto_nn::TensorLayout::NCHW);
        let out = self
            .session
            .run_single(&tensor)
            .map_err(|e| format!("Inference error: {:?}", e))?;
        Ok((out.shape().to_vec(), out.to_f32_vec()))
    }

    #[cfg(feature = "rusto-rten")]
    pub fn run_with_ids(
        &self,
        input_id: u32,
        input_shape: &[usize],
        input_data: &[f32],
        output_id: u32,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let in_id = rusto_rten::NodeId::from_u32(input_id);
        let out_id = rusto_rten::NodeId::from_u32(output_id);
        let tensor = rusto_rten::RtenTensor::<f32>::from_data(input_shape, input_data.to_vec());
        let outputs = self
            .model
            .run(vec![(in_id, tensor.view().into())], &[out_id], None)
            .map_err(|e| format!("RTen inference error: {:?}", e))?;
        let out_val = outputs
            .into_iter()
            .next()
            .ok_or_else(|| "Model produced no outputs".to_string())?;
        match out_val {
            rusto_rten::Value::FloatTensor(t) => Ok((t.shape().to_vec(), t.to_vec())),
            _ => Err("Expected FloatTensor output".to_string()),
        }
    }

    #[cfg(not(feature = "rusto-rten"))]
    pub fn run_with_ids(
        &self,
        input_id: u32,
        input_shape: &[usize],
        input_data: &[f32],
        output_id: u32,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let in_name = self
            .model
            .tensor_names()
            .get(input_id as usize)
            .cloned()
            .or_else(|| self.model.input_names().first().cloned())
            .ok_or_else(|| "Input tensor not found".to_string())?;
        let out_name = self
            .model
            .tensor_names()
            .get(output_id as usize)
            .cloned()
            .or_else(|| self.model.output_names().first().cloned())
            .ok_or_else(|| "Output tensor not found".to_string())?;

        let tensor = rusto_nn::Tensor::from_floats(
            input_shape,
            input_data.to_vec(),
            rusto_nn::TensorLayout::NCHW,
        );
        let outputs = self
            .session
            .run(&[(&in_name, &tensor)])
            .map_err(|e| e.to_string())?;
        let out = outputs
            .get(&out_name)
            .or_else(|| outputs.values().next())
            .ok_or_else(|| format!("Output tensor '{}' not produced", out_name))?;
        Ok((out.shape().to_vec(), out.to_f32_vec()))
    }

    #[cfg(feature = "rusto-rten")]
    pub fn run_order(
        &self,
        num_items: usize,
        feats_data: Vec<f32>,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let feats_id = self
            .model
            .find_node("feats")
            .or_else(|| self.model.input_ids().first().copied())
            .ok_or_else(|| "Order model feats input not found".to_string())?;
        let mask_id = self
            .model
            .find_node("mask")
            .or_else(|| self.model.input_ids().get(1).copied())
            .ok_or_else(|| "Order model mask input not found".to_string())?;

        let feats_tensor =
            rusto_rten::RtenTensor::<f32>::from_data(&[1, num_items, 16], feats_data);
        let mask_tensor =
            rusto_rten::RtenTensor::<i32>::from_data(&[1, num_items], vec![1i32; num_items]);

        let outputs = self
            .model
            .run(
                vec![
                    (feats_id, feats_tensor.view().into()),
                    (mask_id, mask_tensor.view().into()),
                ],
                self.model.output_ids(),
                None,
            )
            .map_err(|e| format!("Order model run error: {:?}", e))?;

        let out_val = outputs
            .into_iter()
            .next()
            .ok_or_else(|| "Expected float tensor output from order model".to_string())?;

        match out_val {
            rusto_rten::Value::FloatTensor(t) => Ok((t.shape().to_vec(), t.to_vec())),
            _ => Err("Expected float tensor output from order model".to_string()),
        }
    }

    #[cfg(not(feature = "rusto-rten"))]
    pub fn run_order(
        &self,
        num_items: usize,
        feats_data: Vec<f32>,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let in_names = self.model.input_names();
        let feats_name = if in_names.iter().any(|n| n == "feats") {
            "feats"
        } else {
            in_names.first().map(|s| s.as_str()).unwrap_or("feats")
        };
        let mask_name = if in_names.iter().any(|n| n == "mask") {
            "mask"
        } else {
            in_names.get(1).map(|s| s.as_str()).unwrap_or("mask")
        };

        let feats_tensor = rusto_nn::Tensor::from_floats(
            &[1, num_items, 16],
            feats_data,
            rusto_nn::TensorLayout::NCHW,
        );
        let mask_tensor = rusto_nn::Tensor::from_ints(&[1, num_items], vec![1i32; num_items]);

        let results = self
            .session
            .run(&[(feats_name, &feats_tensor), (mask_name, &mask_tensor)])
            .map_err(|e| format!("Order model run error: {:?}", e))?;

        let out = results
            .get("a")
            .or_else(|| results.get("logits"))
            .or_else(|| results.values().next())
            .ok_or_else(|| "Expected float tensor output from order model".to_string())?;

        Ok((out.shape().to_vec(), out.to_f32_vec()))
    }

    #[cfg(feature = "rusto-rten")]
    pub fn run_linenet_rec(
        &self,
        b: usize,
        rec_h: usize,
        max_w: usize,
        batch_data: Vec<f32>,
        widths: Vec<i32>,
    ) -> Result<(Vec<i32>, Vec<f32>, Vec<i32>, usize), String> {
        let in_ids = self.model.input_ids();
        if in_ids.len() < 2 {
            return Err("LineNet rec model expects at least 2 inputs".to_string());
        }
        let x_tensor =
            rusto_rten::RtenTensor::<f32>::from_data(&[b, 1, rec_h, max_w], batch_data);
        let widths_tensor = rusto_rten::RtenTensor::<i32>::from_data(&[b], widths);
        let outputs = self
            .model
            .run(
                vec![
                    (in_ids[0], x_tensor.view().into()),
                    (in_ids[1], widths_tensor.view().into()),
                ],
                self.model.output_ids(),
                None,
            )
            .map_err(|e| format!("Rec inference error: {:?}", e))?;

        if outputs.len() < 3 {
            return Err("LineNet rec produced fewer than 3 outputs".to_string());
        }

        let (ids_vec, ids_shape) = match &outputs[0] {
            rusto_rten::Value::Int32Tensor(t) => (t.to_vec(), t.shape().to_vec()),
            _ => return Err("Expected Int32Tensor for ids".to_string()),
        };
        let conf_vec = match &outputs[1] {
            rusto_rten::Value::FloatTensor(t) => t.to_vec(),
            _ => return Err("Expected FloatTensor for conf".to_string()),
        };
        let lengths_vec = match &outputs[2] {
            rusto_rten::Value::Int32Tensor(t) => t.to_vec(),
            _ => return Err("Expected Int32Tensor for lengths".to_string()),
        };

        let max_len = if ids_shape.len() >= 2 {
            ids_shape[1]
        } else {
            ids_vec.len() / b.max(1)
        };

        Ok((ids_vec, conf_vec, lengths_vec, max_len))
    }

    #[cfg(not(feature = "rusto-rten"))]
    pub fn run_linenet_rec(
        &self,
        b: usize,
        rec_h: usize,
        max_w: usize,
        batch_data: Vec<f32>,
        widths: Vec<i32>,
    ) -> Result<(Vec<i32>, Vec<f32>, Vec<i32>, usize), String> {
        let in_names = self.model.input_names();
        let in_name0 = in_names.first().map(|s| s.as_str()).unwrap_or("x");
        let in_name1 = in_names.get(1).map(|s| s.as_str()).unwrap_or("widths");
        let wanted_outputs: Vec<&str> = self
            .model
            .output_names()
            .iter()
            .map(|s| s.as_str())
            .filter(|&n| matches!(n, "ids" | "conf" | "len" | "lengths"))
            .collect();

        let x_tensor = rusto_nn::Tensor::from_floats(
            &[b, 1, rec_h, max_w],
            batch_data,
            rusto_nn::TensorLayout::NCHW,
        );
        let widths_tensor = rusto_nn::Tensor::from_ints(&[b], widths);

        let rec_outputs = if wanted_outputs.len() >= 2 {
            self.session
                .run_outputs(
                    &[(in_name0, &x_tensor), (in_name1, &widths_tensor)],
                    &wanted_outputs,
                )
                .map_err(|e| format!("Rec inference error: {:?}", e))?
        } else {
            self.session
                .run(&[(in_name0, &x_tensor), (in_name1, &widths_tensor)])
                .map_err(|e| format!("Rec inference error: {:?}", e))?
        };

        let ids_tensor = rec_outputs
            .get("ids")
            .or_else(|| rec_outputs.values().next())
            .ok_or_else(|| "Rec output ids not found".to_string())?;
        let conf_tensor = rec_outputs
            .get("conf")
            .or_else(|| rec_outputs.values().nth(1))
            .ok_or_else(|| "Rec output conf not found".to_string())?;
        let len_tensor = rec_outputs
            .get("len")
            .or_else(|| rec_outputs.get("lengths"))
            .or_else(|| rec_outputs.values().nth(2));

        let ids_vec = ids_tensor.to_i32_vec();
        let conf_vec = conf_tensor.to_f32_vec();
        let lens_vec = len_tensor.map(|t| t.to_i32_vec()).unwrap_or_default();

        let ids_shape = ids_tensor.shape();
        let max_len = if ids_shape.len() >= 2 {
            ids_shape[1]
        } else {
            ids_vec.len() / b.max(1)
        };

        Ok((ids_vec, conf_vec, lens_vec, max_len))
    }
}
