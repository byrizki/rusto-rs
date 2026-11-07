use std::collections::{HashMap, HashSet};
use smallvec::smallvec;

use crate::graph::ir::{
    BinaryParams, FusedActivation, Graph, Node, OpKind, UnaryParams,
};
use crate::kernels::binary::BinaryOpType;
use crate::kernels::unary::UnaryOpType;
use crate::tensor::{Tensor, TensorLayout};

/// Run load-time graph optimizations and compute tensor liveness metadata.
pub fn optimize_graph(graph: &mut Graph) {
    let name_to_idx: HashMap<&str, usize> = graph
        .tensor_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let graph_inputs: HashSet<usize> = graph
        .input_names
        .iter()
        .filter_map(|n| name_to_idx.get(n.as_str()).copied())
        .collect();

    let graph_outputs: HashSet<usize> = graph
        .output_names
        .iter()
        .filter_map(|n| name_to_idx.get(n.as_str()).copied())
        .collect();

    eliminate_identities(&mut graph.nodes, &graph_inputs, &graph_outputs);
    fuse_gelu_subgraphs(&mut graph.nodes, &graph_outputs);
    fold_conv_scale(&mut graph.nodes, &graph_inputs, &graph_outputs);
    fuse_post_activations(&mut graph.nodes, &graph_inputs, &graph_outputs);
    pretranspose_const_matmul_b(&mut graph.nodes, &graph_outputs);
    sweep_dead_nodes_and_share_consts(&mut graph.nodes, &graph_outputs);
    graph.last_use = compute_last_use(&graph.nodes, graph.tensor_names.len(), &graph_inputs, &graph_outputs);
}

fn build_def_use(
    nodes: &[Node],
    graph_outputs: &HashSet<usize>,
) -> (HashMap<usize, usize>, HashMap<usize, usize>, HashMap<usize, usize>) {
    let mut producer_of = HashMap::with_capacity(nodes.len());
    let mut def_count: HashMap<usize, usize> = HashMap::with_capacity(nodes.len());
    let mut use_count: HashMap<usize, usize> = HashMap::with_capacity(nodes.len());

    for &out_tid in graph_outputs {
        *use_count.entry(out_tid).or_insert(0) += 1;
    }

    for (idx, node) in nodes.iter().enumerate() {
        if matches!(node.op, OpKind::NoOp) {
            continue;
        }
        for &out_tid in &node.out_indices {
            producer_of.insert(out_tid, idx);
            *def_count.entry(out_tid).or_insert(0) += 1;
        }
        for &in_tid in &node.in_indices {
            *use_count.entry(in_tid).or_insert(0) += 1;
        }
    }

    (producer_of, def_count, use_count)
}

fn is_single_def(tid: usize, def_count: &HashMap<usize, usize>, graph_inputs: Option<&HashSet<usize>>) -> bool {
    let defs = def_count.get(&tid).copied().unwrap_or(0);
    if let Some(inputs) = graph_inputs {
        if inputs.contains(&tid) {
            return defs <= 1;
        }
    }
    defs == 1
}

fn resolve_alias(mut tid: usize, aliases: &HashMap<usize, usize>) -> usize {
    while let Some(&next) = aliases.get(&tid) {
        if next == tid {
            break;
        }
        tid = next;
    }
    tid
}

fn eliminate_identities(
    nodes: &mut Vec<Node>,
    graph_inputs: &HashSet<usize>,
    graph_outputs: &HashSet<usize>,
) {
    let (_, def_count_init, _) = build_def_use(nodes, graph_outputs);

    let mut aliases: HashMap<usize, usize> = HashMap::new();
    for node in nodes.iter_mut() {
        for inp in node.in_indices.iter_mut() {
            *inp = resolve_alias(*inp, &aliases);
        }
        if matches!(node.op, OpKind::ConvertTensor)
            && node.in_indices.len() == 1
            && node.out_indices.len() == 1
        {
            let src = node.in_indices[0];
            let dst = node.out_indices[0];
            if src != dst
                && !graph_outputs.contains(&dst)
                && is_single_def(src, &def_count_init, Some(graph_inputs))
                && is_single_def(dst, &def_count_init, None)
            {
                aliases.insert(dst, src);
                node.op = OpKind::NoOp;
                node.in_indices.clear();
                node.out_indices.clear();
            }
        }
    }

    let (producer_of, def_count, use_count) = build_def_use(nodes, graph_outputs);
    let mut producer_out_rewrites: Vec<(usize, usize, usize)> = Vec::new();
    for idx in 0..nodes.len() {
        if matches!(nodes[idx].op, OpKind::ConvertTensor)
            && nodes[idx].in_indices.len() == 1
            && nodes[idx].out_indices.len() == 1
        {
            let src = nodes[idx].in_indices[0];
            let dst = nodes[idx].out_indices[0];
            if src != dst
                && graph_outputs.contains(&dst)
                && !graph_outputs.contains(&src)
                && !graph_inputs.contains(&src)
                && is_single_def(src, &def_count, None)
                && is_single_def(dst, &def_count, None)
                && use_count.get(&src).copied().unwrap_or(0) == 1
            {
                if let Some(&prod_idx) = producer_of.get(&src) {
                    if !matches!(nodes[prod_idx].op, OpKind::Input | OpKind::NoOp) {
                        producer_out_rewrites.push((prod_idx, src, dst));
                        nodes[idx].op = OpKind::NoOp;
                        nodes[idx].in_indices.clear();
                        nodes[idx].out_indices.clear();
                    }
                }
            }
        }
    }

    for (prod_idx, old_out, new_out) in producer_out_rewrites {
        for out_slot in nodes[prod_idx].out_indices.iter_mut() {
            if *out_slot == old_out {
                *out_slot = new_out;
            }
        }
    }
}

fn scalar_const_val(tid: usize, nodes: &[Node], producer_of: &HashMap<usize, usize>, def_count: &HashMap<usize, usize>) -> Option<f32> {
    if def_count.get(&tid).copied().unwrap_or(0) != 1 {
        return None;
    }
    let &idx = producer_of.get(&tid)?;
    if let OpKind::Const(t) = &nodes[idx].op {
        let floats = t.as_floats()?;
        if floats.len() == 1 {
            return Some(floats[0]);
        }
    }
    None
}

fn approx_eq(val: f32, target: f32, tol: f32) -> bool {
    (val - target).abs() <= tol
}

fn is_plain_binary(node: &Node, expected: BinaryOpType) -> bool {
    matches!(
        &node.op,
        OpKind::Binary(BinaryParams { op_type, activation: FusedActivation::None })
        if *op_type == expected
    ) && node.in_indices.len() == 2
        && node.out_indices.len() == 1
}

fn fuse_gelu_subgraphs(nodes: &mut [Node], graph_outputs: &HashSet<usize>) {
    let (producer_of, def_count, use_count) = build_def_use(nodes, graph_outputs);

    let mut single_consumer: HashMap<usize, usize> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        if matches!(node.op, OpKind::NoOp) {
            continue;
        }
        for &in_tid in &node.in_indices {
            if def_count.get(&in_tid).copied().unwrap_or(0) == 1
                && use_count.get(&in_tid).copied().unwrap_or(0) == 1
            {
                single_consumer.insert(in_tid, idx);
            }
        }
    }

    let mut fusions: Vec<(usize, usize, [usize; 4])> = Vec::new();

    for erf_idx in 0..nodes.len() {
        let is_erf = matches!(
            &nodes[erf_idx].op,
            OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Erf,
                activation: FusedActivation::None,
                ..
            })
        ) && nodes[erf_idx].in_indices.len() == 1
            && nodes[erf_idx].out_indices.len() == 1;

        if !is_erf {
            continue;
        }

        let div_out = nodes[erf_idx].in_indices[0];
        let erf_out = nodes[erf_idx].out_indices[0];
        if def_count.get(&div_out).copied().unwrap_or(0) != 1
            || use_count.get(&div_out).copied().unwrap_or(0) != 1
        {
            continue;
        }

        let Some(&div_idx) = producer_of.get(&div_out) else { continue };
        let div_node = &nodes[div_idx];
        let x_tid = if is_plain_binary(div_node, BinaryOpType::Div) {
            let rhs = div_node.in_indices[1];
            if scalar_const_val(rhs, nodes, &producer_of, &def_count)
                .is_some_and(|v| approx_eq(v, std::f32::consts::SQRT_2, 1e-3))
            {
                Some(div_node.in_indices[0])
            } else {
                None
            }
        } else if is_plain_binary(div_node, BinaryOpType::Mul) {
            let (a, b) = (div_node.in_indices[0], div_node.in_indices[1]);
            if scalar_const_val(b, nodes, &producer_of, &def_count)
                .is_some_and(|v| approx_eq(v, std::f32::consts::FRAC_1_SQRT_2, 1e-3))
            {
                Some(a)
            } else if scalar_const_val(a, nodes, &producer_of, &def_count)
                .is_some_and(|v| approx_eq(v, std::f32::consts::FRAC_1_SQRT_2, 1e-3))
            {
                Some(b)
            } else {
                None
            }
        } else {
            None
        };
        let Some(x_tid) = x_tid else { continue };
        if def_count.get(&x_tid).copied().unwrap_or(0) > 1 {
            continue;
        }

        let Some(&add_idx) = single_consumer.get(&erf_out) else { continue };
        let add_node = &nodes[add_idx];
        if !is_plain_binary(add_node, BinaryOpType::Add) {
            continue;
        }
        let other_add = if add_node.in_indices[0] == erf_out {
            add_node.in_indices[1]
        } else if add_node.in_indices[1] == erf_out {
            add_node.in_indices[0]
        } else {
            continue;
        };
        if !scalar_const_val(other_add, nodes, &producer_of, &def_count)
            .is_some_and(|v| approx_eq(v, 1.0, 1e-4))
        {
            continue;
        }
        let add_out = add_node.out_indices[0];

        let Some(&m_after_add_idx) = single_consumer.get(&add_out) else { continue };
        let m_after_add = &nodes[m_after_add_idx];
        if !is_plain_binary(m_after_add, BinaryOpType::Mul) {
            continue;
        }
        let partner = if m_after_add.in_indices[0] == add_out {
            m_after_add.in_indices[1]
        } else {
            m_after_add.in_indices[0]
        };

        if partner == x_tid {
            let m1_out = m_after_add.out_indices[0];
            if let Some(&m2_idx) = single_consumer.get(&m1_out) {
                let m2 = &nodes[m2_idx];
                if is_plain_binary(m2, BinaryOpType::Mul) {
                    let c_half = if m2.in_indices[0] == m1_out { m2.in_indices[1] } else { m2.in_indices[0] };
                    if scalar_const_val(c_half, nodes, &producer_of, &def_count)
                        .is_some_and(|v| approx_eq(v, 0.5, 1e-4))
                    {
                        fusions.push((m2_idx, x_tid, [div_idx, erf_idx, add_idx, m_after_add_idx]));
                        continue;
                    }
                }
            }
        }

        if scalar_const_val(partner, nodes, &producer_of, &def_count)
            .is_some_and(|v| approx_eq(v, 0.5, 1e-4))
        {
            let m1_out = m_after_add.out_indices[0];
            if let Some(&m2_idx) = single_consumer.get(&m1_out) {
                let m2 = &nodes[m2_idx];
                if is_plain_binary(m2, BinaryOpType::Mul) {
                    let maybe_x = if m2.in_indices[0] == m1_out { m2.in_indices[1] } else { m2.in_indices[0] };
                    if maybe_x == x_tid {
                        fusions.push((m2_idx, x_tid, [div_idx, erf_idx, add_idx, m_after_add_idx]));
                        continue;
                    }
                }
            }
        }

        if def_count.get(&partner).copied().unwrap_or(0) == 1
            && use_count.get(&partner).copied().unwrap_or(0) == 1
        {
            if let Some(&m1_idx) = producer_of.get(&partner) {
                let m1 = &nodes[m1_idx];
                if is_plain_binary(m1, BinaryOpType::Mul) {
                    let (a, b) = (m1.in_indices[0], m1.in_indices[1]);
                    let matches_half_x = (a == x_tid
                        && scalar_const_val(b, nodes, &producer_of, &def_count)
                            .is_some_and(|v| approx_eq(v, 0.5, 1e-4)))
                        || (b == x_tid
                            && scalar_const_val(a, nodes, &producer_of, &def_count)
                                .is_some_and(|v| approx_eq(v, 0.5, 1e-4)));
                    if matches_half_x {
                        fusions.push((m_after_add_idx, x_tid, [div_idx, erf_idx, add_idx, m1_idx]));
                    }
                }
            }
        }
    }

    for (target_idx, x_tid, dead_indices) in fusions {
        nodes[target_idx].in_indices = smallvec![x_tid];
        nodes[target_idx].op = OpKind::Unary(UnaryParams {
            op_type: UnaryOpType::GeluStandard,
            activation: FusedActivation::None,
            table: None,
        });
        for dead in dead_indices {
            nodes[dead].op = OpKind::NoOp;
            nodes[dead].in_indices.clear();
            nodes[dead].out_indices.clear();
        }
    }
}

fn fold_conv_scale(
    nodes: &mut [Node],
    graph_inputs: &HashSet<usize>,
    graph_outputs: &HashSet<usize>,
) {
    let (mut producer_of, def_count, use_count) = build_def_use(nodes, graph_outputs);

    for scale_idx in 0..nodes.len() {
        if nodes[scale_idx].in_indices.len() != 1 || nodes[scale_idx].out_indices.len() != 1 {
            continue;
        }
        if !matches!(nodes[scale_idx].op, OpKind::Scale { .. }) {
            continue;
        }

        let mid_tid = nodes[scale_idx].in_indices[0];
        let final_out_tid = nodes[scale_idx].out_indices[0];
        if graph_inputs.contains(&mid_tid)
            || def_count.get(&mid_tid).copied().unwrap_or(0) != 1
            || use_count.get(&mid_tid).copied().unwrap_or(0) != 1
        {
            continue;
        }

        let Some(&prod_idx) = producer_of.get(&mid_tid) else { continue };
        if prod_idx >= scale_idx || nodes[prod_idx].out_indices.len() != 1 {
            continue;
        }

        let (left, right) = nodes.split_at_mut(scale_idx);
        let prod_node = &mut left[prod_idx];
        let scale_node = &mut right[0];

        let folded = match (&mut prod_node.op, &scale_node.op) {
            (OpKind::Conv(c), OpKind::Scale { scale, bias: scale_bias })
                if c.activation == FusedActivation::None
                    && c.out_channels > 0
                    && c.out_channels == scale.len()
                    && !c.weight.is_empty()
                    && c.weight.len() % c.out_channels == 0 =>
            {
                let elems_per_oc = c.weight.len() / c.out_channels;
                for oc in 0..c.out_channels {
                    let s = scale[oc];
                    for w in &mut c.weight[oc * elems_per_oc..(oc + 1) * elems_per_oc] {
                        *w *= s;
                    }
                }
                let mut new_bias = vec![0.0f32; c.out_channels];
                for oc in 0..c.out_channels {
                    let old_b = c.bias.as_ref().and_then(|bs| bs.get(oc).copied()).unwrap_or(0.0);
                    let sb = scale_bias.as_ref().and_then(|bs| bs.get(oc).copied()).unwrap_or(0.0);
                    new_bias[oc] = old_b * scale[oc] + sb;
                }
                c.bias = Some(new_bias);
                true
            }
            _ => false,
        };

        if folded {
            prod_node.out_indices[0] = final_out_tid;
            scale_node.op = OpKind::NoOp;
            scale_node.in_indices.clear();
            scale_node.out_indices.clear();
            producer_of.remove(&mid_tid);
            producer_of.insert(final_out_tid, prod_idx);
        }
    }
}

fn match_fusable_activation(op: &OpKind) -> Option<FusedActivation> {
    match op {
        OpKind::Relu { slope } if *slope == 0.0 => Some(FusedActivation::Relu),
        OpKind::Relu6 { min_val, max_val } if *min_val == 0.0 && *max_val == 6.0 => {
            Some(FusedActivation::Relu6)
        }
        OpKind::Unary(UnaryParams {
            op_type: UnaryOpType::HardSwish,
            activation: FusedActivation::None,
            ..
        }) => Some(FusedActivation::HardSwish),
        _ => None,
    }
}

fn fuse_post_activations(
    nodes: &mut [Node],
    graph_inputs: &HashSet<usize>,
    graph_outputs: &HashSet<usize>,
) {
    let (producer_of, def_count, use_count) = build_def_use(nodes, graph_outputs);

    for act_idx in 0..nodes.len() {
        let Some(fused_act) = match_fusable_activation(&nodes[act_idx].op) else {
            continue;
        };
        if nodes[act_idx].in_indices.len() != 1 || nodes[act_idx].out_indices.len() != 1 {
            continue;
        }

        let mid_tid = nodes[act_idx].in_indices[0];
        let final_out_tid = nodes[act_idx].out_indices[0];
        if graph_inputs.contains(&mid_tid)
            || def_count.get(&mid_tid).copied().unwrap_or(0) != 1
            || use_count.get(&mid_tid).copied().unwrap_or(0) != 1
        {
            continue;
        }

        let Some(&prod_idx) = producer_of.get(&mid_tid) else { continue };
        if prod_idx >= act_idx || nodes[prod_idx].out_indices.len() != 1 {
            continue;
        }

        let can_fuse = match &mut nodes[prod_idx].op {
            OpKind::Conv(c) if c.activation == FusedActivation::None => {
                c.activation = fused_act;
                true
            }
            OpKind::Deconv(d) if d.activation == FusedActivation::None => {
                d.activation = fused_act;
                true
            }
            OpKind::Binary(b) if b.activation == FusedActivation::None => {
                b.activation = fused_act;
                true
            }
            _ => false,
        };

        if can_fuse {
            nodes[prod_idx].out_indices[0] = final_out_tid;
            nodes[act_idx].op = OpKind::NoOp;
            nodes[act_idx].in_indices.clear();
            nodes[act_idx].out_indices.clear();
        }
    }
}

fn pretranspose_const_matmul_b(nodes: &mut [Node], graph_outputs: &HashSet<usize>) {
    let (producer_of, def_count, use_count) = build_def_use(nodes, graph_outputs);

    let mut transposes: Vec<(usize, usize)> = Vec::new();
    for (idx, node) in nodes.iter().enumerate() {
        if let OpKind::MatMul(p) = &node.op {
            if p.trans_b && node.in_indices.len() >= 2 {
                let b_tid = node.in_indices[1];
                if def_count.get(&b_tid).copied().unwrap_or(0) == 1
                    && use_count.get(&b_tid).copied().unwrap_or(0) == 1
                {
                    if let Some(&const_idx) = producer_of.get(&b_tid) {
                        if let OpKind::Const(t) = &nodes[const_idx].op {
                            if t.shape().len() == 2 && t.as_floats().is_some() {
                                transposes.push((idx, const_idx));
                            }
                        }
                    }
                }
            }
        }
    }

    for (mm_idx, const_idx) in transposes {
        let transposed = if let OpKind::Const(t) = &nodes[const_idx].op {
            let n = t.shape()[0];
            let k = t.shape()[1];
            let src = t.as_floats().unwrap();
            let mut dst = vec![0.0f32; k * n];
            for r in 0..k {
                for c in 0..n {
                    dst[r * n + c] = src[c * k + r];
                }
            }
            Some(Tensor::from_floats(&[k, n], dst, TensorLayout::NCHW))
        } else {
            None
        };

        if let Some(new_t) = transposed {
            nodes[const_idx].op = OpKind::Const(new_t);
            if let OpKind::MatMul(p) = &mut nodes[mm_idx].op {
                p.trans_b = false;
            }
        }
    }
}

fn sweep_dead_nodes_and_share_consts(nodes: &mut Vec<Node>, graph_outputs: &HashSet<usize>) {
    let (_, _, use_count) = build_def_use(nodes, graph_outputs);

    nodes.retain_mut(|node| {
        match &mut node.op {
            OpKind::NoOp => false,
            OpKind::Const(t) => {
                let alive = node
                    .out_indices
                    .iter()
                    .any(|tid| use_count.get(tid).copied().unwrap_or(0) > 0);
                if alive {
                    let owned = std::mem::replace(t, Tensor::zeros(&[0], TensorLayout::NCHW));
                    *t = owned.into_shared();
                }
                alive
            }
            _ => true,
        }
    });
}

fn compute_last_use(
    nodes: &[Node],
    num_tensors: usize,
    graph_inputs: &HashSet<usize>,
    graph_outputs: &HashSet<usize>,
) -> Vec<usize> {
    let mut last_use = vec![0usize; num_tensors];
    for (op_idx, node) in nodes.iter().enumerate() {
        for &out_tid in &node.out_indices {
            if out_tid < num_tensors && last_use[out_tid] < op_idx {
                last_use[out_tid] = op_idx;
            }
        }
        for &in_tid in &node.in_indices {
            if in_tid < num_tensors {
                last_use[in_tid] = op_idx;
            }
        }
    }
    for &tid in graph_inputs.iter().chain(graph_outputs.iter()) {
        if tid < num_tensors {
            last_use[tid] = usize::MAX;
        }
    }
    last_use
}

pub(crate) fn compute_pruned_plan(
    graph: &Graph,
    requested_outputs: &[usize],
) -> (Vec<bool>, Vec<usize>) {
    let num_tensors = graph.tensor_names.len();
    let num_nodes = graph.nodes.len();
    let mut needed = vec![false; num_tensors];
    for &tid in requested_outputs {
        if tid < num_tensors {
            needed[tid] = true;
        }
    }

    let mut active = vec![false; num_nodes];
    for (idx, node) in graph.nodes.iter().enumerate().rev() {
        if matches!(node.op, OpKind::Input | OpKind::NoOp) {
            continue;
        }
        if node
            .out_indices
            .iter()
            .any(|&tid| tid < num_tensors && needed[tid])
        {
            active[idx] = true;
            for &in_tid in &node.in_indices {
                if in_tid < num_tensors {
                    needed[in_tid] = true;
                }
            }
        }
    }

    let mut last_use = vec![0usize; num_tensors];
    for (op_idx, node) in graph.nodes.iter().enumerate() {
        if !active[op_idx] {
            continue;
        }
        for &out_tid in &node.out_indices {
            if out_tid < num_tensors && last_use[out_tid] < op_idx {
                last_use[out_tid] = op_idx;
            }
        }
        for &in_tid in &node.in_indices {
            if in_tid < num_tensors {
                last_use[in_tid] = op_idx;
            }
        }
    }

    let name_to_idx: HashMap<&str, usize> = graph
        .tensor_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    for input_name in &graph.input_names {
        if let Some(&tid) = name_to_idx.get(input_name.as_str()) {
            if tid < num_tensors {
                last_use[tid] = usize::MAX;
            }
        }
    }
    for &tid in requested_outputs {
        if tid < num_tensors {
            last_use[tid] = usize::MAX;
        }
    }

    (active, last_use)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::prepack::PrepackedConv;
    use crate::schema::PadMode;

    #[test]
    fn test_fold_conv_scale_and_fuse_activation() {
        let conv = PrepackedConv {
            weight: vec![1.0, -2.0, 0.5, 4.0],
            bias: Some(vec![0.5, -1.0]),
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            dilate_h: 1,
            dilate_w: 1,
            pad_mode: PadMode::VALID,
            pad_x: 0,
            pad_y: 0,
            pads: None,
            group: 1,
            out_channels: 2,
            activation: FusedActivation::None,
            is_depthwise: false,
        };

        let nodes = vec![
            Node {
                name: "conv".into(),
                in_indices: smallvec![0],
                out_indices: smallvec![1],
                op: OpKind::Conv(conv),
            },
            Node {
                name: "bn".into(),
                in_indices: smallvec![1],
                out_indices: smallvec![2],
                op: OpKind::Scale {
                    scale: vec![2.0, 0.5],
                    bias: Some(vec![1.0, 3.0]),
                },
            },
            Node {
                name: "act".into(),
                in_indices: smallvec![2],
                out_indices: smallvec![3],
                op: OpKind::Unary(UnaryParams {
                    op_type: UnaryOpType::HardSwish,
                    activation: FusedActivation::None,
                    table: None,
                }),
            },
            Node {
                name: "dead_tail".into(),
                in_indices: smallvec![3],
                out_indices: smallvec![4],
                op: OpKind::LogSoftmax { axis: -1 },
            },
        ];

        let mut graph = Graph::new(
            nodes,
            vec!["x".into(), "t1".into(), "t2".into(), "y".into(), "aux".into()],
            vec!["x".into()],
            vec!["y".into(), "aux".into()],
            2,
        );
        optimize_graph(&mut graph);

        assert_eq!(graph.nodes.len(), 2);
        match &graph.nodes[0].op {
            OpKind::Conv(c) => {
                assert_eq!(c.activation, FusedActivation::HardSwish);
                assert_eq!(c.weight, vec![2.0, -4.0, 0.25, 2.0]);
                assert_eq!(c.bias, Some(vec![2.0, 2.5]));
                assert_eq!(graph.nodes[0].out_indices.as_slice(), &[3]);
            }
            _ => panic!("expected folded Conv"),
        }

        let (active, pruned_last_use) = compute_pruned_plan(&graph, &[3]);
        assert_eq!(active, vec![true, false]);
        assert_eq!(pruned_last_use[3], usize::MAX);
    }
}
