struct GemmParams {
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
    act: u32,
    has_bias: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: GemmParams;
@group(0) @binding(1) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(2) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> matrix_c: array<f32>;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    var acc: f32 = 0.0;
    let num_tiles = (params.k + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_col;
        if (row < params.m && a_col < params.k) {
            tile_a[local_row][local_col] = matrix_a[row * params.k + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        let b_row = t * TILE_SIZE + local_row;
        if (b_row < params.k && col < params.n) {
            tile_b[local_row][local_col] = matrix_b[b_row * params.n + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        for (var p: u32 = 0u; p < TILE_SIZE; p = p + 1u) {
            acc = acc + tile_a[local_row][p] * tile_b[p][local_col];
        }

        workgroupBarrier();
    }

    if (row < params.m && col < params.n) {
        var val = params.alpha * acc;
        if (params.has_bias != 0u) {
            val = val + bias[row];
        }
        if (params.beta != 0.0) {
            let prev = matrix_c[row * params.n + col];
            val = val + params.beta * prev;
        }

        if (params.act == 1u) {
            val = max(0.0, val);
        } else if (params.act == 2u) {
            val = clamp(val, 0.0, 6.0);
        } else if (params.act == 3u) {
            let val_p3 = val + 3.0;
            let clamped = clamp(val_p3, 0.0, 6.0);
            val = val * (clamped / 6.0);
        }

        matrix_c[row * params.n + col] = val;
    }
}

