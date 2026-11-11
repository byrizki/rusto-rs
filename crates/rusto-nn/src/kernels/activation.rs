/// Fused post-activation functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusedActivation {
    None,
    Relu,
    Relu6,
    Clip(f32, f32),
    Sigmoid,
    HardSigmoid { alpha: f32, beta: f32 },
    HardSwish,
    Gelu,
    Tanh,
}

#[inline(always)]
pub fn apply_activation_scalar(x: f32, act: FusedActivation) -> f32 {
    match act {
        FusedActivation::None => x,
        FusedActivation::Relu => x.max(0.0),
        FusedActivation::Relu6 => x.clamp(0.0, 6.0),
        FusedActivation::Clip(min, max) => x.clamp(min, max),
        FusedActivation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        FusedActivation::HardSigmoid { alpha, beta } => (x * alpha + beta).clamp(0.0, 1.0),
        FusedActivation::HardSwish => x * ((x + 3.0) / 6.0).clamp(0.0, 1.0),
        FusedActivation::Tanh => x.tanh(),
        FusedActivation::Gelu => {
            let x3 = x * x * x;
            let inner = 0.79788456 * (x + 0.044715 * x3);
            0.5 * x * (1.0 + inner.tanh())
        }
    }
}

#[inline(always)]
pub fn apply_activation_i32(x: i32, act: FusedActivation) -> Option<i32> {
    match act {
        FusedActivation::None => Some(x),
        FusedActivation::Relu => Some(x.max(0)),
        FusedActivation::Relu6 => Some(x.clamp(0, 6)),
        FusedActivation::Clip(min, max) => {
            let v = (x as f64).max(min as f64).min(max as f64);
            Some(v.clamp(i32::MIN as f64, i32::MAX as f64) as i32)
        }
        _ => None,
    }
}

/// Vectorized in-place elementwise activation
pub fn apply_activation_slice(slice: &mut [f32], act: FusedActivation) {
    if act == FusedActivation::None {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                apply_activation_avx2(slice, act);
                return;
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe {
            apply_activation_wasm_slice(slice, act);
            return;
        }
    }

    #[allow(unreachable_code)]
    for v in slice.iter_mut() {
        *v = apply_activation_scalar(*v, act);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_activation_avx2(slice: &mut [f32], act: FusedActivation) {
    use core::arch::x86_64::*;

    let len = slice.len();
    let chunks = len / 8;
    let ptr = slice.as_mut_ptr();

    match act {
        FusedActivation::Relu => {
            let zero = _mm256_setzero_ps();
            for i in 0..chunks {
                let v = _mm256_loadu_ps(ptr.add(i * 8));
                let res = _mm256_max_ps(zero, v);
                _mm256_storeu_ps(ptr.add(i * 8), res);
            }
        }
        FusedActivation::Relu6 => {
            let zero = _mm256_setzero_ps();
            let six = _mm256_set1_ps(6.0);
            for i in 0..chunks {
                let v = _mm256_loadu_ps(ptr.add(i * 8));
                let res = _mm256_min_ps(six, _mm256_max_ps(zero, v));
                _mm256_storeu_ps(ptr.add(i * 8), res);
            }
        }
        FusedActivation::Clip(min, max) => {
            let min_vec = _mm256_set1_ps(min);
            let max_vec = _mm256_set1_ps(max);
            for i in 0..chunks {
                let v = _mm256_loadu_ps(ptr.add(i * 8));
                let res = _mm256_min_ps(max_vec, _mm256_max_ps(min_vec, v));
                _mm256_storeu_ps(ptr.add(i * 8), res);
            }
        }
        FusedActivation::HardSigmoid { alpha, beta } => {
            let zero = _mm256_setzero_ps();
            let one = _mm256_set1_ps(1.0);
            let alpha_v = _mm256_set1_ps(alpha);
            let beta_v = _mm256_set1_ps(beta);
            for i in 0..chunks {
                let v = _mm256_loadu_ps(ptr.add(i * 8));
                let linear = _mm256_fmadd_ps(v, alpha_v, beta_v);
                let res = _mm256_min_ps(one, _mm256_max_ps(zero, linear));
                _mm256_storeu_ps(ptr.add(i * 8), res);
            }
        }
        FusedActivation::HardSwish => {
            let zero = _mm256_setzero_ps();
            let six = _mm256_set1_ps(6.0);
            let three = _mm256_set1_ps(3.0);
            let inv_six = _mm256_set1_ps(1.0 / 6.0);
            for i in 0..chunks {
                let v = _mm256_loadu_ps(ptr.add(i * 8));
                let v_p3 = _mm256_add_ps(v, three);
                let clamped = _mm256_min_ps(six, _mm256_max_ps(zero, v_p3));
                let res = _mm256_mul_ps(v, _mm256_mul_ps(clamped, inv_six));
                _mm256_storeu_ps(ptr.add(i * 8), res);
            }
        }
        _ => {
            for v in slice.iter_mut() {
                *v = apply_activation_scalar(*v, act);
            }
            return;
        }
    }

    for i in (chunks * 8)..len {
        slice[i] = apply_activation_scalar(slice[i], act);
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn apply_activation_wasm_slice(slice: &mut [f32], act: FusedActivation) {
    use core::arch::wasm32::*;

    let len = slice.len();
    let chunks = len / 4;
    let ptr = slice.as_mut_ptr();

    match act {
        FusedActivation::Relu => {
            let zero = f32x4_splat(0.0);
            for i in 0..chunks {
                let v = v128_load(ptr.add(i * 4) as *const v128);
                v128_store(ptr.add(i * 4) as *mut v128, f32x4_pmax(v, zero));
            }
        }
        FusedActivation::Relu6 => {
            let zero = f32x4_splat(0.0);
            let six = f32x4_splat(6.0);
            for i in 0..chunks {
                let v = v128_load(ptr.add(i * 4) as *const v128);
                v128_store(ptr.add(i * 4) as *mut v128, f32x4_pmin(f32x4_pmax(v, zero), six));
            }
        }
        FusedActivation::Clip(min, max) => {
            let min_v = f32x4_splat(min);
            let max_v = f32x4_splat(max);
            for i in 0..chunks {
                let v = v128_load(ptr.add(i * 4) as *const v128);
                v128_store(ptr.add(i * 4) as *mut v128, f32x4_pmin(f32x4_pmax(v, min_v), max_v));
            }
        }
        FusedActivation::HardSigmoid { alpha, beta } => {
            let zero = f32x4_splat(0.0);
            let one = f32x4_splat(1.0);
            let alpha_v = f32x4_splat(alpha);
            let beta_v = f32x4_splat(beta);
            for i in 0..chunks {
                let v = v128_load(ptr.add(i * 4) as *const v128);
                let linear = f32x4_add(f32x4_mul(v, alpha_v), beta_v);
                let res = f32x4_pmin(f32x4_pmax(linear, zero), one);
                v128_store(ptr.add(i * 4) as *mut v128, res);
            }
        }
        FusedActivation::HardSwish => {
            let zero = f32x4_splat(0.0);
            let six = f32x4_splat(6.0);
            let three = f32x4_splat(3.0);
            let inv_six = f32x4_splat(1.0 / 6.0);
            for i in 0..chunks {
                let v = v128_load(ptr.add(i * 4) as *const v128);
                let v_p3 = f32x4_add(v, three);
                let clamped = f32x4_pmin(f32x4_pmax(v_p3, zero), six);
                let res = f32x4_mul(v, f32x4_mul(clamped, inv_six));
                v128_store(ptr.add(i * 4) as *mut v128, res);
            }
        }
        _ => {
            for v in slice.iter_mut() {
                *v = apply_activation_scalar(*v, act);
            }
            return;
        }
    }

    for i in (chunks * 4)..len {
        slice[i] = apply_activation_scalar(slice[i], act);
    }
}
