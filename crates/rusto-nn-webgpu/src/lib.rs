use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use rusto_nn::kernels::activation::FusedActivation;
use rusto_nn::{DeviceBackend, MnnError};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct GemmParamsGpu {
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
    act: u32,
    has_bias: u32,
    _pad: u32,
}

struct CachedBuffer {
    buffer: wgpu::Buffer,
    capacity: usize,
}

impl CachedBuffer {
    fn new(device: &wgpu::Device, label: &'static str, capacity: usize, usage: wgpu::BufferUsages) -> Self {
        let size = capacity.max(64);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });
        Self { buffer, capacity: size }
    }

    fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        label: &'static str,
        needed: usize,
        usage: wgpu::BufferUsages,
    ) -> bool {
        if needed > self.capacity {
            let new_cap = (needed * 2).max(self.capacity * 2).max(64);
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: new_cap as u64,
                usage,
                mapped_at_creation: false,
            });
            self.capacity = new_cap;
            true
        } else {
            false
        }
    }
}

struct GpuBufferPool {
    params_buf: CachedBuffer,
    a_buf: CachedBuffer,
    b_buf: CachedBuffer,
    bias_buf: CachedBuffer,
    c_buf: CachedBuffer,
    staging_buf: CachedBuffer,
    cached_bind_group: Option<wgpu::BindGroup>,
}

/// Pluggable WebGPU hardware accelerator for rusto-nn
pub struct WebGpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    gemm_pipeline: wgpu::ComputePipeline,
    gemm_bind_group_layout: wgpu::BindGroupLayout,
    pool: Mutex<GpuBufferPool>,
}

impl WebGpuBackend {
    /// Check whether WebGPU is supported by the current runtime environment.
    pub fn is_webgpu_supported() -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            let global = js_sys::global();
            if let Ok(nav) = js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str("navigator")) {
                if !nav.is_undefined() && !nav.is_null() {
                    if let Ok(gpu) = js_sys::Reflect::get(&nav, &wasm_bindgen::JsValue::from_str("gpu")) {
                        return !gpu.is_undefined() && !gpu.is_null();
                    }
                }
            }
            false
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            true
        }
    }

    /// Initialize WebGPU backend with auto-detected high-performance GPU adapter
    pub fn init() -> Result<Self, MnnError> {
        if !Self::is_webgpu_supported() {
            return Err(MnnError::BackendError(
                "WebGPU is not supported or not available in the current runtime environment".into(),
            ));
        }

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| MnnError::BackendError("No suitable WebGPU adapter found".into()))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rusto_nn_webgpu_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| MnnError::BackendError(format!("Failed to request WebGPU device: {}", e)))?;

        Self::from_device(device, queue)
    }

    /// Async WebGPU initialization for WebAssembly browser runtimes
    pub async fn init_async() -> Result<Self, MnnError> {
        if !Self::is_webgpu_supported() {
            return Err(MnnError::BackendError(
                "WebGPU is not supported or not available in the current runtime environment".into(),
            ));
        }

        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| MnnError::BackendError("No suitable WebGPU adapter found".into()))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rusto_nn_webgpu_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| MnnError::BackendError(format!("Failed to request WebGPU device: {}", e)))?;

        Self::from_device(device, queue)
    }

    /// Construct WebGPU backend from an existing wgpu Device and Queue (browser WASM async ready)
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self, MnnError> {
        let shader_src = include_str!("shaders/gemm.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemm_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_src)),
        });

        let gemm_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemm_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemm_pipeline_layout"),
            bind_group_layouts: &[&gemm_bind_group_layout],
            push_constant_ranges: &[],
        });

        let gemm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemm_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let initial_capacity = 1024 * 1024 * 4;
        let pool = GpuBufferPool {
            params_buf: CachedBuffer::new(
                &device,
                "params_buf",
                std::mem::size_of::<GemmParamsGpu>(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            a_buf: CachedBuffer::new(
                &device,
                "a_buf",
                initial_capacity,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            b_buf: CachedBuffer::new(
                &device,
                "b_buf",
                initial_capacity,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            bias_buf: CachedBuffer::new(
                &device,
                "bias_buf",
                1024 * 4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            c_buf: CachedBuffer::new(
                &device,
                "c_buf",
                initial_capacity,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            ),
            staging_buf: CachedBuffer::new(
                &device,
                "staging_buf",
                initial_capacity,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            cached_bind_group: None,
        };

        Ok(Self {
            device,
            queue,
            gemm_pipeline,
            gemm_bind_group_layout,
            pool: Mutex::new(pool),
        })
    }

    /// Register this backend globally into rusto-nn
    pub fn register(self) {
        rusto_nn::register_backend(Arc::new(self));
    }
}

impl DeviceBackend for WebGpuBackend {
    fn name(&self) -> &'static str {
        "WebGPU"
    }

    fn is_available(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            // On single-threaded browser WASM, blocking rx.recv() inside synchronous gemm
            // deadlocks because the browser event loop cannot process map_async callbacks.
            false
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            true
        }
    }

    fn gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        alpha: f32,
        beta: f32,
        bias: Option<&[f32]>,
        act: FusedActivation,
    ) -> Result<(), MnnError> {
        let mut pool = self.pool.lock().unwrap();

        let a_bytes = m * k * 4;
        let b_bytes = k * n * 4;
        let c_bytes = m * n * 4;
        let bias_bytes = bias.map(|bs| bs.len() * 4).unwrap_or(4);

        let mut rebind = pool.cached_bind_group.is_none();
        rebind |= pool.a_buf.ensure_capacity(
            &self.device,
            "a_buf",
            a_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        rebind |= pool.b_buf.ensure_capacity(
            &self.device,
            "b_buf",
            b_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        rebind |= pool.bias_buf.ensure_capacity(
            &self.device,
            "bias_buf",
            bias_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        rebind |= pool.c_buf.ensure_capacity(
            &self.device,
            "c_buf",
            c_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        rebind |= pool.staging_buf.ensure_capacity(
            &self.device,
            "staging_buf",
            c_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        if rebind {
            pool.cached_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemm_pooled_bind_group"),
                layout: &self.gemm_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pool.params_buf.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pool.a_buf.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pool.b_buf.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pool.bias_buf.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: pool.c_buf.buffer.as_entire_binding(),
                    },
                ],
            }));
        }

        let act_code: u32 = match act {
            FusedActivation::None => 0,
            FusedActivation::Relu => 1,
            FusedActivation::Relu6 => 2,
            FusedActivation::HardSwish => 3,
            _ => 0,
        };

        let params = GemmParamsGpu {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            alpha,
            beta,
            act: act_code,
            has_bias: if bias.is_some() { 1 } else { 0 },
            _pad: 0,
        };

        let params_slice = unsafe {
            std::slice::from_raw_parts(
                &params as *const GemmParamsGpu as *const u8,
                std::mem::size_of::<GemmParamsGpu>(),
            )
        };
        self.queue.write_buffer(&pool.params_buf.buffer, 0, params_slice);

        let a_u8 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a_bytes) };
        self.queue.write_buffer(&pool.a_buf.buffer, 0, a_u8);

        let b_u8 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b_bytes) };
        self.queue.write_buffer(&pool.b_buf.buffer, 0, b_u8);

        if let Some(bs) = bias {
            let bias_u8 = unsafe { std::slice::from_raw_parts(bs.as_ptr() as *const u8, bias_bytes) };
            self.queue.write_buffer(&pool.bias_buf.buffer, 0, bias_u8);
        }

        if beta != 0.0 {
            let c_u8 = unsafe { std::slice::from_raw_parts(c.as_ptr() as *const u8, c_bytes) };
            self.queue.write_buffer(&pool.c_buf.buffer, 0, c_u8);
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gemm_command_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm_compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.gemm_pipeline);
            compute_pass.set_bind_group(0, pool.cached_bind_group.as_ref().unwrap(), &[]);

            let workgroups_x = (n as u32 + 15) / 16;
            let workgroups_y = (m as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(&pool.c_buf.buffer, 0, &pool.staging_buf.buffer, 0, c_bytes as u64);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = pool.staging_buf.buffer.slice(..c_bytes as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| MnnError::BackendError("Channel receive failed on GPU mapping".into()))?
            .map_err(|e| MnnError::BackendError(format!("GPU buffer map failed: {:?}", e)))?;

        {
            let data = buffer_slice.get_mapped_range();
            let floats: &[f32] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, m * n)
            };
            c[..m * n].copy_from_slice(floats);
        }
        pool.staging_buf.buffer.unmap();

        Ok(())
    }
}

/// Helper to initialize WebGPU and register it into rusto-nn in a single line
pub fn init_and_register() -> Result<(), MnnError> {
    let backend = WebGpuBackend::init()?;
    backend.register();
    Ok(())
}

/// Helper to clear the active hardware accelerator
pub fn clear() {
    rusto_nn::clear_backend();
}

