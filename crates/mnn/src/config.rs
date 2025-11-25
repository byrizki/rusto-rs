#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardType {
    CPU = 0,
    Auto = 4,
    Metal = 1,
    Cuda = 2,
    OpenCL = 3,
    OpenGL = 6,
    Vulkan = 7,
    NN = 5,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    Normal = 0,
    High = 1,
    Low = 2,
    LowBF16 = 3,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerMode {
    Normal = 0,
    High = 1,
    Low = 2,
}

#[derive(Debug, Clone)]
pub struct BackendConfig {
    precision: PrecisionMode,
    power: PowerMode,
}

impl BackendConfig {
    pub fn new() -> Self {
        Self {
            precision: PrecisionMode::Normal,
            power: PowerMode::Normal,
        }
    }

    pub fn set_precision_mode(&mut self, mode: PrecisionMode) {
        self.precision = mode;
    }

    pub fn set_power_mode(&mut self, mode: PowerMode) {
        self.power = mode;
    }

    pub(crate) fn precision(&self) -> PrecisionMode {
        self.precision
    }

    pub(crate) fn power(&self) -> PowerMode {
        self.power
    }
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct ScheduleConfig {
    forward_type: ForwardType,
    num_thread: i32,
    backend_config: Option<BackendConfig>,
}

impl ScheduleConfig {
    pub fn new() -> Self {
        Self {
            forward_type: ForwardType::CPU,
            num_thread: 4,
            backend_config: None,
        }
    }

    pub fn set_type(&mut self, forward_type: ForwardType) {
        self.forward_type = forward_type;
    }

    pub fn set_num_thread(&mut self, num: i32) {
        self.num_thread = num;
    }

    pub fn set_backend_config(&mut self, config: BackendConfig) {
        self.backend_config = Some(config);
    }

    pub(crate) fn forward_type(&self) -> ForwardType {
        self.forward_type
    }

    pub(crate) fn num_thread(&self) -> i32 {
        self.num_thread
    }

    pub(crate) fn backend_config(&self) -> Option<&BackendConfig> {
        self.backend_config.as_ref()
    }
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self::new()
    }
}
