/// Runtime-selected GPU backend.
///
/// Constructed once at startup via `ActiveBackend::detect` and then held
/// for the lifetime of the processing thread. Contains the backend stream
/// by value (streams are `Arc`-backed internally so moving them is cheap).
pub enum ActiveBackend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(pelorus_cuda::Stream),
    #[cfg(feature = "metal")]
    Metal(pelorus_metal::Stream),
    #[cfg(feature = "vulkan")]
    Vulkan(pelorus_vulkan::Stream),
}

impl ActiveBackend {
    /// Select the best available backend at runtime.
    ///
    /// Precedence:
    /// 1. `cpu_only == true`  → always `Cpu`
    /// 2. macOS + metal       → `Metal` (Metal is macOS-only; the `target_os` cfg makes this compile-time-safe)
    /// 3. cuda + device found → `Cuda`
    /// 4. vulkan + device     → `Vulkan`
    /// 5. else                → `Cpu`
    #[allow(unreachable_code)]
    pub fn detect(cpu_only: bool) -> Self {
        if cpu_only {
            log::info!("Backend: CPU (forced)");
            return Self::Cpu;
        }

        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            log::info!("Backend: Metal");
            return Self::Metal(pelorus_metal::Stream::new());
        }

        #[cfg(feature = "cuda")]
        if pelorus_cuda::is_available() {
            log::info!("Backend: CUDA");
            return Self::Cuda(pelorus_cuda::Stream::new());
        }

        #[cfg(feature = "vulkan")]
        if pelorus_vulkan::is_available() {
            log::info!("Backend: Vulkan");
            return Self::Vulkan(pelorus_vulkan::Stream::new());
        }

        log::info!("Backend: CPU");
        Self::Cpu
    }

    pub fn stream_sync(&self) {
        match self {
            Self::Cpu => {}
            #[cfg(feature = "cuda")]
            Self::Cuda(s) => s.sync(),
            #[cfg(feature = "metal")]
            Self::Metal(s) => s.sync(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(s) => s.sync(),
        }
    }
}
