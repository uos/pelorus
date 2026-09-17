use pelorus_core::LioConfig;

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
    #[allow(unreachable_code)]
    pub fn detect(config: &LioConfig) -> Self {
        if config.cpu_only {
            return Self::Cpu;
        }

        if config.map_multiresolution {
            #[cfg(feature = "cuda")]
            if pelorus_cuda::is_available() {
                return Self::Cuda(pelorus_cuda::Stream::new());
            }
            #[cfg(feature = "vulkan")]
            if pelorus_vulkan::is_multiresolution_available() {
                return Self::Vulkan(pelorus_vulkan::Stream::new());
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if pelorus_metal::is_multiresolution_available() {
                return Self::Metal(pelorus_metal::Stream::new());
            }
            return Self::Cpu;
        }

        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            return Self::Metal(pelorus_metal::Stream::new());
        }

        #[cfg(feature = "cuda")]
        if pelorus_cuda::is_available() {
            return Self::Cuda(pelorus_cuda::Stream::new());
        }

        #[cfg(feature = "vulkan")]
        if pelorus_vulkan::is_available() {
            return Self::Vulkan(pelorus_vulkan::Stream::new());
        }

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
