use crate::context::VulkanContext;
#[cfg(feature = "vulkan_enable")]
use crate::context::global_vulkan_context;
use std::sync::{Arc, OnceLock};

#[derive(Default)]
pub(crate) struct HostContext;

pub(crate) enum StreamBackend {
    Vulkan(Arc<VulkanContext>),
    Host(Arc<HostContext>),
}

#[derive(Clone)]
pub struct Stream {
    pub(crate) backend: StreamBackend,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let backend = match &self.backend {
            StreamBackend::Vulkan(_) => "vulkan",
            StreamBackend::Host(_) => "host-fallback",
        };
        f.debug_struct("Stream").field("backend", &backend).finish()
    }
}

impl Clone for StreamBackend {
    fn clone(&self) -> Self {
        match self {
            Self::Vulkan(ctx) => Self::Vulkan(Arc::clone(ctx)),
            Self::Host(ctx) => Self::Host(Arc::clone(ctx)),
        }
    }
}

pub(crate) static GLOBAL_HOST_CONTEXT: OnceLock<Arc<HostContext>> = OnceLock::new();

impl Stream {
    #[cfg(feature = "vulkan_enable")]
    pub fn new() -> Self {
        match global_vulkan_context() {
            Ok(ctx) => Self {
                backend: StreamBackend::Vulkan(ctx),
            },
            Err(e) => {
                log::warn!(
                    "Vulkan backend unavailable, falling back to host buffer mode: {}",
                    e
                );
                let host_ctx = GLOBAL_HOST_CONTEXT.get_or_init(|| Arc::new(HostContext));
                Self {
                    backend: StreamBackend::Host(Arc::clone(host_ctx)),
                }
            }
        }
    }

    #[cfg(not(feature = "vulkan_enable"))]
    pub fn new() -> Self {
        let host_ctx = GLOBAL_HOST_CONTEXT.get_or_init(|| Arc::new(HostContext));
        Self {
            backend: StreamBackend::Host(Arc::clone(host_ctx)),
        }
    }

    pub fn sync(&self) {
        if let StreamBackend::Vulkan(ctx) = &self.backend {
            ctx.sync();
        }
    }

    pub fn is_vulkan(&self) -> bool {
        matches!(self.backend, StreamBackend::Vulkan(_))
    }
}

impl Default for Stream {
    fn default() -> Self {
        Self::new()
    }
}
