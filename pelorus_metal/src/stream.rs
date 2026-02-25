use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};

pub struct Stream {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");
        let queue = device
            .newCommandQueue()
            .expect("Failed to create command queue");
        Self { device, queue }
    }

    /// Commit a blank command buffer and block until the GPU is idle.
    pub fn sync(&self) {
        let cb = self
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        cb.commit();
        cb.waitUntilCompleted();
    }
}

impl Default for Stream {
    fn default() -> Self {
        Self::new()
    }
}
