use metal::{CommandBuffer, CommandQueue, Device};

pub struct Stream {
    pub(crate) device: Device,
    pub(crate) queue: CommandQueue,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new() -> Self {
        let device = Device::system_default().expect("No Metal device found");
        let queue = device.new_command_queue();
        Self { device, queue }
    }

    /// Commit a blank command buffer and block until the GPU is idle.
    pub fn sync(&self) {
        let cb = self.queue.new_command_buffer();
        cb.commit();
        cb.wait_until_completed();
    }

    /// Create a new command buffer for encoding work.
    pub(crate) fn command_buffer(&self) -> CommandBuffer {
        self.queue.new_command_buffer().to_owned()
    }
}

impl Default for Stream {
    fn default() -> Self {
        Self::new()
    }
}
