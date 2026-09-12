use crate::context::VulkanContext;
use crate::stream::{HostContext, Stream, StreamBackend};
use ash::{Device, vk};
use pelorus_core::PointXYZIT;
use std::mem::size_of;
use std::sync::Arc;

pub(crate) struct BufferAllocation {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) memory_flags: vk::MemoryPropertyFlags,
    pub(crate) allocation_size: vk::DeviceSize,
}

pub(crate) fn allocate_buffer_required(
    context: &Arc<VulkanContext>,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    required: vk::MemoryPropertyFlags,
    debug_label: &str,
) -> BufferAllocation {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        context
            .device
            .create_buffer(&info, None)
            .expect("failed to create Vulkan buffer")
    };

    let req = unsafe { context.device.get_buffer_memory_requirements(buffer) };
    let Some(memory_type_index) =
        find_memory_type(&context.memory_properties, req.memory_type_bits, required)
    else {
        unsafe {
            context.device.destroy_buffer(buffer, None);
        }
        panic!(
            "strict Vulkan allocation failed for {debug_label}: no compatible memory type with required flags {:?} (requested {} bytes, driver-required {} bytes). This run requires VRAM-first DEVICE_LOCAL buffers.",
            required, size, req.size
        );
    };

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(memory_type_index);

    let memory = unsafe {
        context
            .device
            .allocate_memory(&alloc_info, None)
            .expect("failed to allocate Vulkan memory")
    };

    unsafe {
        context
            .device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("failed to bind Vulkan buffer memory");
    }

    BufferAllocation {
        buffer,
        memory,
        memory_flags: required,
        allocation_size: req.size,
    }
}

pub(crate) fn find_memory_type(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..memory_properties.memory_type_count {
        let is_supported = (type_filter & (1 << i)) != 0;
        let has_flags = memory_properties.memory_types[i as usize]
            .property_flags
            .contains(required);
        if is_supported && has_flags {
            return Some(i);
        }
    }
    None
}

pub(crate) fn allocate_buffer(
    context: &Arc<VulkanContext>,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    memory_preferences: &[vk::MemoryPropertyFlags],
) -> BufferAllocation {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        context
            .device
            .create_buffer(&info, None)
            .expect("failed to create Vulkan buffer")
    };

    let req = unsafe { context.device.get_buffer_memory_requirements(buffer) };

    let mut selected: Option<(u32, vk::MemoryPropertyFlags)> = None;
    for &flags in memory_preferences {
        if let Some(index) =
            find_memory_type(&context.memory_properties, req.memory_type_bits, flags)
        {
            selected = Some((index, flags));
            break;
        }
    }

    let (memory_type_index, memory_flags) =
        selected.expect("failed to find compatible Vulkan memory type");

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(memory_type_index);

    let memory = unsafe {
        context
            .device
            .allocate_memory(&alloc_info, None)
            .expect("failed to allocate Vulkan memory")
    };

    unsafe {
        context
            .device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("failed to bind Vulkan buffer memory");
    }

    BufferAllocation {
        buffer,
        memory,
        memory_flags,
        allocation_size: req.size,
    }
}

pub(crate) fn destroy_buffer(context: &Arc<VulkanContext>, alloc: BufferAllocation) {
    unsafe {
        context.device.destroy_buffer(alloc.buffer, None);
        context.device.free_memory(alloc.memory, None);
    }
}

fn allocate_upload_staging_buffer(
    context: &Arc<VulkanContext>,
    size: vk::DeviceSize,
) -> BufferAllocation {
    allocate_buffer(
        context,
        size.max(1),
        vk::BufferUsageFlags::TRANSFER_SRC,
        &[
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        ],
    )
}

fn allocate_download_staging_buffer(
    context: &Arc<VulkanContext>,
    size: vk::DeviceSize,
) -> BufferAllocation {
    allocate_buffer(
        context,
        size.max(1),
        vk::BufferUsageFlags::TRANSFER_DST,
        &HOST_READBACK_PREFERENCES,
    )
}

pub(crate) fn submit_copy(
    context: &Arc<VulkanContext>,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: vk::DeviceSize,
) {
    if size == 0 {
        return;
    }

    let _lock = context
        .submit_lock
        .lock()
        .expect("vulkan submit mutex poisoned");

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(context.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd = unsafe {
        context
            .device
            .allocate_command_buffers(&alloc_info)
            .expect("failed to allocate Vulkan command buffer")[0]
    };

    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        context
            .device
            .begin_command_buffer(cmd, &begin)
            .expect("failed to begin Vulkan command buffer");

        let copy_region = vk::BufferCopy::default().size(size);
        context
            .device
            .cmd_copy_buffer(cmd, src, dst, std::slice::from_ref(&copy_region));

        context
            .device
            .end_command_buffer(cmd)
            .expect("failed to end Vulkan command buffer");

        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        context
            .device
            .queue_submit(
                context.queue,
                std::slice::from_ref(&submit),
                vk::Fence::null(),
            )
            .expect("failed to submit Vulkan transfer");

        context
            .device
            .queue_wait_idle(context.queue)
            .expect("failed waiting Vulkan transfer completion");

        context
            .device
            .free_command_buffers(context.command_pool, std::slice::from_ref(&cmd));
    }
}

unsafe fn map_write_bytes(
    device: &Device,
    memory: vk::DeviceMemory,
    memory_flags: vk::MemoryPropertyFlags,
    src: &[u8],
) {
    if src.is_empty() {
        return;
    }

    let ptr = unsafe {
        device
            .map_memory(
                memory,
                0,
                src.len() as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .expect("failed to map Vulkan memory for write")
    } as *mut u8;

    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len());
    }

    if !memory_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
        let range = vk::MappedMemoryRange::default()
            .memory(memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe {
            device
                .flush_mapped_memory_ranges(std::slice::from_ref(&range))
                .expect("failed to flush Vulkan mapped memory");
        }
    }

    unsafe {
        device.unmap_memory(memory);
    }
}

unsafe fn map_read_bytes(
    device: &Device,
    memory: vk::DeviceMemory,
    memory_flags: vk::MemoryPropertyFlags,
    dst: &mut [u8],
) {
    if dst.is_empty() {
        return;
    }

    let ptr = unsafe {
        device
            .map_memory(
                memory,
                0,
                dst.len() as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .expect("failed to map Vulkan memory for read")
    } as *const u8;

    if !memory_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
        let range = vk::MappedMemoryRange::default()
            .memory(memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe {
            device
                .invalidate_mapped_memory_ranges(std::slice::from_ref(&range))
                .expect("failed to invalidate Vulkan mapped memory");
        }
    }

    unsafe {
        std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len());
        device.unmap_memory(memory);
    }
}

unsafe fn persistent_map_ptr(
    device: &Device,
    memory: vk::DeviceMemory,
    memory_flags: vk::MemoryPropertyFlags,
    allocation_size: vk::DeviceSize,
) -> Option<*mut u8> {
    if !memory_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
        return None;
    }

    let ptr = unsafe {
        device
            .map_memory(
                memory,
                0,
                allocation_size.max(1),
                vk::MemoryMapFlags::empty(),
            )
            .expect("failed to persistently map Vulkan memory")
    } as *mut u8;

    Some(ptr)
}

unsafe fn persistent_write_bytes(
    device: &Device,
    memory: vk::DeviceMemory,
    memory_flags: vk::MemoryPropertyFlags,
    mapped_ptr: *mut u8,
    src: &[u8],
) {
    if src.is_empty() {
        return;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), mapped_ptr, src.len());
    }

    if !memory_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
        let range = vk::MappedMemoryRange::default()
            .memory(memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe {
            device
                .flush_mapped_memory_ranges(std::slice::from_ref(&range))
                .expect("failed to flush Vulkan mapped memory");
        }
    }
}

unsafe fn persistent_read_bytes(
    device: &Device,
    memory: vk::DeviceMemory,
    memory_flags: vk::MemoryPropertyFlags,
    mapped_ptr: *const u8,
    dst: &mut [u8],
) {
    if dst.is_empty() {
        return;
    }

    if !memory_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
        let range = vk::MappedMemoryRange::default()
            .memory(memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe {
            device
                .invalidate_mapped_memory_ranges(std::slice::from_ref(&range))
                .expect("failed to invalidate Vulkan mapped memory");
        }
    }

    unsafe {
        std::ptr::copy_nonoverlapping(mapped_ptr, dst.as_mut_ptr(), dst.len());
    }
}

pub(crate) enum BufferStorage {
    Vulkan {
        context: Arc<VulkanContext>,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        memory_flags: vk::MemoryPropertyFlags,
        mapped_ptr: Option<*mut u8>,
        allocation_size: u64,
    },
    Host {
        owner: Arc<HostContext>,
        data: Vec<u8>,
    },
}

pub struct VulkanBuffer {
    pub(crate) storage: BufferStorage,
    pub element_size: usize,
    pub n: usize,
    capacity_elems: usize,
    byte_capacity: usize,
}

unsafe impl Send for VulkanBuffer {}

impl std::fmt::Debug for VulkanBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let backend = match &self.storage {
            BufferStorage::Vulkan { .. } => "vulkan",
            BufferStorage::Host { .. } => "host-fallback",
        };
        f.debug_struct("VulkanBuffer")
            .field("backend", &backend)
            .field("element_size", &self.element_size)
            .field("n", &self.n)
            .field("capacity", &self.capacity())
            .finish()
    }
}

impl Clone for VulkanBuffer {
    fn clone(&self) -> Self {
        match &self.storage {
            BufferStorage::Vulkan {
                context, buffer, ..
            } => {
                let requested = self.byte_capacity as vk::DeviceSize;
                let alloc = allocate_buffer_required(
                    context,
                    requested.max(1),
                    vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    "vulkan buffer clone",
                );
                if requested > 0 {
                    submit_copy(context, *buffer, alloc.buffer, requested);
                }
                increment_allocation(alloc.allocation_size as u64);
                Self {
                    storage: BufferStorage::Vulkan {
                        context: Arc::clone(context),
                        buffer: alloc.buffer,
                        memory: alloc.memory,
                        memory_flags: alloc.memory_flags,
                        mapped_ptr: None,
                        allocation_size: alloc.allocation_size as u64,
                    },
                    element_size: self.element_size,
                    n: self.n,
                    capacity_elems: self.capacity_elems,
                    byte_capacity: self.byte_capacity,
                }
            }
            BufferStorage::Host { owner, data } => {
                let copied = data.clone();
                increment_allocation(copied.len() as u64);
                Self {
                    storage: BufferStorage::Host {
                        owner: Arc::clone(owner),
                        data: copied,
                    },
                    element_size: self.element_size,
                    n: self.n,
                    capacity_elems: self.capacity_elems,
                    byte_capacity: self.byte_capacity,
                }
            }
        }
    }
}

impl VulkanBuffer {
    pub fn new(stream: &Stream, element_size: usize, n: usize) -> Self {
        Self::new_device_local_required(stream, element_size, n)
    }

    pub fn new_device_local_required(stream: &Stream, element_size: usize, n: usize) -> Self {
        Self::new_with_policy(
            stream,
            element_size,
            n,
            BufferMemoryPolicy::DeviceLocalRequired,
        )
    }

    pub fn new_host_visible(stream: &Stream, element_size: usize, n: usize) -> Self {
        Self::new_with_policy(
            stream,
            element_size,
            n,
            BufferMemoryPolicy::HostVisiblePreferred,
        )
    }

    /// Host-visible buffer the CPU mainly *reads*.
    ///
    /// `find_memory_type` returns the lowest matching index, and on NVIDIA that is
    /// the plain `HOST_VISIBLE|HOST_COHERENT` type, which is write-combined: fine to
    /// write, but host reads from it run at a few hundred MB/s because they are
    /// uncached. Asking for `HOST_CACHED` picks the cached system-memory type
    /// instead, which reads at normal RAM speed.
    pub fn new_host_readback(stream: &Stream, element_size: usize, n: usize) -> Self {
        Self::new_with_policy(stream, element_size, n, BufferMemoryPolicy::HostReadback)
    }

    fn new_with_policy(
        stream: &Stream,
        element_size: usize,
        n: usize,
        policy: BufferMemoryPolicy,
    ) -> Self {
        assert!(element_size > 0, "element size must be > 0");
        let requested = element_size
            .checked_mul(n)
            .expect("buffer size overflow creating VulkanBuffer");

        match &stream.backend {
            StreamBackend::Vulkan(context) => {
                let usage = vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER;
                let alloc = match policy {
                    BufferMemoryPolicy::DeviceLocalRequired => allocate_buffer_required(
                        context,
                        (requested as vk::DeviceSize).max(1),
                        usage,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        "vulkan compute buffer",
                    ),
                    BufferMemoryPolicy::HostVisiblePreferred => allocate_buffer(
                        context,
                        (requested as vk::DeviceSize).max(1),
                        usage,
                        &[
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                            vk::MemoryPropertyFlags::HOST_VISIBLE,
                            vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        ],
                    ),
                    BufferMemoryPolicy::HostReadback => allocate_buffer(
                        context,
                        (requested as vk::DeviceSize).max(1),
                        usage,
                        &HOST_READBACK_PREFERENCES,
                    ),
                };
                let mapped_ptr = unsafe {
                    persistent_map_ptr(
                        &context.device,
                        alloc.memory,
                        alloc.memory_flags,
                        alloc.allocation_size,
                    )
                };
                increment_allocation(alloc.allocation_size as u64);
                Self {
                    storage: BufferStorage::Vulkan {
                        context: Arc::clone(context),
                        buffer: alloc.buffer,
                        memory: alloc.memory,
                        memory_flags: alloc.memory_flags,
                        mapped_ptr,
                        allocation_size: alloc.allocation_size as u64,
                    },
                    element_size,
                    n,
                    capacity_elems: n,
                    byte_capacity: requested,
                }
            }
            StreamBackend::Host(owner) => {
                let data = vec![0u8; requested];
                increment_allocation(data.len() as u64);
                Self {
                    storage: BufferStorage::Host {
                        owner: Arc::clone(owner),
                        data,
                    },
                    element_size,
                    n,
                    capacity_elems: n,
                    byte_capacity: requested,
                }
            }
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity_elems
    }

    fn assert_stream_compatibility(&self, stream: &Stream) {
        match (&self.storage, &stream.backend) {
            (BufferStorage::Vulkan { context, .. }, StreamBackend::Vulkan(stream_ctx)) => {
                assert!(
                    Arc::ptr_eq(context, stream_ctx),
                    "stream and VulkanBuffer must belong to the same Vulkan device"
                );
            }
            (BufferStorage::Host { owner, .. }, StreamBackend::Host(stream_owner)) => {
                assert!(
                    Arc::ptr_eq(owner, stream_owner),
                    "stream and VulkanBuffer must belong to the same host fallback context"
                );
            }
            _ => {
                panic!("stream and VulkanBuffer backend mismatch");
            }
        }
    }

    pub fn upload<T: Copy>(&mut self, stream: &Stream, src: &[T]) {
        self.assert_stream_compatibility(stream);
        assert_eq!(size_of::<T>(), self.element_size, "element size mismatch");
        assert!(
            src.len() <= self.capacity(),
            "source larger than destination buffer"
        );

        let byte_len = size_of::<T>()
            .checked_mul(src.len())
            .expect("upload size overflow");
        let src_bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, byte_len) };

        match &mut self.storage {
            BufferStorage::Host { data, .. } => {
                data[..byte_len].copy_from_slice(src_bytes);
            }
            BufferStorage::Vulkan {
                context,
                buffer,
                memory,
                memory_flags,
                mapped_ptr,
                ..
            } => {
                if memory_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
                    unsafe {
                        match mapped_ptr {
                            Some(ptr) => persistent_write_bytes(
                                &context.device,
                                *memory,
                                *memory_flags,
                                *ptr,
                                src_bytes,
                            ),
                            None => {
                                map_write_bytes(&context.device, *memory, *memory_flags, src_bytes)
                            }
                        }
                    }
                } else {
                    let staging =
                        allocate_upload_staging_buffer(context, byte_len as vk::DeviceSize);

                    unsafe {
                        map_write_bytes(
                            &context.device,
                            staging.memory,
                            staging.memory_flags,
                            src_bytes,
                        );
                    }
                    submit_copy(context, staging.buffer, *buffer, byte_len as vk::DeviceSize);
                    destroy_buffer(context, staging);
                }
            }
        }

        self.n = src.len();
    }

    pub fn download<T: Copy + Default>(&self, stream: &Stream) -> Vec<T> {
        self.assert_stream_compatibility(stream);
        assert_eq!(size_of::<T>(), self.element_size, "element size mismatch");

        let mut out = vec![T::default(); self.n];
        self.download_into(stream, &mut out);
        out
    }

    pub fn download_into<T: Copy>(&self, stream: &Stream, dst: &mut [T]) {
        self.assert_stream_compatibility(stream);
        assert_eq!(size_of::<T>(), self.element_size, "element size mismatch");
        assert!(dst.len() <= self.n, "destination larger than source length");

        let byte_len = size_of::<T>()
            .checked_mul(dst.len())
            .expect("download size overflow");
        let dst_bytes =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, byte_len) };

        match &self.storage {
            BufferStorage::Host { data, .. } => {
                dst_bytes.copy_from_slice(&data[..byte_len]);
            }
            BufferStorage::Vulkan {
                context,
                buffer,
                memory,
                memory_flags,
                mapped_ptr,
                ..
            } => {
                if memory_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
                    unsafe {
                        match mapped_ptr {
                            Some(ptr) => persistent_read_bytes(
                                &context.device,
                                *memory,
                                *memory_flags,
                                *ptr,
                                dst_bytes,
                            ),
                            None => {
                                map_read_bytes(&context.device, *memory, *memory_flags, dst_bytes)
                            }
                        }
                    }
                } else {
                    let staging =
                        allocate_download_staging_buffer(context, byte_len as vk::DeviceSize);

                    submit_copy(context, *buffer, staging.buffer, byte_len as vk::DeviceSize);
                    unsafe {
                        map_read_bytes(
                            &context.device,
                            staging.memory,
                            staging.memory_flags,
                            dst_bytes,
                        );
                    }
                    destroy_buffer(context, staging);
                }
            }
        }
    }

    pub fn from_cloud(stream: &Stream, cloud: &[PointXYZIT]) -> Self {
        let mut b = Self::new(stream, size_of::<PointXYZIT>(), cloud.len());
        b.upload(stream, cloud);
        b
    }

    pub fn into_cloud(&self, stream: &Stream) -> Vec<PointXYZIT> {
        self.download(stream)
    }

    pub(crate) fn vulkan_context_and_buffer(&self) -> Option<(Arc<VulkanContext>, vk::Buffer)> {
        match &self.storage {
            BufferStorage::Vulkan {
                context, buffer, ..
            } => Some((Arc::clone(context), *buffer)),
            BufferStorage::Host { .. } => None,
        }
    }
}

#[derive(Clone, Copy)]
enum BufferMemoryPolicy {
    DeviceLocalRequired,
    HostVisiblePreferred,
    HostReadback,
}

/// Memory preferences for buffers the host reads, best first. `HOST_CACHED` is what
/// makes the read fast; everything after it is a correctness fallback.
const HOST_READBACK_PREFERENCES: [vk::MemoryPropertyFlags; 4] = [
    vk::MemoryPropertyFlags::from_raw(
        vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
            | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw()
            | vk::MemoryPropertyFlags::HOST_CACHED.as_raw(),
    ),
    vk::MemoryPropertyFlags::from_raw(
        vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
            | vk::MemoryPropertyFlags::HOST_CACHED.as_raw(),
    ),
    vk::MemoryPropertyFlags::from_raw(
        vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
            | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
    ),
    vk::MemoryPropertyFlags::HOST_VISIBLE,
];

pub(crate) static ALLOCATED_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(crate) fn increment_allocation(bytes: u64) {
    ALLOCATED_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn decrement_allocation(bytes: u64) {
    ALLOCATED_BYTES.fetch_sub(bytes, std::sync::atomic::Ordering::Relaxed);
}

pub fn get_used_memory() -> u64 {
    ALLOCATED_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        match &self.storage {
            BufferStorage::Vulkan {
                context,
                buffer,
                memory,
                mapped_ptr,
                allocation_size,
                ..
            } => {
                // After `shutdown` the device is gone and these handles died with it.
                if context.is_alive() {
                    unsafe {
                        if mapped_ptr.is_some() {
                            context.device.unmap_memory(*memory);
                        }
                        context.device.destroy_buffer(*buffer, None);
                        context.device.free_memory(*memory, None);
                    }
                }
                decrement_allocation(*allocation_size);
            }
            BufferStorage::Host { data, .. } => {
                decrement_allocation(data.len() as u64);
            }
        }
    }
}
