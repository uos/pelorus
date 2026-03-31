#![cfg(vulkan_available)]

use ash::{Device, Entry, Instance, vk};
use bytemuck::{Pod, Zeroable};
use pelorus_core::PointXYZIT;
use ros_pointcloud2::points::PointXYZINormal;
use spole::{IVoxHashMap, IVoxNearbyType};
use std::ffi::CString;
use std::mem::size_of;
use std::sync::{Arc, Mutex, OnceLock};

pub use types::{Matrix3, NeighborType, Point, PointNormal, Pose, Quaternion, Vector3};

mod shaders;
mod types;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct VrdxSorter(u64);

#[repr(C)]
struct VrdxSorterCreateInfo {
    physical_device: vk::PhysicalDevice,
    device: vk::Device,
    pipeline_cache: vk::PipelineCache,
}

#[repr(C)]
struct VrdxSorterStorageRequirements {
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
}

#[link(name = "vk_radix_sort", kind = "static")]
unsafe extern "C" {
    fn vrdxCreateSorter(p_create_info: *const VrdxSorterCreateInfo, p_sorter: *mut VrdxSorter);
    fn vrdxGetSorterKeyValueStorageRequirements(
        sorter: VrdxSorter,
        max_element_count: u32,
        requirements: *mut VrdxSorterStorageRequirements,
    );
    fn vrdxCmdSortKeyValue(
        command_buffer: vk::CommandBuffer,
        sorter: VrdxSorter,
        element_count: u32,
        keys_buffer: vk::Buffer,
        keys_offset: vk::DeviceSize,
        values_buffer: vk::Buffer,
        values_offset: vk::DeviceSize,
        storage_buffer: vk::Buffer,
        storage_offset: vk::DeviceSize,
        query_pool: vk::QueryPool,
        query: u32,
    );
}

struct VulkanContext {
    #[allow(dead_code)]
    entry: Entry,
    instance: Instance,
    device: Device,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    submit_lock: Mutex<()>,
}

impl VulkanContext {
    fn try_new() -> Result<Self, String> {
        let entry =
            unsafe { Entry::load().map_err(|e| format!("failed to load Vulkan loader: {e}"))? };

        let app_name = CString::new("pelorus-vulkan").expect("invalid app name");
        let engine_name = CString::new("pelorus").expect("invalid engine name");
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name.as_c_str())
            .application_version(0)
            .engine_name(engine_name.as_c_str())
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let instance_ci = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe {
            entry
                .create_instance(&instance_ci, None)
                .map_err(|e| format!("failed to create Vulkan instance: {e}"))?
        };

        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .map_err(|e| format!("failed to enumerate Vulkan physical devices: {e}"))?
        };
        if physical_devices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            return Err("no Vulkan physical device available".to_string());
        }

        let mut selected: Option<(vk::PhysicalDevice, u32, i32)> = None;
        for physical in physical_devices {
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            if properties.api_version < vk::make_api_version(0, 1, 2, 0) {
                continue;
            }

            let mut feat12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut feat = vk::PhysicalDeviceFeatures2::default().push_next(&mut feat12);
            unsafe { instance.get_physical_device_features2(physical, &mut feat) };

            if feat12.buffer_device_address == vk::FALSE {
                continue;
            }

            let families =
                unsafe { instance.get_physical_device_queue_family_properties(physical) };

            let mut best_family: Option<(u32, i32)> = None;
            for (idx, family) in families.iter().enumerate() {
                if family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    let mut score = 10;
                    if !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        score += 100; // Prefer compute-only queues
                    }
                    if best_family.is_none() || score > best_family.unwrap().1 {
                        best_family = Some((idx as u32, score));
                    }
                }
            }

            if let Some((idx, family_score)) = best_family {
                let mut device_score = family_score;
                if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                    device_score += 1000;
                } else if properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                    device_score += 500;
                }

                if selected.is_none() || device_score > selected.unwrap().2 {
                    selected = Some((physical, idx, device_score));
                }
            }
        }

        let Some((physical_device, queue_family_index, _)) = selected else {
            unsafe { instance.destroy_instance(None) };
            return Err("no Vulkan 1.2 physical device with compute and bufferDeviceAddress support available".to_string());
        };

        let queue_priorities = [1.0_f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .scalar_block_layout(true);
        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_ci))
            .push_next(&mut features12);

        let device = match unsafe { instance.create_device(physical_device, &device_ci, None) } {
            Ok(d) => d,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(format!("failed to create Vulkan device: {e}"));
            }
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = match unsafe { device.create_command_pool(&pool_ci, None) } {
            Ok(pool) => pool,
            Err(e) => {
                unsafe {
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(format!("failed to create Vulkan command pool: {e}"));
            }
        };

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Ok(Self {
            entry,
            instance,
            device,
            physical_device,
            queue,
            command_pool,
            memory_properties,
            submit_lock: Mutex::new(()),
        })
    }

    fn sync(&self) {
        let _lock = self
            .submit_lock
            .lock()
            .expect("vulkan submit mutex poisoned");
        unsafe {
            self.device
                .queue_wait_idle(self.queue)
                .expect("failed waiting for Vulkan queue idle");
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Default)]
struct HostContext;

enum StreamBackend {
    Vulkan(Arc<VulkanContext>),
    Host(Arc<HostContext>),
}

#[derive(Clone)]
pub struct Stream {
    backend: StreamBackend,
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

static GLOBAL_VULKAN_CONTEXT: OnceLock<Result<Arc<VulkanContext>, String>> = OnceLock::new();
static GLOBAL_HOST_CONTEXT: OnceLock<Arc<HostContext>> = OnceLock::new();

impl Stream {
    pub fn new() -> Self {
        let vulkan_res =
            GLOBAL_VULKAN_CONTEXT.get_or_init(|| VulkanContext::try_new().map(Arc::new));

        match vulkan_res {
            Ok(ctx) => Self {
                backend: StreamBackend::Vulkan(Arc::clone(ctx)),
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

struct BufferAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_flags: vk::MemoryPropertyFlags,
    allocation_size: vk::DeviceSize,
}

fn find_memory_type(
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

fn allocate_buffer(
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

fn destroy_buffer(context: &Arc<VulkanContext>, alloc: BufferAllocation) {
    unsafe {
        context.device.destroy_buffer(alloc.buffer, None);
        context.device.free_memory(alloc.memory, None);
    }
}

fn submit_copy(
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

enum BufferStorage {
    Vulkan {
        context: Arc<VulkanContext>,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        memory_flags: vk::MemoryPropertyFlags,
        allocation_size: u64,
    },
    Host {
        owner: Arc<HostContext>,
        data: Vec<u8>,
    },
}

pub struct VulkanBuffer {
    storage: BufferStorage,
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
                let alloc = allocate_buffer(
                    context,
                    requested.max(1),
                    vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                    &[
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                        vk::MemoryPropertyFlags::HOST_VISIBLE,
                    ],
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
        Self::new_with_preferences(
            stream,
            element_size,
            n,
            &[
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            ],
        )
    }

    pub fn new_host_visible(stream: &Stream, element_size: usize, n: usize) -> Self {
        Self::new_with_preferences(
            stream,
            element_size,
            n,
            &[
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ],
        )
    }

    fn new_with_preferences(
        stream: &Stream,
        element_size: usize,
        n: usize,
        memory_preferences: &[vk::MemoryPropertyFlags],
    ) -> Self {
        assert!(element_size > 0, "element size must be > 0");
        let requested = element_size
            .checked_mul(n)
            .expect("buffer size overflow creating VulkanBuffer");

        match &stream.backend {
            StreamBackend::Vulkan(context) => {
                let alloc = allocate_buffer(
                    context,
                    (requested as vk::DeviceSize).max(1),
                    vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                    memory_preferences,
                );
                increment_allocation(alloc.allocation_size as u64);
                Self {
                    storage: BufferStorage::Vulkan {
                        context: Arc::clone(context),
                        buffer: alloc.buffer,
                        memory: alloc.memory,
                        memory_flags: alloc.memory_flags,
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
                ..
            } => {
                if memory_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
                    unsafe {
                        map_write_bytes(&context.device, *memory, *memory_flags, src_bytes);
                    }
                } else {
                    let staging = allocate_buffer(
                        context,
                        (byte_len as vk::DeviceSize).max(1),
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        &[
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                            vk::MemoryPropertyFlags::HOST_VISIBLE,
                        ],
                    );

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
                ..
            } => {
                if memory_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
                    unsafe {
                        map_read_bytes(&context.device, *memory, *memory_flags, dst_bytes);
                    }
                } else {
                    let staging = allocate_buffer(
                        context,
                        (byte_len as vk::DeviceSize).max(1),
                        vk::BufferUsageFlags::TRANSFER_DST,
                        &[
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                            vk::MemoryPropertyFlags::HOST_VISIBLE,
                        ],
                    );

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

    fn vulkan_context_and_buffer(&self) -> Option<(Arc<VulkanContext>, vk::Buffer)> {
        match &self.storage {
            BufferStorage::Vulkan {
                context, buffer, ..
            } => Some((Arc::clone(context), *buffer)),
            BufferStorage::Host { .. } => None,
        }
    }
}

const EMPTY_KEY: i32 = i32::MIN;
const THREADS_PER_GROUP: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct GpuVoxelDist {
    lock_flag: i32,
    transform: [f32; 9],
    point_count: u32,
    mean: [f32; 3],
    covariance: [f32; 9],
    last_scan: u32,
}

const _: () = assert!(size_of::<GpuVoxelDist>() == 96);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstants {
    pose_quat: [f32; 4],
    pose_trans: [f32; 3],
    _pad0: f32,
    lidar_to_imu_quat: [f32; 4],
    lidar_to_imu_pos: [f32; 3],
    _pad1: f32,
    resolution: f32,
    scan_number: u32,
    sensitivity: f32,
    min_dist: f32,
    unique_key_count: u32,
    table_cap: u32,
    dist_cap: u32,
    nbtype: u32,
    sphericity_threshold: f32,
    j: u32,
    k: u32,
    n: u32,
    cap: u32,
    valid_size: u32,
}

impl Default for PushConstants {
    fn default() -> Self {
        Self {
            pose_quat: [0.0, 0.0, 0.0, 1.0],
            pose_trans: [0.0; 3],
            _pad0: 0.0,
            lidar_to_imu_quat: [0.0, 0.0, 0.0, 1.0],
            lidar_to_imu_pos: [0.0; 3],
            _pad1: 0.0,
            resolution: 1.0,
            scan_number: 0,
            sensitivity: 0.1,
            min_dist: 0.0,
            unique_key_count: 0,
            table_cap: 0,
            dist_cap: 0,
            nbtype: 26,
            sphericity_threshold: 0.5,
            j: 0,
            k: 0,
            n: 0,
            cap: 0,
            valid_size: 0,
        }
    }
}

struct VulkanComputePipelines {
    context: Arc<VulkanContext>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    hashmap_init: vk::Pipeline,
    rebuild_hashmap_keyed: vk::Pipeline,
    calculate_keys: vk::Pipeline,
    reorder_points: vk::Pipeline,
    build_voxels: vk::Pipeline,
    plane_krnl: vk::Pipeline,
    rle: vk::Pipeline,
    assign_indices: vk::Pipeline,
    lru_score: vk::Pipeline,
    shader_modules: Vec<vk::ShaderModule>,
}

impl Drop for VulkanComputePipelines {
    fn drop(&mut self) {
        unsafe {
            let device = &self.context.device;
            device.destroy_pipeline(self.hashmap_init, None);
            device.destroy_pipeline(self.rebuild_hashmap_keyed, None);
            device.destroy_pipeline(self.calculate_keys, None);
            device.destroy_pipeline(self.reorder_points, None);
            device.destroy_pipeline(self.build_voxels, None);
            device.destroy_pipeline(self.plane_krnl, None);
            device.destroy_pipeline(self.rle, None);
            device.destroy_pipeline(self.assign_indices, None);
            device.destroy_pipeline(self.lru_score, None);
            for module in self.shader_modules.drain(..) {
                device.destroy_shader_module(module, None);
            }
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

struct ScratchBuffers {
    keys: VulkanBuffer,
    indices: VulkanBuffer,
    packed: VulkanBuffer,
    run_starts: VulkanBuffer,
    run_ends: VulkanBuffer,
    unique_keys: VulkanBuffer,
    target_indices: VulkanBuffer,
    update_flags: VulkanBuffer,
    sorted_points: VulkanBuffer,
    counter: VulkanBuffer,
    sorter_scratch: VulkanBuffer,
    lru_mapping: VulkanBuffer,
    lru_keys: VulkanBuffer,
    sorter: VrdxSorter,
    capacity: usize,
}

struct VulkanGpuMapState {
    context: Arc<VulkanContext>,
    compute: Arc<VulkanComputePipelines>,
    keys_buf: VulkanBuffer,
    values_buf: VulkanBuffer,
    dist_buf: VulkanBuffer,
    dist_keys_buf: VulkanBuffer,
    dummy_buf: VulkanBuffer,
    scratch: ScratchBuffers,
    dist_cap: u32,
    table_cap: u32,
    resolution: f32,
    sensitivity: f32,
    min_dist: f32,
    scan_number: u32,
    nbtype: u32,
    add_cursor: u32,
    lru_cursor: u32,
}

struct HostMapState {
    map: IVoxHashMap,
}

enum MapBackendState {
    Vulkan(VulkanGpuMapState),
    Host(HostMapState),
}

struct VulkanMapState {
    backend: MapBackendState,
}

static MAP_STATE: OnceLock<Mutex<Option<VulkanMapState>>> = OnceLock::new();

fn global_state() -> &'static Mutex<Option<VulkanMapState>> {
    MAP_STATE.get_or_init(|| Mutex::new(None))
}

fn with_state<R>(f: impl FnOnce(&mut VulkanMapState) -> R) -> R {
    let mut guard = global_state().lock().expect("vulkan map mutex poisoned");
    let state = guard
        .as_mut()
        .expect("vulkan map is not initialized; call init_map first");
    f(state)
}

fn compile_shader(
    compiler: &mut shaderc::Compiler,
    source: &str,
    name: &str,
    defines: &[(&str, Option<&str>)],
) -> Vec<u32> {
    let mut options = shaderc::CompileOptions::new().expect("failed creating shaderc options");
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_2 as u32,
    );
    for (key, value) in defines {
        options.add_macro_definition(key, *value);
    }
    let artifact = compiler
        .compile_into_spirv(
            source,
            shaderc::ShaderKind::Compute,
            name,
            "main",
            Some(&options),
        )
        .unwrap_or_else(|e| panic!("failed compiling {name}: {e}"));
    artifact.as_binary().to_vec()
}

fn create_shader_module(device: &Device, code: &[u32]) -> vk::ShaderModule {
    let ci = vk::ShaderModuleCreateInfo::default().code(code);
    unsafe {
        device
            .create_shader_module(&ci, None)
            .expect("failed creating shader module")
    }
}

fn create_compute_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    module: vk::ShaderModule,
) -> vk::Pipeline {
    let entry = CString::new("main").expect("invalid entry point");
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(entry.as_c_str());
    let ci = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);
    unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&ci), None)
            .expect("failed creating compute pipeline")[0]
    }
}

fn build_pipelines(context: Arc<VulkanContext>) -> Arc<VulkanComputePipelines> {
    let bindings: Vec<_> = (0..16)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let descriptor_set_layout = unsafe {
        context
            .device
            .create_descriptor_set_layout(&dsl_ci, None)
            .expect("failed creating descriptor set layout")
    };

    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(size_of::<PushConstants>() as u32);
    let pl_ci = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_set_layout))
        .push_constant_ranges(std::slice::from_ref(&push_range));
    let pipeline_layout = unsafe {
        context
            .device
            .create_pipeline_layout(&pl_ci, None)
            .expect("failed creating pipeline layout")
    };

    let mut compiler = shaderc::Compiler::new().expect("failed creating shader compiler");

    let hashmap_src = shaders::source_with_common(shaders::HASHMAP);
    let localmap_src = shaders::source_with_common(shaders::LOCALMAP);
    let reorder_src = shaders::source_with_common(shaders::REORDER);
    let planes_src = shaders::source_with_common(shaders::PLANES);
    let rle_src = shaders::source_with_common(shaders::RLE);
    let assign_indices_src = shaders::source_with_common(shaders::ASSIGN_INDICES);
    let lru_score_src = shaders::source_with_common(shaders::LRU_SCORE);

    let shader_defs = [
        ("hashmap_init", &hashmap_src, vec![("HASHMAP_INIT", None)]),
        (
            "rebuild_hashmap_keyed",
            &hashmap_src,
            vec![("REBUILD_HASHMAP_KEYED", None)],
        ),
        (
            "calculate_keys",
            &localmap_src,
            vec![("CALCULATE_KEYS", None)],
        ),
        ("reorder_points", &reorder_src, vec![]),
        ("build_voxels", &localmap_src, vec![("BUILD_VOXELS", None)]),
        ("plane_krnl", &planes_src, vec![]),
        ("rle", &rle_src, vec![]),
        ("assign_indices", &assign_indices_src, vec![]),
        ("lru_score", &lru_score_src, vec![]),
    ];

    let mut modules = Vec::with_capacity(shader_defs.len());
    let mut pipelines = Vec::with_capacity(shader_defs.len());
    for (name, source, defines) in shader_defs {
        let spirv = compile_shader(&mut compiler, source, name, &defines);
        let module = create_shader_module(&context.device, &spirv);
        let pipeline = create_compute_pipeline(&context.device, pipeline_layout, module);
        modules.push(module);
        pipelines.push(pipeline);
    }

    Arc::new(VulkanComputePipelines {
        context,
        descriptor_set_layout,
        pipeline_layout,
        hashmap_init: pipelines[0],
        rebuild_hashmap_keyed: pipelines[1],
        calculate_keys: pipelines[2],
        reorder_points: pipelines[3],
        build_voxels: pipelines[4],
        plane_krnl: pipelines[5],
        rle: pipelines[6],
        assign_indices: pipelines[7],
        lru_score: pipelines[8],
        shader_modules: modules,
    })
}

fn div_ceil_u32(v: u32, d: u32) -> u32 {
    if v == 0 { 0 } else { 1 + (v - 1) / d }
}

struct ComputeCommandSession<'a> {
    context: &'a Arc<VulkanContext>,
    command_buffer: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    next_set_idx: std::sync::atomic::AtomicUsize,
    _submit_lock: std::sync::MutexGuard<'a, ()>,
}

impl<'a> ComputeCommandSession<'a> {
    fn begin(
        context: &'a Arc<VulkanContext>,
        pipelines: &VulkanComputePipelines,
    ) -> ComputeCommandSession<'a> {
        let submit_lock = context
            .submit_lock
            .lock()
            .expect("vulkan submit mutex poisoned");

        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(context.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe {
            context
                .device
                .allocate_command_buffers(&alloc)
                .expect("failed to allocate command buffer")[0]
        };

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            context
                .device
                .begin_command_buffer(command_buffer, &begin)
                .expect("failed to begin command buffer");
        }

        // We allow up to 16 dispatches per session to avoid descriptor set overwrite bugs
        let max_dispatches = 16;
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 16 * max_dispatches,
        }];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_dispatches)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            context
                .device
                .create_descriptor_pool(&pool_ci, None)
                .expect("failed creating descriptor pool")
        };
        let set_layouts = vec![pipelines.descriptor_set_layout; max_dispatches as usize];
        let dset_alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_sets = unsafe {
            context
                .device
                .allocate_descriptor_sets(&dset_alloc)
                .expect("failed allocating descriptor sets")
        };

        ComputeCommandSession {
            context,
            command_buffer,
            descriptor_pool,
            descriptor_sets,
            next_set_idx: std::sync::atomic::AtomicUsize::new(0),
            _submit_lock: submit_lock,
        }
    }

    fn dispatch(
        &self,
        pipelines: &VulkanComputePipelines,
        pipeline: vk::Pipeline,
        buffers: &[vk::Buffer],
        params: &PushConstants,
        work_items: u32,
    ) {
        if work_items == 0 {
            return;
        }

        let set_idx = self
            .next_set_idx
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if set_idx >= self.descriptor_sets.len() {
            panic!("exceeded maximum dispatches (16) in a single ComputeCommandSession");
        }
        let descriptor_set = self.descriptor_sets[set_idx];

        let infos: Vec<_> = buffers
            .iter()
            .map(|buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(*buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe {
            self.context.device.update_descriptor_sets(&writes, &[]);

            self.context.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            self.context.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            self.context.device.cmd_push_constants(
                self.command_buffer,
                pipelines.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(params),
            );
            self.context.device.cmd_dispatch(
                self.command_buffer,
                div_ceil_u32(work_items, THREADS_PER_GROUP),
                1,
                1,
            );
        }
    }

    fn barrier(&self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ);
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    fn submit(self) {
        unsafe {
            self.context
                .device
                .end_command_buffer(self.command_buffer)
                .expect("failed to end command buffer");
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            self.context
                .device
                .queue_submit(
                    self.context.queue,
                    std::slice::from_ref(&submit),
                    vk::Fence::null(),
                )
                .expect("failed submitting compute work");
            self.context
                .device
                .queue_wait_idle(self.context.queue)
                .expect("failed waiting for compute queue");
            self.context.device.free_command_buffers(
                self.context.command_pool,
                std::slice::from_ref(&self.command_buffer),
            );
            self.context
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

pub fn init_map(
    stream: &Stream,
    capacity: usize,
    voxel_size: f32,
    sensitivity: f32,
    min_dist: f32,
    nearby_search_strategy: NeighborType,
) {
    assert!(capacity > 0, "capacity must be > 0");

    {
        let guard = global_state().lock().expect("vulkan map mutex poisoned");
        if guard.is_some() {
            return;
        }
    }

    let backend = match &stream.backend {
        StreamBackend::Vulkan(context) => {
            let dist_cap = capacity.next_power_of_two() as u32;
            let table_cap = dist_cap * 2;
            let compute = build_pipelines(Arc::clone(context));

            let mut keys_buf = VulkanBuffer::new(stream, size_of::<i32>(), table_cap as usize);
            let values_buf = VulkanBuffer::new(stream, size_of::<u32>(), table_cap as usize);
            let mut dist_buf =
                VulkanBuffer::new(stream, size_of::<GpuVoxelDist>(), dist_cap as usize);
            let mut dist_keys_buf = VulkanBuffer::new(stream, size_of::<i32>(), dist_cap as usize);
            let dummy_buf = VulkanBuffer::new(stream, size_of::<u32>(), 1);

            let sorter_ci = VrdxSorterCreateInfo {
                physical_device: context.physical_device,
                device: context.device.handle(),
                pipeline_cache: vk::PipelineCache::null(),
            };
            let mut sorter = VrdxSorter(0);
            unsafe { vrdxCreateSorter(&sorter_ci, &mut sorter) };

            let mut requirements = VrdxSorterStorageRequirements {
                size: 0,
                usage: vk::BufferUsageFlags::empty(),
            };
            unsafe { vrdxGetSorterKeyValueStorageRequirements(sorter, 1, &mut requirements) };

            let mut scratch = ScratchBuffers {
                keys: VulkanBuffer::new_host_visible(stream, size_of::<i32>(), 1),
                indices: VulkanBuffer::new_host_visible(stream, size_of::<u32>(), 1),
                packed: VulkanBuffer::new(stream, size_of::<u64>(), 1),
                run_starts: VulkanBuffer::new_host_visible(stream, size_of::<u32>(), 1),
                run_ends: VulkanBuffer::new_host_visible(stream, size_of::<u32>(), 1),
                unique_keys: VulkanBuffer::new_host_visible(stream, size_of::<i32>(), 1),
                target_indices: VulkanBuffer::new_host_visible(stream, size_of::<u32>(), 1),
                update_flags: VulkanBuffer::new_host_visible(stream, size_of::<u32>(), 1),
                sorted_points: VulkanBuffer::new(stream, size_of::<crate::types::Point>(), 1),
                counter: VulkanBuffer::new_host_visible(stream, size_of::<u32>(), 3),
                sorter_scratch: VulkanBuffer::new(stream, 1, requirements.size as usize),
                lru_mapping: VulkanBuffer::new_host_visible(
                    stream,
                    size_of::<u32>(),
                    dist_cap as usize,
                ),
                lru_keys: VulkanBuffer::new(stream, size_of::<u32>(), dist_cap as usize),
                sorter,
                capacity: 0,
            };

            let empty_keys = vec![EMPTY_KEY; table_cap as usize];
            keys_buf.upload(stream, &empty_keys);
            let zero_dists = vec![GpuVoxelDist::default(); dist_cap as usize];
            dist_buf.upload(stream, &zero_dists);
            let dist_keys_host = vec![EMPTY_KEY; dist_cap as usize];
            dist_keys_buf.upload(stream, &dist_keys_host);
            let initial_lru_mapping: Vec<u32> = (0..dist_cap).collect();
            scratch.lru_mapping.upload(stream, &initial_lru_mapping);
            scratch.counter.upload(stream, &[0u32, 0u32, 0u32]);

            let params = PushConstants {
                cap: table_cap,
                ..PushConstants::default()
            };
            let k0 = vk_buffer_handle(&keys_buf);
            let dmy = vk_buffer_handle(&dummy_buf);
            let all_init_buffers = [
                dmy, dmy, dmy, dmy, dmy, dmy, dmy, dmy, dmy, dmy, dmy, dmy, k0, dmy, dmy, dmy,
            ];
            let session = ComputeCommandSession::begin(context, &compute);
            session.dispatch(
                &compute,
                compute.hashmap_init,
                &all_init_buffers,
                &params,
                table_cap,
            );
            session.submit();

            MapBackendState::Vulkan(VulkanGpuMapState {
                context: Arc::clone(context),
                compute,
                keys_buf,
                values_buf,
                dist_buf,
                dist_keys_buf,
                dummy_buf,
                scratch,
                dist_cap,
                table_cap,
                resolution: voxel_size,
                sensitivity,
                min_dist,
                scan_number: 0,
                nbtype: nearby_search_strategy.as_u32(),
                add_cursor: 0,
                lru_cursor: 0,
            })
        }
        StreamBackend::Host(_) => {
            let map = IVoxHashMap::new(
                voxel_size,
                capacity.next_power_of_two(),
                nearby_search_strategy.into(),
                sensitivity,
            );
            MapBackendState::Host(HostMapState { map })
        }
    };

    *global_state().lock().expect("vulkan map mutex poisoned") = Some(VulkanMapState { backend });
    ensure_cleanup_registered();
}

fn vk_buffer_handle(buf: &VulkanBuffer) -> vk::Buffer {
    match &buf.storage {
        BufferStorage::Vulkan { buffer, .. } => *buffer,
        BufferStorage::Host { .. } => panic!("expected Vulkan buffer"),
    }
}

fn ensure_scratch_capacity(state: &mut VulkanGpuMapState, stream: &Stream, n_points: usize) {
    if state.scratch.capacity >= n_points {
        return;
    }
    let new_cap = (((n_points as f32) * 1.5).ceil() as usize).max(n_points.next_power_of_two());
    state.scratch.keys = VulkanBuffer::new_host_visible(stream, size_of::<i32>(), new_cap);
    state.scratch.indices = VulkanBuffer::new_host_visible(stream, size_of::<u32>(), new_cap);
    state.scratch.packed = VulkanBuffer::new(stream, size_of::<u64>(), new_cap);
    state.scratch.run_starts = VulkanBuffer::new_host_visible(stream, size_of::<u32>(), new_cap);
    state.scratch.run_ends = VulkanBuffer::new_host_visible(stream, size_of::<u32>(), new_cap);
    state.scratch.unique_keys = VulkanBuffer::new_host_visible(stream, size_of::<i32>(), new_cap);
    state.scratch.target_indices =
        VulkanBuffer::new_host_visible(stream, size_of::<u32>(), new_cap);
    state.scratch.update_flags = VulkanBuffer::new_host_visible(stream, size_of::<u32>(), new_cap);
    state.scratch.sorted_points =
        VulkanBuffer::new(stream, size_of::<crate::types::Point>(), new_cap);

    let mut requirements = VrdxSorterStorageRequirements {
        size: 0,
        usage: vk::BufferUsageFlags::empty(),
    };
    unsafe {
        vrdxGetSorterKeyValueStorageRequirements(
            state.scratch.sorter,
            new_cap as u32,
            &mut requirements,
        )
    };
    state.scratch.sorter_scratch = VulkanBuffer::new(stream, 1, requirements.size as usize);

    state.scratch.capacity = new_cap;
}

fn rebuild_gpu_lru_order(
    compute: &VulkanComputePipelines,
    scratch: &mut ScratchBuffers,
    dist_buf: &VulkanBuffer,
    scan_number: u32,
    dist_cap: u32,
    session: &mut ComputeCommandSession,
) {
    let dist_b = vk_buffer_handle(dist_buf);
    let lru_keys = vk_buffer_handle(&scratch.lru_keys);
    let lru_mapping = vk_buffer_handle(&scratch.lru_mapping);
    let sorter_scratch = vk_buffer_handle(&scratch.sorter_scratch);
    let counter_b = vk_buffer_handle(&scratch.counter);
    let dummy = vk::Buffer::null();

    let all_buffers = [
        dummy,
        dummy,
        lru_keys,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dist_b,
        counter_b,
        lru_mapping,
        dummy,
        dummy,
        dummy,
        dummy,
    ];

    let mut params = PushConstants::default();
    params.dist_cap = dist_cap;
    params.scan_number = scan_number;

    session.dispatch(compute, compute.lru_score, &all_buffers, &params, dist_cap);
    session.barrier();

    unsafe {
        vrdxCmdSortKeyValue(
            session.command_buffer,
            scratch.sorter,
            dist_cap,
            lru_keys,
            0,
            lru_mapping,
            0,
            sorter_scratch,
            0,
            vk::QueryPool::null(),
            0,
        );
    }
    session.barrier();
}

pub fn add_to_map(
    stream: &Stream,
    cloud: &VulkanBuffer,
    ekf_pose_t: &Vector3,
    ekf_pose_r: &Quaternion,
) {
    with_state(|state| match &mut state.backend {
        MapBackendState::Host(host) => {
            let cloud_cpu: Vec<PointXYZIT> = cloud.download(stream);
            let pose = nalgebra::Isometry3::from_parts(
                nalgebra::Translation3::new(ekf_pose_t.x, ekf_pose_t.y, ekf_pose_t.z),
                nalgebra::UnitQuaternion::from(*ekf_pose_r),
            );
            host.map.add(&cloud_cpu, &pose);
        }
        MapBackendState::Vulkan(state) => {
            let n_points = cloud.n;
            if n_points == 0 {
                return;
            }

            let (cloud_ctx, cloud_buf) = cloud
                .vulkan_context_and_buffer()
                .expect("expected Vulkan cloud buffer for Vulkan map update");
            assert!(
                Arc::ptr_eq(&cloud_ctx, &state.context),
                "cloud buffer and Vulkan map context mismatch"
            );

            ensure_scratch_capacity(state, stream, n_points);
            state.scan_number += 1;
            let scan_number = state.scan_number;

            let q: nalgebra::UnitQuaternion<f64> = (*ekf_pose_r).into();
            let mut params = PushConstants {
                pose_quat: [q.i as f32, q.j as f32, q.k as f32, q.w as f32],
                pose_trans: [
                    ekf_pose_t.x as f32,
                    ekf_pose_t.y as f32,
                    ekf_pose_t.z as f32,
                ],
                resolution: state.resolution,
                scan_number,
                sensitivity: state.sensitivity,
                min_dist: state.min_dist,
                unique_key_count: n_points as u32,
                table_cap: state.table_cap,
                dist_cap: state.dist_cap,
                nbtype: state.nbtype,
                n: n_points as u32,
                ..PushConstants::default()
            };

            let points_in_b = cloud_buf;
            let sorted_points_b = vk_buffer_handle(&state.scratch.sorted_points);
            let keys_b = vk_buffer_handle(&state.scratch.keys);
            let indices_b = vk_buffer_handle(&state.scratch.indices);
            let unique_keys_b = vk_buffer_handle(&state.scratch.unique_keys);
            let target_indices_b = vk_buffer_handle(&state.scratch.target_indices);
            let run_starts_b = vk_buffer_handle(&state.scratch.run_starts);
            let run_ends_b = vk_buffer_handle(&state.scratch.run_ends);
            let update_flags_b = vk_buffer_handle(&state.scratch.update_flags);
            let dists_b = vk_buffer_handle(&state.dist_buf);
            let counters_b = vk_buffer_handle(&state.scratch.counter);
            let lru_mapping_b = vk_buffer_handle(&state.scratch.lru_mapping);
            let table_keys_b = vk_buffer_handle(&state.keys_buf);
            let table_vals_b = vk_buffer_handle(&state.values_buf);
            let dist_keys_b = vk_buffer_handle(&state.dist_keys_buf);
            let dummy = vk_buffer_handle(&state.dummy_buf);

            let all_buffers = [
                points_in_b,
                sorted_points_b,
                keys_b,
                indices_b,
                unique_keys_b,
                target_indices_b,
                run_starts_b,
                run_ends_b,
                update_flags_b,
                dists_b,
                counters_b,
                lru_mapping_b,
                table_keys_b,
                table_vals_b,
                dist_keys_b,
                dummy,
            ];

            let mut session = ComputeCommandSession::begin(&state.context, &state.compute);

            // 1. Calculate Keys
            session.dispatch(
                &state.compute,
                state.compute.calculate_keys,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            // 2. GPU Sort
            let sorter_scratch = vk_buffer_handle(&state.scratch.sorter_scratch);
            unsafe {
                vrdxCmdSortKeyValue(
                    session.command_buffer,
                    state.scratch.sorter,
                    n_points as u32,
                    keys_b,
                    0,
                    indices_b,
                    0,
                    sorter_scratch,
                    0,
                    vk::QueryPool::null(),
                    0,
                );
            }
            session.barrier();

            // 3. GPU Run-Length Encoding
            let counters_host = [0u32, state.add_cursor, state.lru_cursor];
            state.scratch.counter.upload(stream, &counters_host);

            session.dispatch(
                &state.compute,
                state.compute.rle,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            // 4. Assign Indices
            session.dispatch(
                &state.compute,
                state.compute.assign_indices,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            // 5. Reorder Points
            session.dispatch(
                &state.compute,
                state.compute.reorder_points,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            // 6. Build Voxels
            session.dispatch(
                &state.compute,
                state.compute.build_voxels,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            // 7. Rebuild Hashmap
            params.cap = state.table_cap;
            session.dispatch(
                &state.compute,
                state.compute.hashmap_init,
                &all_buffers,
                &params,
                state.table_cap,
            );
            session.barrier();

            params.table_cap = state.table_cap;
            session.dispatch(
                &state.compute,
                state.compute.rebuild_hashmap_keyed,
                &all_buffers,
                &params,
                state.dist_cap,
            );
            session.barrier();

            if state.add_cursor >= state.dist_cap {
                rebuild_gpu_lru_order(
                    &state.compute,
                    &mut state.scratch,
                    &state.dist_buf,
                    state.scan_number,
                    state.dist_cap,
                    &mut session,
                );
            }

            session.submit();

            // Download counters to update host-side state
            let counters_result: Vec<u32> = state.scratch.counter.download(stream);
            state.add_cursor = counters_result[1].min(state.dist_cap);
            state.lru_cursor = counters_result[2].min(state.dist_cap);
        }
    });
}

pub fn get_map() -> Vec<nalgebra::Point3<f32>> {
    with_state(|state| match &mut state.backend {
        MapBackendState::Host(host) => host.map.get_entire_map(),
        MapBackendState::Vulkan(gpu) => {
            let stream = Stream {
                backend: StreamBackend::Vulkan(Arc::clone(&gpu.context)),
            };
            let dists: Vec<GpuVoxelDist> = gpu.dist_buf.download(&stream);
            let valid = (gpu.add_cursor as usize).min(gpu.dist_cap as usize);
            dists[..valid]
                .iter()
                .filter(|d| d.point_count > 0)
                .map(|d| nalgebra::Point3::new(d.mean[0], d.mean[1], d.mean[2]))
                .collect()
        }
    })
}

pub fn get_map_size() -> u32 {
    with_state(|state| match &mut state.backend {
        MapBackendState::Host(host) => host.map.get_entire_map().len() as u32,
        MapBackendState::Vulkan(gpu) => gpu.add_cursor.min(gpu.dist_cap),
    })
}

pub fn cleanup_map() {
    *global_state().lock().expect("vulkan map mutex poisoned") = None;
}

pub fn planes_vulkan(
    stream: &Stream,
    cloud: &VulkanBuffer,
    ekf_pose_t: &Vector3,
    ekf_pose_r: &Quaternion,
    ekf_imu_offset_t: &Vector3,
    ekf_imu_offset_r: &Quaternion,
    out_buff: &mut Vec<PointXYZINormal>,
    out_buff_gpu: &mut VulkanBuffer,
) {
    assert!(
        out_buff.len() >= cloud.n,
        "output buffer must fit cloud size"
    );
    with_state(|state| match &mut state.backend {
        MapBackendState::Host(host) => {
            let cloud_cpu: Vec<PointXYZIT> = cloud.download(stream);
            let pose = nalgebra::Isometry3::from_parts(
                nalgebra::Translation3::new(ekf_pose_t.x, ekf_pose_t.y, ekf_pose_t.z),
                nalgebra::UnitQuaternion::from(*ekf_pose_r),
            );
            let lidar_to_imu = nalgebra::Isometry3::from_parts(
                nalgebra::Translation3::new(
                    ekf_imu_offset_t.x,
                    ekf_imu_offset_t.y,
                    ekf_imu_offset_t.z,
                ),
                nalgebra::UnitQuaternion::from(*ekf_imu_offset_r),
            );

            let mut planes = Vec::<PointXYZINormal>::with_capacity(cloud_cpu.len());
            for point_local in &cloud_cpu {
                let p_l = nalgebra::Point3::new(point_local.x, point_local.y, point_local.z);
                let p_imu = lidar_to_imu
                    .transform_point(&p_l.cast::<f64>())
                    .cast::<f32>();
                let p_world = pose.transform_point(&p_imu.cast::<f64>()).cast::<f32>();
                let query = ros_pointcloud2::points::PointXYZ::new(p_world.x, p_world.y, p_world.z);
                if let Some(plane) = host.map.plane_fit_from_distributions(&query) {
                    let pd2 = plane[0] * p_world.x
                        + plane[1] * p_world.y
                        + plane[2] * p_world.z
                        + plane[3];
                    let l2_sq = point_local.x * point_local.x
                        + point_local.y * point_local.y
                        + point_local.z * point_local.z;
                    let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
                    if s > 0.9 {
                        planes.push(PointXYZINormal::new(
                            point_local.x,
                            point_local.y,
                            point_local.z,
                            pd2,
                            plane[0],
                            plane[1],
                            plane[2],
                        ));
                    }
                }
            }
            let capped = planes.len().min(out_buff.len());
            out_buff.clear();
            out_buff.extend_from_slice(&planes[..capped]);
            out_buff_gpu.upload(stream, &planes[..capped]);
        }
        MapBackendState::Vulkan(state) => {
            let (cloud_ctx, cloud_buf) = cloud
                .vulkan_context_and_buffer()
                .expect("expected Vulkan cloud buffer for Vulkan planes");
            log::debug!("vulkan planes start: points={}", cloud.n);
            assert!(
                Arc::ptr_eq(&cloud_ctx, &state.context),
                "cloud buffer and Vulkan map context mismatch"
            );
            assert!(
                out_buff_gpu.capacity() >= cloud.n,
                "GPU output buffer must fit cloud size"
            );

            state.scratch.counter.upload(stream, &[0u32]);
            let ext_q: nalgebra::UnitQuaternion<f64> = (*ekf_imu_offset_r).into();
            let pose_q: nalgebra::UnitQuaternion<f64> = (*ekf_pose_r).into();
            let params = PushConstants {
                lidar_to_imu_quat: [
                    ext_q.i as f32,
                    ext_q.j as f32,
                    ext_q.k as f32,
                    ext_q.w as f32,
                ],
                lidar_to_imu_pos: [
                    ekf_imu_offset_t.x as f32,
                    ekf_imu_offset_t.y as f32,
                    ekf_imu_offset_t.z as f32,
                ],
                pose_quat: [
                    pose_q.i as f32,
                    pose_q.j as f32,
                    pose_q.k as f32,
                    pose_q.w as f32,
                ],
                pose_trans: [
                    ekf_pose_t.x as f32,
                    ekf_pose_t.y as f32,
                    ekf_pose_t.z as f32,
                ],
                resolution: state.resolution,
                table_cap: state.table_cap,
                unique_key_count: cloud.n as u32,
                nbtype: state.nbtype,
                sphericity_threshold: 0.5,
                ..PushConstants::default()
            };

            let points_in_b = cloud_buf;
            let table_keys_b = vk_buffer_handle(&state.keys_buf);
            let table_vals_b = vk_buffer_handle(&state.values_buf);
            let dists_b = vk_buffer_handle(&state.dist_buf);
            let counter_b = vk_buffer_handle(&state.scratch.counter);
            let out_gpu_b = vk_buffer_handle(out_buff_gpu);
            let dummy = vk_buffer_handle(&state.dummy_buf);

            let all_buffers = [
                points_in_b,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dists_b,
                counter_b,
                dummy,
                table_keys_b,
                table_vals_b,
                dummy,
                out_gpu_b,
            ];

            let session = ComputeCommandSession::begin(&state.context, &state.compute);
            session.dispatch(
                &state.compute,
                state.compute.plane_krnl,
                &all_buffers,
                &params,
                cloud.n as u32,
            );
            session.submit();
            log::debug!("vulkan planes: kernel complete, downloading counter");

            state.scratch.counter.n = 1;
            let out_size = state.scratch.counter.download::<u32>(stream)[0] as usize;
            assert!(
                out_size <= cloud.n,
                "plane kernel produced out_size={} for cloud size={}",
                out_size,
                cloud.n
            );
            log::debug!("vulkan planes done: out_size={}", out_size);
            let capped = out_size.min(out_buff.len());

            out_buff_gpu.n = capped;
            let mut gpu_normals = vec![crate::types::PointNormal::default(); capped];
            out_buff_gpu.download_into(stream, &mut gpu_normals);

            out_buff.clear();
            out_buff.extend(gpu_normals.into_iter().map(PointXYZINormal::from));
        }
    });
}

fn ensure_cleanup_registered() {
    // Avoid registering atexit cleanup for Vulkan map state.
    // On some macOS setups this can run during a fragile late shutdown phase
    // and trigger bus errors while tearing down driver resources.
}

impl From<NeighborType> for IVoxNearbyType {
    fn from(value: NeighborType) -> Self {
        match value {
            NeighborType::Center => IVoxNearbyType::Center,
            NeighborType::Neighbor6 => IVoxNearbyType::Nearby6,
            NeighborType::Neighbor18 => IVoxNearbyType::Nearby18,
            NeighborType::Neighbor26 => IVoxNearbyType::Nearby26,
        }
    }
}

static ALLOCATED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
                allocation_size,
                ..
            } => {
                unsafe {
                    context.device.destroy_buffer(*buffer, None);
                    context.device.free_memory(*memory, None);
                }
                decrement_allocation(*allocation_size);
            }
            BufferStorage::Host { data, .. } => {
                decrement_allocation(data.len() as u64);
            }
        }
    }
}
