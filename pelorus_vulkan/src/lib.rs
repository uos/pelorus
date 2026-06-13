#![cfg(vulkan_available)]

use ash::{Device, Entry, Instance, khr::push_descriptor, vk};
use bytemuck::{Pod, Zeroable};
use pelorus_core::PointXYZIT;
use ros_pointcloud2::points::PointXYZINormal;
use spole::{IVoxHashMap, IVoxNearbyType};
use std::ffi::{CStr, CString};
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

            let device_extensions = unsafe {
                instance
                    .enumerate_device_extension_properties(physical)
                    .map_err(|e| format!("failed to enumerate Vulkan device extensions: {e}"))?
            };
            let has_push_descriptor = device_extensions.iter().any(|ext| unsafe {
                CStr::from_ptr(ext.extension_name.as_ptr()) == push_descriptor::NAME
            });
            if !has_push_descriptor {
                continue;
            }

            let mut push_desc_props = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
            let mut props2 =
                vk::PhysicalDeviceProperties2::default().push_next(&mut push_desc_props);
            unsafe {
                instance.get_physical_device_properties2(physical, &mut props2);
            }
            if push_desc_props.max_push_descriptors == 0 {
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
            return Err(
                "no Vulkan device with compute + bufferDeviceAddress + VK_KHR_push_descriptor support available"
                    .to_string(),
            );
        };

        let queue_priorities = [1.0_f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .scalar_block_layout(true);
        let enabled_device_extensions = [push_descriptor::NAME.as_ptr()];
        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_ci))
            .enabled_extension_names(&enabled_device_extensions)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl VulkanVersion {
    fn from_vk_encoded(version: u32) -> Self {
        Self {
            major: vk::api_version_major(version),
            minor: vk::api_version_minor(version),
            patch: vk::api_version_patch(version),
        }
    }
}

impl std::fmt::Display for VulkanVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanMetadata {
    pub device_name: String,
    pub api_version: VulkanVersion,
    pub driver_version: VulkanVersion,
    pub driver_version_raw: u32,
}

pub fn is_available() -> bool {
    #[cfg(not(feature = "vulkan_enable"))]
    {
        return false;
    }

    #[cfg(feature = "vulkan_enable")]
    GLOBAL_VULKAN_CONTEXT
        .get_or_init(|| VulkanContext::try_new().map(Arc::new))
        .is_ok()
}

pub fn get_vulkan_metadata() -> Option<VulkanMetadata> {
    let context = GLOBAL_VULKAN_CONTEXT
        .get_or_init(|| VulkanContext::try_new().map(Arc::new))
        .as_ref()
        .ok()?;

    let properties = unsafe {
        context
            .instance
            .get_physical_device_properties(context.physical_device)
    };
    let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    Some(VulkanMetadata {
        device_name,
        api_version: VulkanVersion::from_vk_encoded(properties.api_version),
        driver_version: VulkanVersion::from_vk_encoded(properties.driver_version),
        driver_version_raw: properties.driver_version,
    })
}

impl Stream {
    #[cfg(feature = "vulkan_enable")]
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

struct BufferAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_flags: vk::MemoryPropertyFlags,
    allocation_size: vk::DeviceSize,
}

fn allocate_buffer_required(
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
        &[
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        ],
    )
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

enum BufferStorage {
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

    fn vulkan_context_and_buffer(&self) -> Option<(Arc<VulkanContext>, vk::Buffer)> {
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
}

const EMPTY_KEY: i32 = i32::MIN;
const THREADS_PER_GROUP: u32 = 256;
const DESCRIPTOR_BINDING_COUNT: u32 = 18;
const MAX_DISPATCHES_PER_SESSION: u32 = 1024;

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
    compact_planes: vk::Pipeline,
    plane_count: vk::Pipeline,
    portable_sort: vk::Pipeline,
    rle: vk::Pipeline,
    assign_indices: vk::Pipeline,
    lru_score: vk::Pipeline,
    classify_keys: vk::Pipeline,
    exclusive_scan: vk::Pipeline,
    assign_new_targets: vk::Pipeline,
    finalize_cursors: vk::Pipeline,
    scan_block_sums: vk::Pipeline,
    add_block_prefix: vk::Pipeline,
    downsample_build_keys: vk::Pipeline,
    downsample_medoids: vk::Pipeline,
    downsample_timestamp_keys: vk::Pipeline,
    downsample_reorder: vk::Pipeline,
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
            device.destroy_pipeline(self.compact_planes, None);
            device.destroy_pipeline(self.plane_count, None);
            device.destroy_pipeline(self.portable_sort, None);
            device.destroy_pipeline(self.rle, None);
            device.destroy_pipeline(self.assign_indices, None);
            device.destroy_pipeline(self.lru_score, None);
            device.destroy_pipeline(self.classify_keys, None);
            device.destroy_pipeline(self.exclusive_scan, None);
            device.destroy_pipeline(self.assign_new_targets, None);
            device.destroy_pipeline(self.finalize_cursors, None);
            device.destroy_pipeline(self.scan_block_sums, None);
            device.destroy_pipeline(self.add_block_prefix, None);
            device.destroy_pipeline(self.downsample_build_keys, None);
            device.destroy_pipeline(self.downsample_medoids, None);
            device.destroy_pipeline(self.downsample_timestamp_keys, None);
            device.destroy_pipeline(self.downsample_reorder, None);
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
    is_new: VulkanBuffer,
    new_offset: VulkanBuffer,
    block_sums: VulkanBuffer,
    plane_compact: VulkanBuffer,
    plane_readback: VulkanBuffer,
    sorter: VrdxSorter,
    capacity: usize,
}

struct DownsampleScratchBuffers {
    input_upload: VulkanBuffer,
    input: VulkanBuffer,
    keys: VulkanBuffer,
    indices: VulkanBuffer,
    selected: VulkanBuffer,
    medoids: VulkanBuffer,
    output_download: VulkanBuffer,
    run_starts: VulkanBuffer,
    run_ends: VulkanBuffer,
    unique_keys: VulkanBuffer,
    start_flags: VulkanBuffer,
    start_offsets: VulkanBuffer,
    block_sums: VulkanBuffer,
    counter_upload: VulkanBuffer,
    counter: VulkanBuffer,
    counter_download: VulkanBuffer,
    sorter_scratch: VulkanBuffer,
    sorter: VrdxSorter,
    capacity: usize,
}

struct VulkanDownsampleState {
    context: Arc<VulkanContext>,
    compute: Arc<VulkanComputePipelines>,
    command_resources: Mutex<ComputeCommandResources>,
    dummy_buf: VulkanBuffer,
    scratch: DownsampleScratchBuffers,
    timestamp_pool: vk::QueryPool,
    timestamp_period: f32,
}

impl Drop for VulkanDownsampleState {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_query_pool(self.timestamp_pool, None);
        }
    }
}

static DOWNSAMPLE_STATE: OnceLock<Mutex<Option<VulkanDownsampleState>>> = OnceLock::new();

struct VulkanGpuMapState {
    context: Arc<VulkanContext>,
    compute: Arc<VulkanComputePipelines>,
    command_resources: Mutex<ComputeCommandResources>,
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
    timestamp_pool: vk::QueryPool,
    timestamp_period: f32,
    pending_planes_timing: std::sync::atomic::AtomicBool,
    pending_add_to_map_timing: std::sync::atomic::AtomicBool,
}

impl Drop for VulkanGpuMapState {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_query_pool(self.timestamp_pool, None);
        }
    }
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

fn requires_portable_map_sort(context: &VulkanContext) -> bool {
    let properties = unsafe {
        context
            .instance
            .get_physical_device_properties(context.physical_device)
    };
    properties.vendor_id == 0x8086
        && properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
}

#[cfg(feature = "vulkan_enable")]
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

#[cfg(feature = "vulkan_enable")]
fn build_pipelines(context: Arc<VulkanContext>) -> Arc<VulkanComputePipelines> {
    let bindings: Vec<_> = (0..DESCRIPTOR_BINDING_COUNT)
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
    let compact_planes_src = shaders::source_with_common(shaders::COMPACT_PLANES);
    let plane_count_src = shaders::source_with_common(shaders::PLANE_COUNT);
    let portable_sort_src = shaders::source_with_common(shaders::PORTABLE_SORT);
    let rle_src = shaders::source_with_common(shaders::RLE);
    let assign_indices_src = shaders::source_with_common(shaders::ASSIGN_INDICES);
    let lru_score_src = shaders::source_with_common(shaders::LRU_SCORE);
    let classify_keys_src = shaders::source_with_common(shaders::CLASSIFY_KEYS);
    let exclusive_scan_src = shaders::source_with_common(shaders::EXCLUSIVE_SCAN);
    let assign_new_targets_src = shaders::source_with_common(shaders::ASSIGN_NEW_TARGETS);
    let finalize_cursors_src = shaders::source_with_common(shaders::FINALIZE_CURSORS);
    let scan_block_sums_src = shaders::source_with_common(shaders::SCAN_BLOCK_SUMS);
    let add_block_prefix_src = shaders::source_with_common(shaders::ADD_BLOCK_PREFIX);
    let downsample_src = shaders::source_with_common(shaders::DOWNSAMPLE);
    let downsample_reorder_src = shaders::source_with_common(shaders::DOWNSAMPLE_REORDER);

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
        ("compact_planes", &compact_planes_src, vec![]),
        ("plane_count", &plane_count_src, vec![]),
        ("portable_sort", &portable_sort_src, vec![]),
        ("rle", &rle_src, vec![]),
        ("assign_indices", &assign_indices_src, vec![]),
        ("lru_score", &lru_score_src, vec![]),
        ("classify_keys", &classify_keys_src, vec![]),
        ("exclusive_scan", &exclusive_scan_src, vec![]),
        ("assign_new_targets", &assign_new_targets_src, vec![]),
        ("finalize_cursors", &finalize_cursors_src, vec![]),
        ("scan_block_sums", &scan_block_sums_src, vec![]),
        ("add_block_prefix", &add_block_prefix_src, vec![]),
        (
            "downsample_build_keys",
            &downsample_src,
            vec![("BUILD_KEYS", None)],
        ),
        (
            "downsample_medoids",
            &downsample_src,
            vec![("MEDOIDS", None)],
        ),
        (
            "downsample_timestamp_keys",
            &downsample_src,
            vec![("TIMESTAMP_KEYS", None)],
        ),
        ("downsample_reorder", &downsample_reorder_src, vec![]),
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
        compact_planes: pipelines[6],
        plane_count: pipelines[7],
        portable_sort: pipelines[8],
        rle: pipelines[9],
        assign_indices: pipelines[10],
        lru_score: pipelines[11],
        classify_keys: pipelines[12],
        exclusive_scan: pipelines[13],
        assign_new_targets: pipelines[14],
        finalize_cursors: pipelines[15],
        scan_block_sums: pipelines[16],
        add_block_prefix: pipelines[17],
        downsample_build_keys: pipelines[18],
        downsample_medoids: pipelines[19],
        downsample_timestamp_keys: pipelines[20],
        downsample_reorder: pipelines[21],
        shader_modules: modules,
    })
}

#[cfg(not(feature = "vulkan_enable"))]
fn build_pipelines(_context: Arc<VulkanContext>) -> Arc<VulkanComputePipelines> {
    panic!("vulkan pipeline initialization requires feature `vulkan_enable`")
}

fn div_ceil_u32(v: u32, d: u32) -> u32 {
    if v == 0 { 0 } else { 1 + (v - 1) / d }
}

struct ComputeCommandResources {
    context: Arc<VulkanContext>,
    command_buffer: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl ComputeCommandResources {
    fn new(context: &Arc<VulkanContext>, pipelines: &VulkanComputePipelines) -> Self {
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

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: DESCRIPTOR_BINDING_COUNT * MAX_DISPATCHES_PER_SESSION,
        }];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(MAX_DISPATCHES_PER_SESSION)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            context
                .device
                .create_descriptor_pool(&pool_ci, None)
                .expect("failed creating descriptor pool")
        };
        let set_layouts =
            vec![pipelines.descriptor_set_layout; MAX_DISPATCHES_PER_SESSION as usize];
        let dset_alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_sets = unsafe {
            context
                .device
                .allocate_descriptor_sets(&dset_alloc)
                .expect("failed allocating descriptor sets")
        };

        Self {
            context: Arc::clone(context),
            command_buffer,
            descriptor_pool,
            descriptor_sets,
        }
    }
}

impl Drop for ComputeCommandResources {
    fn drop(&mut self) {
        unsafe {
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

struct ComputeCommandSession<'a> {
    context: &'a Arc<VulkanContext>,
    resources: std::sync::MutexGuard<'a, ComputeCommandResources>,
    next_set_idx: usize,
    dummy_buffer: vk::Buffer,
    _submit_lock: std::sync::MutexGuard<'a, ()>,
}

impl<'a> ComputeCommandSession<'a> {
    fn begin(
        context: &'a Arc<VulkanContext>,
        resources: &'a Mutex<ComputeCommandResources>,
        dummy_buffer: vk::Buffer,
    ) -> ComputeCommandSession<'a> {
        let submit_lock = context
            .submit_lock
            .lock()
            .expect("vulkan submit mutex poisoned");
        let resources = resources
            .lock()
            .expect("vulkan compute resources mutex poisoned");
        assert!(
            Arc::ptr_eq(context, &resources.context),
            "compute resources and Vulkan context mismatch"
        );

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            context
                .device
                .reset_command_buffer(
                    resources.command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .expect("failed to reset command buffer");
            context
                .device
                .begin_command_buffer(resources.command_buffer, &begin)
                .expect("failed to begin command buffer");
        }

        ComputeCommandSession {
            context,
            resources,
            next_set_idx: 0,
            dummy_buffer,
            _submit_lock: submit_lock,
        }
    }

    fn dispatch(
        &mut self,
        pipelines: &VulkanComputePipelines,
        pipeline: vk::Pipeline,
        buffers: &[vk::Buffer],
        params: &PushConstants,
        work_items: u32,
    ) {
        if work_items == 0 {
            return;
        }
        assert!(
            buffers.len() <= DESCRIPTOR_BINDING_COUNT as usize,
            "too many buffers for Vulkan descriptor layout"
        );

        let set_idx = self.next_set_idx;
        self.next_set_idx += 1;
        if set_idx >= self.resources.descriptor_sets.len() {
            panic!(
                "exceeded maximum dispatches ({}) in a single ComputeCommandSession",
                self.resources.descriptor_sets.len()
            );
        }
        let descriptor_set = self.resources.descriptor_sets[set_idx];

        let infos: Vec<_> = (0..DESCRIPTOR_BINDING_COUNT as usize)
            .map(|idx| {
                let buffer = buffers.get(idx).copied().unwrap_or(self.dummy_buffer);
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
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
                self.resources.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            self.context.device.cmd_bind_descriptor_sets(
                self.resources.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            self.context.device.cmd_push_constants(
                self.resources.command_buffer,
                pipelines.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(params),
            );
            self.context.device.cmd_dispatch(
                self.resources.command_buffer,
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
                self.resources.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    fn shader_to_transfer_barrier(&self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.resources.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    fn transfer_to_host_barrier(&self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.resources.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    fn transfer_to_shader_barrier(&self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.resources.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    fn copy_buffer(&self, src: vk::Buffer, dst: vk::Buffer, size: vk::DeviceSize) {
        if size == 0 {
            return;
        }
        let copy_region = vk::BufferCopy::default().size(size);
        unsafe {
            self.context.device.cmd_copy_buffer(
                self.resources.command_buffer,
                src,
                dst,
                std::slice::from_ref(&copy_region),
            );
        }
    }

    fn write_timestamp(&self, timestamp_pool: vk::QueryPool, query_index: u32) {
        unsafe {
            self.context.device.cmd_write_timestamp(
                self.resources.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                timestamp_pool,
                query_index,
            );
        }
    }

    fn command_buffer(&self) -> vk::CommandBuffer {
        self.resources.command_buffer
    }

    fn submit(self) {
        unsafe {
            self.context
                .device
                .end_command_buffer(self.resources.command_buffer)
                .expect("failed to end command buffer");
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.resources.command_buffer));
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
        }
    }
}

fn record_portable_sort(
    session: &mut ComputeCommandSession,
    compute: &VulkanComputePipelines,
    dummy: vk::Buffer,
    keys: vk::Buffer,
    values: vk::Buffer,
    count: u32,
    capacity: u32,
) {
    assert!(capacity.is_power_of_two());
    assert!(count <= capacity);

    let buffers = [
        dummy, dummy, keys, values, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy,
        dummy, dummy, dummy, dummy, dummy,
    ];
    let mut params = PushConstants {
        n: count,
        cap: capacity,
        k: 0,
        ..PushConstants::default()
    };
    session.dispatch(compute, compute.portable_sort, &buffers, &params, capacity);
    session.barrier();

    let mut sequence_size = 2;
    while sequence_size <= capacity {
        let mut comparison_stride = sequence_size / 2;
        while comparison_stride > 0 {
            params.j = comparison_stride;
            params.k = sequence_size;
            session.dispatch(compute, compute.portable_sort, &buffers, &params, capacity);
            session.barrier();
            comparison_stride /= 2;
        }
        sequence_size *= 2;
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
    if cfg!(target_os = "macos") && std::env::var_os("PELORUS_VULKAN_ALLOW_MACOS_UNSAFE").is_none()
    {
        panic!(
            "Vulkan radix-sort path is not safely supported on macOS/MoltenVK in this build. \
             Refusing to run Vulkan backend to avoid segmentation faults. \
             Set PELORUS_VULKAN_ALLOW_MACOS_UNSAFE=1 to bypass at your own risk."
        );
    }

    {
        let guard = global_state().lock().expect("vulkan map mutex poisoned");
        if guard.is_some() {
            return;
        }
    }

    let backend = match &stream.backend {
        StreamBackend::Vulkan(context) => {
            log::info!(
                "Vulkan strict memory mode enabled: heavy map/scratch buffers require DEVICE_LOCAL memory; host-visible fallback is disabled for these buffers"
            );
            let dist_cap = capacity as u32;
            let table_cap = dist_cap
                .checked_mul(2)
                .expect("Vulkan map capacity is too large")
                .next_power_of_two();
            let compute = build_pipelines(Arc::clone(context));

            let mut keys_buf = VulkanBuffer::new(stream, size_of::<i32>(), table_cap as usize);
            let values_buf = VulkanBuffer::new(stream, size_of::<u32>(), table_cap as usize);
            let mut dist_buf =
                VulkanBuffer::new(stream, size_of::<GpuVoxelDist>(), dist_cap as usize);
            let mut dist_keys_buf = VulkanBuffer::new(stream, size_of::<i32>(), dist_cap as usize);
            let dummy_buf = VulkanBuffer::new(stream, size_of::<u32>(), 1);
            let command_resources = Mutex::new(ComputeCommandResources::new(context, &compute));

            let sorter_ci = VrdxSorterCreateInfo {
                physical_device: context.physical_device,
                device: context.device.handle(),
                pipeline_cache: vk::PipelineCache::null(),
            };
            let mut sorter = VrdxSorter(0);
            unsafe { vrdxCreateSorter(&sorter_ci, &mut sorter) };
            assert!(
                sorter.0 != 0,
                "vk_radix_sort initialization failed: sorter handle is null. \
                 This backend requires VK_KHR_push_descriptor support and a compatible Vulkan implementation."
            );

            let mut requirements = VrdxSorterStorageRequirements {
                size: 0,
                usage: vk::BufferUsageFlags::empty(),
            };
            unsafe {
                vrdxGetSorterKeyValueStorageRequirements(sorter, dist_cap, &mut requirements)
            };
            assert!(
                requirements.size > 0
                    && requirements
                        .usage
                        .contains(vk::BufferUsageFlags::STORAGE_BUFFER),
                "vk_radix_sort reported invalid storage requirements. \
                 Refusing to continue to avoid undefined behavior."
            );

            let mut scratch = ScratchBuffers {
                keys: VulkanBuffer::new(stream, size_of::<i32>(), 1),
                indices: VulkanBuffer::new(stream, size_of::<u32>(), 1),
                packed: VulkanBuffer::new(stream, size_of::<u64>(), 1),
                run_starts: VulkanBuffer::new(stream, size_of::<u32>(), 1),
                run_ends: VulkanBuffer::new(stream, size_of::<u32>(), 1),
                unique_keys: VulkanBuffer::new(stream, size_of::<i32>(), 1),
                target_indices: VulkanBuffer::new(stream, size_of::<u32>(), 1),
                update_flags: VulkanBuffer::new(stream, size_of::<u32>(), 1),
                sorted_points: VulkanBuffer::new(stream, size_of::<crate::types::Point>(), 1),
                // Keep tiny counters host-visible for frequent scalar readback/writes.
                counter: VulkanBuffer::new_host_visible(stream, size_of::<u32>(), 3),
                sorter_scratch: VulkanBuffer::new(stream, 1, requirements.size as usize),
                // This buffer is hot in map-full replacement and LRU sort; keep it device-local.
                lru_mapping: VulkanBuffer::new(
                    stream,
                    size_of::<u32>(),
                    (dist_cap as usize).next_power_of_two(),
                ),
                lru_keys: VulkanBuffer::new(
                    stream,
                    size_of::<u32>(),
                    (dist_cap as usize).next_power_of_two(),
                ),
                is_new: VulkanBuffer::new(stream, size_of::<u32>(), 1),
                new_offset: VulkanBuffer::new(stream, size_of::<u32>(), 1),
                block_sums: VulkanBuffer::new(stream, size_of::<u32>(), 1),
                plane_compact: VulkanBuffer::new(stream, size_of::<crate::types::PointNormal>(), 1),
                plane_readback: VulkanBuffer::new_host_visible(
                    stream,
                    size_of::<crate::types::PointNormal>(),
                    1,
                ),
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
            let mut session = ComputeCommandSession::begin(context, &command_resources, dmy);
            session.dispatch(
                &compute,
                compute.hashmap_init,
                &all_init_buffers,
                &params,
                table_cap,
            );
            session.submit();

            // Create timestamp query pool for GPU timing measurements
            // Query indices: 0-1 for planes, 2-3 for add_to_map
            let query_pool_ci = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(4);
            let timestamp_pool = unsafe {
                context
                    .device
                    .create_query_pool(&query_pool_ci, None)
                    .expect("failed to create timestamp query pool")
            };

            // Get timestamp period (nanoseconds per timestamp tick)
            let props = unsafe {
                context
                    .instance
                    .get_physical_device_properties(context.physical_device)
            };
            let timestamp_period = props.limits.timestamp_period;

            MapBackendState::Vulkan(VulkanGpuMapState {
                context: Arc::clone(context),
                compute,
                command_resources,
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
                timestamp_pool,
                timestamp_period,
                pending_planes_timing: std::sync::atomic::AtomicBool::new(false),
                pending_add_to_map_timing: std::sync::atomic::AtomicBool::new(false),
            })
        }
        StreamBackend::Host(_) => {
            log::warn!(
                "Vulkan map initialized in host fallback mode; backend equivalence with CUDA is not guaranteed"
            );
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

fn downsample_state() -> &'static Mutex<Option<VulkanDownsampleState>> {
    DOWNSAMPLE_STATE.get_or_init(|| Mutex::new(None))
}

fn ensure_downsample_state(stream: &Stream, capacity: usize) {
    if cfg!(target_os = "macos") && std::env::var_os("PELORUS_VULKAN_ALLOW_MACOS_UNSAFE").is_none()
    {
        panic!(
            "Vulkan radix-sort path is not safely supported on macOS/MoltenVK in this build. \
             Refusing to run Vulkan downsampling to avoid segmentation faults. \
             Set PELORUS_VULKAN_ALLOW_MACOS_UNSAFE=1 to bypass at your own risk."
        );
    }

    let mut guard = downsample_state()
        .lock()
        .expect("vulkan downsample mutex poisoned");
    let needs_init = guard
        .as_ref()
        .is_none_or(|state| state.scratch.capacity < capacity);
    if !needs_init {
        return;
    }

    let StreamBackend::Vulkan(context) = &stream.backend else {
        return;
    };

    let new_cap = capacity.next_power_of_two().max(1);
    let compute = build_pipelines(Arc::clone(context));
    let command_resources = Mutex::new(ComputeCommandResources::new(context, &compute));
    let dummy_buf = VulkanBuffer::new(stream, size_of::<u32>(), 1);

    let sorter_ci = VrdxSorterCreateInfo {
        physical_device: context.physical_device,
        device: context.device.handle(),
        pipeline_cache: vk::PipelineCache::null(),
    };
    let mut sorter = VrdxSorter(0);
    unsafe { vrdxCreateSorter(&sorter_ci, &mut sorter) };
    assert!(
        sorter.0 != 0,
        "vk_radix_sort initialization failed for Vulkan downsampling"
    );

    let mut requirements = VrdxSorterStorageRequirements {
        size: 0,
        usage: vk::BufferUsageFlags::empty(),
    };
    unsafe { vrdxGetSorterKeyValueStorageRequirements(sorter, new_cap as u32, &mut requirements) };

    let block_sums_cap = (new_cap / 512 + 1).next_power_of_two().max(1024);
    let scratch = DownsampleScratchBuffers {
        input_upload: VulkanBuffer::new_host_visible(stream, size_of::<PointXYZIT>(), new_cap),
        input: VulkanBuffer::new(stream, size_of::<PointXYZIT>(), new_cap),
        keys: VulkanBuffer::new(stream, size_of::<u32>(), new_cap),
        indices: VulkanBuffer::new(stream, size_of::<u32>(), new_cap),
        selected: VulkanBuffer::new(stream, size_of::<PointXYZIT>(), new_cap),
        medoids: VulkanBuffer::new(stream, size_of::<PointXYZIT>(), new_cap),
        output_download: VulkanBuffer::new_host_visible(stream, size_of::<PointXYZIT>(), new_cap),
        run_starts: VulkanBuffer::new(stream, size_of::<u32>(), new_cap),
        run_ends: VulkanBuffer::new(stream, size_of::<u32>(), new_cap),
        unique_keys: VulkanBuffer::new(stream, size_of::<i32>(), new_cap),
        start_flags: VulkanBuffer::new(stream, size_of::<u32>(), new_cap),
        start_offsets: VulkanBuffer::new(stream, size_of::<u32>(), new_cap),
        block_sums: VulkanBuffer::new(stream, size_of::<u32>(), block_sums_cap),
        counter_upload: VulkanBuffer::new_host_visible(stream, size_of::<u32>(), 3),
        counter: VulkanBuffer::new(stream, size_of::<u32>(), 3),
        counter_download: VulkanBuffer::new_host_visible(stream, size_of::<u32>(), 3),
        sorter_scratch: VulkanBuffer::new(stream, 1, requirements.size as usize),
        sorter,
        capacity: new_cap,
    };

    let query_pool_ci = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(12);
    let timestamp_pool = unsafe {
        context
            .device
            .create_query_pool(&query_pool_ci, None)
            .expect("failed to create Vulkan downsample timestamp query pool")
    };
    let props = unsafe {
        context
            .instance
            .get_physical_device_properties(context.physical_device)
    };

    *guard = Some(VulkanDownsampleState {
        context: Arc::clone(context),
        compute,
        command_resources,
        dummy_buf,
        scratch,
        timestamp_pool,
        timestamp_period: props.limits.timestamp_period,
    });
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VulkanDownsampleTimings {
    pub total_ms: f64,
    pub upload_ms: f64,
    pub key_build_ms: f64,
    pub keyed_sort_ms: f64,
    pub run_length_encode_ms: f64,
    pub run_starts_scan_ms: f64,
    pub unique_count_sync_ms: f64,
    pub medoid_reduce_ms: f64,
    pub timestamp_key_build_ms: f64,
    pub timestamp_sort_ms: f64,
    pub download_ms: f64,
}

pub fn downsample_points_vulkan(
    stream: &Stream,
    filtered_cloud: &[PointXYZIT],
    point_skip: usize,
    voxel_size: f32,
) -> (Vec<PointXYZIT>, VulkanDownsampleTimings) {
    if filtered_cloud.is_empty() {
        return (Vec::new(), VulkanDownsampleTimings::default());
    }

    if !stream.is_vulkan() {
        return downsample_points_host(filtered_cloud, point_skip, voxel_size);
    }

    let start_total = std::time::Instant::now();
    let point_skip = point_skip.max(1);
    let selected_size = filtered_cloud.len().div_ceil(point_skip);
    ensure_downsample_state(stream, filtered_cloud.len().max(selected_size));

    let mut guard = downsample_state()
        .lock()
        .expect("vulkan downsample mutex poisoned");
    let state = guard
        .as_mut()
        .expect("vulkan downsample state was not initialized");

    let start = std::time::Instant::now();
    state.scratch.input_upload.upload(stream, filtered_cloud);
    let upload_ms = start.elapsed().as_secs_f64() * 1000.0;

    let input_upload_b = vk_buffer_handle(&state.scratch.input_upload);
    let input_b = vk_buffer_handle(&state.scratch.input);
    let selected_b = vk_buffer_handle(&state.scratch.selected);
    let keys_b = vk_buffer_handle(&state.scratch.keys);
    let indices_b = vk_buffer_handle(&state.scratch.indices);
    let unique_keys_b = vk_buffer_handle(&state.scratch.unique_keys);
    let run_starts_b = vk_buffer_handle(&state.scratch.run_starts);
    let run_ends_b = vk_buffer_handle(&state.scratch.run_ends);
    let counter_upload_b = vk_buffer_handle(&state.scratch.counter_upload);
    let counter_b = vk_buffer_handle(&state.scratch.counter);
    let counter_download_b = vk_buffer_handle(&state.scratch.counter_download);
    let start_flags_b = vk_buffer_handle(&state.scratch.start_flags);
    let start_offsets_b = vk_buffer_handle(&state.scratch.start_offsets);
    let block_sums_b = vk_buffer_handle(&state.scratch.block_sums);
    let medoids_b = vk_buffer_handle(&state.scratch.medoids);
    let output_download_b = vk_buffer_handle(&state.scratch.output_download);
    let dummy = vk_buffer_handle(&state.dummy_buf);
    let sorter_scratch = vk_buffer_handle(&state.scratch.sorter_scratch);

    let all_buffers = [
        input_b,
        selected_b,
        keys_b,
        indices_b,
        unique_keys_b,
        dummy,
        run_starts_b,
        run_ends_b,
        dummy,
        dummy,
        counter_b,
        dummy,
        dummy,
        dummy,
        dummy,
        start_flags_b,
        start_offsets_b,
        block_sums_b,
    ];

    let mut params = PushConstants {
        resolution: voxel_size,
        n: selected_size as u32,
        cap: point_skip as u32,
        ..PushConstants::default()
    };

    state
        .scratch
        .counter_upload
        .upload(stream, &[selected_size as u32, 0u32, 0u32]);

    let mut session = ComputeCommandSession::begin(&state.context, &state.command_resources, dummy);
    unsafe {
        state.context.device.cmd_reset_query_pool(
            session.command_buffer(),
            state.timestamp_pool,
            0,
            12,
        );
    }
    session.write_timestamp(state.timestamp_pool, 0);
    session.copy_buffer(
        input_upload_b,
        input_b,
        (filtered_cloud.len() * size_of::<PointXYZIT>()) as vk::DeviceSize,
    );
    session.copy_buffer(
        counter_upload_b,
        counter_b,
        (3 * size_of::<u32>()) as vk::DeviceSize,
    );
    session.transfer_to_shader_barrier();
    session.write_timestamp(state.timestamp_pool, 1);
    session.dispatch(
        &state.compute,
        state.compute.downsample_build_keys,
        &all_buffers,
        &params,
        selected_size as u32,
    );
    session.barrier();
    session.write_timestamp(state.timestamp_pool, 2);

    unsafe {
        vrdxCmdSortKeyValue(
            session.command_buffer(),
            state.scratch.sorter,
            selected_size as u32,
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
    session.write_timestamp(state.timestamp_pool, 3);
    session.write_timestamp(state.timestamp_pool, 4);

    params.j = 0;
    let rle_buffers = [
        dummy,
        dummy,
        keys_b,
        indices_b,
        unique_keys_b,
        dummy,
        run_starts_b,
        run_ends_b,
        dummy,
        dummy,
        counter_b,
        dummy,
        dummy,
        dummy,
        dummy,
        start_flags_b,
        start_offsets_b,
        block_sums_b,
    ];
    session.dispatch(
        &state.compute,
        state.compute.rle,
        &rle_buffers,
        &params,
        selected_size as u32,
    );
    session.barrier();
    session.write_timestamp(state.timestamp_pool, 5);
    session.dispatch(
        &state.compute,
        state.compute.exclusive_scan,
        &rle_buffers,
        &params,
        selected_size as u32,
    );
    session.barrier();
    session.dispatch(
        &state.compute,
        state.compute.scan_block_sums,
        &rle_buffers,
        &params,
        1,
    );
    session.barrier();
    session.dispatch(
        &state.compute,
        state.compute.add_block_prefix,
        &rle_buffers,
        &params,
        selected_size as u32,
    );
    session.barrier();
    session.write_timestamp(state.timestamp_pool, 6);
    params.j = 1;
    session.dispatch(
        &state.compute,
        state.compute.rle,
        &rle_buffers,
        &params,
        selected_size as u32,
    );
    session.barrier();
    session.write_timestamp(state.timestamp_pool, 7);

    let medoid_buffers = [
        dummy,
        selected_b,
        keys_b,
        indices_b,
        unique_keys_b,
        dummy,
        run_starts_b,
        run_ends_b,
        dummy,
        dummy,
        counter_b,
        dummy,
        dummy,
        dummy,
        dummy,
        medoids_b,
        dummy,
        dummy,
    ];
    session.dispatch(
        &state.compute,
        state.compute.downsample_medoids,
        &medoid_buffers,
        &params,
        selected_size as u32,
    );
    session.barrier();
    session.write_timestamp(state.timestamp_pool, 8);
    session.shader_to_transfer_barrier();
    session.copy_buffer(
        medoids_b,
        output_download_b,
        (selected_size * size_of::<PointXYZIT>()) as vk::DeviceSize,
    );
    session.copy_buffer(
        counter_b,
        counter_download_b,
        (3 * size_of::<u32>()) as vk::DeviceSize,
    );
    session.transfer_to_host_barrier();
    session.write_timestamp(state.timestamp_pool, 9);
    session.submit();

    let mut timestamps = [0u64; 12];
    let query_ok = unsafe {
        state.context.device.get_query_pool_results(
            state.timestamp_pool,
            0,
            &mut timestamps,
            vk::QueryResultFlags::TYPE_64,
        )
    }
    .is_ok();
    let elapsed_query_ms = |a: usize, b: usize| -> f64 {
        if query_ok && timestamps[b] >= timestamps[a] {
            (timestamps[b] - timestamps[a]) as f64 * state.timestamp_period as f64 / 1_000_000.0
        } else {
            0.0
        }
    };

    let unique_start = std::time::Instant::now();
    let counters: Vec<u32> = state.scratch.counter_download.download(stream);
    let unique_count = (counters[0] as usize).min(selected_size);
    let unique_count_sync_ms = unique_start.elapsed().as_secs_f64() * 1000.0;

    state.scratch.output_download.n = unique_count;
    let download_start = std::time::Instant::now();
    let mut out: Vec<PointXYZIT> = state.scratch.output_download.download(stream);
    let download_ms = download_start.elapsed().as_secs_f64() * 1000.0;

    let sort_start = std::time::Instant::now();
    out.sort_unstable_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
    let _timestamp_sort_ms = sort_start.elapsed().as_secs_f64() * 1000.0;

    (
        out,
        VulkanDownsampleTimings {
            total_ms: start_total.elapsed().as_secs_f64() * 1000.0,
            upload_ms,
            key_build_ms: elapsed_query_ms(1, 2),
            keyed_sort_ms: elapsed_query_ms(2, 3),
            run_length_encode_ms: elapsed_query_ms(4, 5) + elapsed_query_ms(6, 7),
            run_starts_scan_ms: elapsed_query_ms(5, 6),
            unique_count_sync_ms,
            medoid_reduce_ms: elapsed_query_ms(7, 8),
            download_ms: download_ms + elapsed_query_ms(8, 9),
            ..Default::default()
        },
    )
}

fn downsample_points_host(
    filtered_cloud: &[PointXYZIT],
    point_skip: usize,
    voxel_size: f32,
) -> (Vec<PointXYZIT>, VulkanDownsampleTimings) {
    use std::collections::BTreeMap;

    fn hash_point(point: &PointXYZIT, voxel_size: f32) -> i32 {
        let inv = 1.0 / voxel_size;
        let x = (point.x * inv).floor() as i32;
        let y = (point.y * inv).floor() as i32;
        let z = (point.z * inv).floor() as i32;
        let h = (x as u32)
            .wrapping_mul(73_856_093)
            .wrapping_add((y as u32).wrapping_mul(471_943))
            .wrapping_add((z as u32).wrapping_mul(83_492_791));
        h as i32
    }

    let start = std::time::Instant::now();
    let point_skip = point_skip.max(1);
    let mut voxels: BTreeMap<i32, Vec<PointXYZIT>> = BTreeMap::new();
    for point in filtered_cloud.iter().step_by(point_skip) {
        voxels
            .entry(hash_point(point, voxel_size))
            .or_default()
            .push(*point);
    }

    let mut out = Vec::with_capacity(voxels.len());
    for points in voxels.values() {
        let inv_n = 1.0 / points.len() as f32;
        let center = points.iter().fold([0.0f32; 3], |mut acc, p| {
            acc[0] += p.x * inv_n;
            acc[1] += p.y * inv_n;
            acc[2] += p.z * inv_n;
            acc
        });
        let medoid = points
            .iter()
            .min_by(|a, b| {
                let da = (a.x - center[0]).powi(2)
                    + (a.y - center[1]).powi(2)
                    + (a.z - center[2]).powi(2);
                let db = (b.x - center[0]).powi(2)
                    + (b.y - center[1]).powi(2)
                    + (b.z - center[2]).powi(2);
                da.total_cmp(&db)
            })
            .copied()
            .expect("voxel must not be empty");
        out.push(medoid);
    }
    out.sort_unstable_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());

    (
        out,
        VulkanDownsampleTimings {
            total_ms: start.elapsed().as_secs_f64() * 1000.0,
            ..Default::default()
        },
    )
}

fn ensure_scratch_capacity(state: &mut VulkanGpuMapState, stream: &Stream, n_points: usize) {
    if state.scratch.capacity >= n_points {
        return;
    }
    let new_cap = (((n_points as f32) * 1.5).ceil() as usize)
        .max(n_points)
        .next_power_of_two();
    state.scratch.keys = VulkanBuffer::new(stream, size_of::<i32>(), new_cap);
    state.scratch.indices = VulkanBuffer::new(stream, size_of::<u32>(), new_cap);
    state.scratch.packed = VulkanBuffer::new(stream, size_of::<u64>(), new_cap);
    state.scratch.run_starts = VulkanBuffer::new(stream, size_of::<u32>(), new_cap);
    state.scratch.run_ends = VulkanBuffer::new(stream, size_of::<u32>(), new_cap);
    state.scratch.unique_keys = VulkanBuffer::new(stream, size_of::<i32>(), new_cap);
    state.scratch.target_indices = VulkanBuffer::new(stream, size_of::<u32>(), new_cap);
    state.scratch.update_flags = VulkanBuffer::new(stream, size_of::<u32>(), new_cap);
    state.scratch.sorted_points =
        VulkanBuffer::new(stream, size_of::<crate::types::Point>(), new_cap);
    state.scratch.is_new = VulkanBuffer::new(stream, size_of::<u32>(), new_cap);
    state.scratch.new_offset = VulkanBuffer::new(stream, size_of::<u32>(), new_cap);
    state.scratch.plane_compact =
        VulkanBuffer::new(stream, size_of::<crate::types::PointNormal>(), new_cap);
    state.scratch.plane_readback =
        VulkanBuffer::new_host_visible(stream, size_of::<crate::types::PointNormal>(), new_cap);
    // block_sums needs ceil(new_cap / 512) elements, round up to next power of 2 for safety
    let block_sums_cap = (new_cap / 512 + 1).next_power_of_two().max(1024);
    state.scratch.block_sums = VulkanBuffer::new(stream, size_of::<u32>(), block_sums_cap);

    let mut requirements = VrdxSorterStorageRequirements {
        size: 0,
        usage: vk::BufferUsageFlags::empty(),
    };
    let sort_capacity = new_cap.max(state.dist_cap as usize);
    unsafe {
        vrdxGetSorterKeyValueStorageRequirements(
            state.scratch.sorter,
            sort_capacity as u32,
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
    dummy_buf: &VulkanBuffer,
    scan_number: u32,
    dist_cap: u32,
    session: &mut ComputeCommandSession,
) {
    let dist_b = vk_buffer_handle(dist_buf);
    let lru_keys = vk_buffer_handle(&scratch.lru_keys);
    let lru_mapping = vk_buffer_handle(&scratch.lru_mapping);
    let sorter_scratch = vk_buffer_handle(&scratch.sorter_scratch);
    let counter_b = vk_buffer_handle(&scratch.counter);
    let dummy = vk_buffer_handle(dummy_buf);

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

    if requires_portable_map_sort(&session.context) {
        record_portable_sort(
            session,
            compute,
            dummy,
            lru_keys,
            lru_mapping,
            dist_cap,
            scratch.lru_keys.capacity() as u32,
        );
    } else {
        unsafe {
            vrdxCmdSortKeyValue(
                session.command_buffer(),
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
}

pub fn add_to_map(
    stream: &Stream,
    cloud: &VulkanBuffer,
    ekf_pose_t: &Vector3,
    ekf_pose_r: &Quaternion,
    scan_id: usize,
) {
    with_state(|state| match &mut state.backend {
        MapBackendState::Host(host) => {
            let cloud_cpu: Vec<PointXYZIT> = cloud.download(stream);
            let pose = nalgebra::Isometry3::from_parts(
                nalgebra::Translation3::new(ekf_pose_t.x, ekf_pose_t.y, ekf_pose_t.z),
                nalgebra::UnitQuaternion::from(*ekf_pose_r),
            );
            host.map.add(&cloud_cpu, &pose, scan_id);
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
            state.scan_number = scan_id as u32;
            let scan_number = state.scan_number;
            let add_cursor_before = state.add_cursor;

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
            let is_new_b = vk_buffer_handle(&state.scratch.is_new);
            let new_offset_b = vk_buffer_handle(&state.scratch.new_offset);
            let block_sums_b = vk_buffer_handle(&state.scratch.block_sums);

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
                is_new_b,
                new_offset_b,
                block_sums_b,
            ];

            let dummy = vk_buffer_handle(&state.dummy_buf);
            let mut session =
                ComputeCommandSession::begin(&state.context, &state.command_resources, dummy);
            // Reset timestamp queries for add_to_map (queries 2-3)
            unsafe {
                state.context.device.cmd_reset_query_pool(
                    session.command_buffer(),
                    state.timestamp_pool,
                    2,
                    2,
                );
            }
            session.write_timestamp(state.timestamp_pool, 2);

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
            if requires_portable_map_sort(&state.context) {
                record_portable_sort(
                    &mut session,
                    &state.compute,
                    dummy,
                    keys_b,
                    indices_b,
                    n_points as u32,
                    state.scratch.capacity as u32,
                );
            } else {
                let sorter_scratch = vk_buffer_handle(&state.scratch.sorter_scratch);
                unsafe {
                    vrdxCmdSortKeyValue(
                        session.command_buffer(),
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
            }

            // 3. GPU Run-Length Encoding
            // RLE is split into deterministic mark/scan/emit passes. The old
            // atomic append made unique-key order depend on shader scheduling,
            // which changes later LRU victim assignment.
            let counters_host = [n_points as u32, state.add_cursor, 0u32];
            state.scratch.counter.upload(stream, &counters_host);

            params.j = 0;
            session.dispatch(
                &state.compute,
                state.compute.rle,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            session.dispatch(
                &state.compute,
                state.compute.exclusive_scan,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            session.dispatch(
                &state.compute,
                state.compute.scan_block_sums,
                &all_buffers,
                &params,
                1,
            );
            session.barrier();

            session.dispatch(
                &state.compute,
                state.compute.add_block_prefix,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            params.j = 1;
            session.dispatch(
                &state.compute,
                state.compute.rle,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            // 4. Classify Keys (deterministic: mark each key as new or existing)
            // Run classify_keys.comp to populate is_new[], target_indices[], update_flags[]
            session.dispatch(
                &state.compute,
                state.compute.classify_keys,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            // 5. Exclusive Prefix Sum on is_new[] -> new_offset[]
            // Multi-pass Blelloch scan for arbitrary array sizes:
            //   Pass 1: local scan per workgroup, store block sums
            //   Pass 2: prefix sum of block sums (single workgroup)
            // Pass 3: add block prefix to each element

            // Calculate number of workgroups needed (each processes 2 * THREADS_PER_GROUP = 512 elements)
            let _scan_workgroups = div_ceil_u32(n_points as u32, 2 * THREADS_PER_GROUP);

            // Pass 1: Local exclusive scan
            session.dispatch(
                &state.compute,
                state.compute.exclusive_scan,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            // Pass 2: Prefix sum of block sums (single workgroup, handles up to 1024 blocks)
            session.dispatch(
                &state.compute,
                state.compute.scan_block_sums,
                &all_buffers,
                &params,
                1, // Single workgroup
            );
            session.barrier();

            // Pass 3: Add block prefix to each element
            session.dispatch(
                &state.compute,
                state.compute.add_block_prefix,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            // 6. Assign New Targets (deterministic slot assignment)
            // Uses prev_add_cursor + new_offset[idx] for deterministic slots
            session.dispatch(
                &state.compute,
                state.compute.assign_new_targets,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            // 7. Finalize Cursors (single-thread update)
            session.dispatch(
                &state.compute,
                state.compute.finalize_cursors,
                &all_buffers,
                &params,
                1, // Single thread
            );
            session.barrier();

            // 8. Reorder Points
            session.dispatch(
                &state.compute,
                state.compute.reorder_points,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            // 9. Build Voxels
            session.dispatch(
                &state.compute,
                state.compute.build_voxels,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            // 10. Rebuild Hashmap
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

            if add_cursor_before >= state.dist_cap {
                rebuild_gpu_lru_order(
                    &state.compute,
                    &mut state.scratch,
                    &state.dist_buf,
                    &state.dummy_buf,
                    state.scan_number,
                    state.dist_cap,
                    &mut session,
                );
            }

            session.write_timestamp(state.timestamp_pool, 3);
            session.submit();
            state
                .pending_add_to_map_timing
                .store(true, std::sync::atomic::Ordering::Release);

            // Download counters to update host-side state
            let counters_result: Vec<u32> = state.scratch.counter.download(stream);
            state.add_cursor = counters_result[1].min(state.dist_cap);
            state.lru_cursor = counters_result[2].min(state.dist_cap);

            if add_cursor_before < state.dist_cap && state.add_cursor >= state.dist_cap {
                let dummy = vk_buffer_handle(&state.dummy_buf);
                let mut session =
                    ComputeCommandSession::begin(&state.context, &state.command_resources, dummy);
                rebuild_gpu_lru_order(
                    &state.compute,
                    &mut state.scratch,
                    &state.dist_buf,
                    &state.dummy_buf,
                    state.scan_number,
                    state.dist_cap,
                    &mut session,
                );
                session.submit();
            }
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

#[derive(Debug, Clone, Copy, Default)]
pub struct VulkanPlaneTimings {
    pub kernel_compact_ms: f64,
    pub download_ms: f64,
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
    download_buff: &mut Vec<PointNormal>,
    counter_buff: &mut [u32; 3],
) -> VulkanPlaneTimings {
    assert!(
        out_buff.len() >= cloud.n,
        "output buffer must fit cloud size"
    );
    with_state(|state| match &mut state.backend {
        MapBackendState::Host(host) => {
            let start = std::time::Instant::now();
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
                    let s = 1.0_f32 - 0.9_f32 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
                    if s > 0.9_f32 {
                        planes.push(PointXYZINormal::new(
                            point_local.x,
                            point_local.y,
                            point_local.z,
                            plane[3],
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
            VulkanPlaneTimings {
                kernel_compact_ms: start.elapsed().as_secs_f64() * 1000.0,
                download_ms: 0.0,
            }
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

            ensure_scratch_capacity(state, stream, cloud.n);

            // Capacity-based reuse keeps allocations down, but downloads are
            // bounded by logical length. Reset plane buffers for the current cloud.
            out_buff_gpu.n = cloud.n;
            state.scratch.plane_compact.n = cloud.n;
            state.scratch.plane_readback.n = cloud.n;

            state
                .scratch
                .counter
                .upload(stream, &[cloud.n as u32, 0u32, 0u32]);
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
                n: cloud.n as u32,
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
            let valid_flags_b = vk_buffer_handle(&state.scratch.is_new);
            let offsets_b = vk_buffer_handle(&state.scratch.new_offset);
            let block_sums_b = vk_buffer_handle(&state.scratch.block_sums);
            let compact_out_b = vk_buffer_handle(&state.scratch.plane_compact);
            let dummy = vk_buffer_handle(&state.dummy_buf);

            let plane_buffers = [
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
                valid_flags_b,
                dummy,
            ];
            let scan_buffers = [
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                counter_b,
                dummy,
                dummy,
                dummy,
                dummy,
                valid_flags_b,
                offsets_b,
                block_sums_b,
            ];
            let compact_buffers = [
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                counter_b,
                dummy,
                dummy,
                dummy,
                dummy,
                out_gpu_b,
                offsets_b,
                compact_out_b,
            ];

            let kernel_compact_start = std::time::Instant::now();
            let mut session =
                ComputeCommandSession::begin(&state.context, &state.command_resources, dummy);
            // Reset timestamp queries for this operation
            unsafe {
                state.context.device.cmd_reset_query_pool(
                    session.command_buffer(),
                    state.timestamp_pool,
                    0,
                    4,
                );
            }
            session.write_timestamp(state.timestamp_pool, 0);
            session.dispatch(
                &state.compute,
                state.compute.plane_krnl,
                &plane_buffers,
                &params,
                cloud.n as u32,
            );
            session.barrier();
            session.dispatch(
                &state.compute,
                state.compute.exclusive_scan,
                &scan_buffers,
                &params,
                cloud.n as u32,
            );
            session.barrier();
            session.dispatch(
                &state.compute,
                state.compute.scan_block_sums,
                &scan_buffers,
                &params,
                1,
            );
            session.barrier();
            session.dispatch(
                &state.compute,
                state.compute.add_block_prefix,
                &scan_buffers,
                &params,
                cloud.n as u32,
            );
            session.barrier();
            session.dispatch(
                &state.compute,
                state.compute.compact_planes,
                &compact_buffers,
                &params,
                cloud.n as u32,
            );
            session.dispatch(
                &state.compute,
                state.compute.plane_count,
                &scan_buffers,
                &params,
                1,
            );
            session.write_timestamp(state.timestamp_pool, 1);
            session.shader_to_transfer_barrier();
            let plane_readback_b = vk_buffer_handle(&state.scratch.plane_readback);
            let plane_copy_bytes =
                (cloud.n * size_of::<crate::types::PointNormal>()) as vk::DeviceSize;
            session.copy_buffer(compact_out_b, plane_readback_b, plane_copy_bytes);
            session.transfer_to_host_barrier();
            session.submit();
            let kernel_compact_ms = kernel_compact_start.elapsed().as_secs_f64() * 1000.0;
            state
                .pending_planes_timing
                .store(true, std::sync::atomic::Ordering::Release);
            log::debug!("vulkan planes: kernel complete");

            let download_start = std::time::Instant::now();
            state.scratch.counter.download_into(stream, counter_buff);
            let downloaded = (counter_buff[0] as usize).min(out_buff.len()).min(cloud.n);

            assert!(
                download_buff.len() >= downloaded,
                "persistent Vulkan plane download buffer must fit compacted output"
            );
            state.scratch.plane_readback.n = downloaded;
            state
                .scratch
                .plane_readback
                .download_into(stream, &mut download_buff[..downloaded]);
            out_buff.clear();
            out_buff.extend(
                download_buff
                    .iter()
                    .take(downloaded)
                    .map(|p| PointXYZINormal::from(*p)),
            );
            let download_ms = download_start.elapsed().as_secs_f64() * 1000.0;

            VulkanPlaneTimings {
                kernel_compact_ms,
                download_ms,
            }
        }
    })
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

/// Try to retrieve GPU timing measurements from previous operations.
/// Returns (planes_ms, add_to_map_ms) if timing results are available.
/// This is non-blocking - returns None if queries are not ready.
pub fn try_get_gpu_timings() -> Option<(f64, f64)> {
    with_state(|state| match &mut state.backend {
        MapBackendState::Host(_) => None,
        MapBackendState::Vulkan(gpu) => {
            let mut planes_ms = None;
            let mut add_to_map_ms = None;

            // Check planes timing (queries 0-1)
            if gpu
                .pending_planes_timing
                .load(std::sync::atomic::Ordering::Acquire)
            {
                if let Some(ms) = get_query_result(&gpu, 0) {
                    log::debug!("Vulkan planes GPU timing: {}ms", ms);
                    gpu.pending_planes_timing
                        .store(false, std::sync::atomic::Ordering::Release);
                    planes_ms = Some(ms);
                } else {
                    log::debug!("Vulkan planes GPU timing not ready yet");
                }
            }

            // Check add_to_map timing (queries 2-3)
            if gpu
                .pending_add_to_map_timing
                .load(std::sync::atomic::Ordering::Acquire)
            {
                if let Some(ms) = get_query_result(&gpu, 2) {
                    log::debug!("Vulkan add_to_map GPU timing: {}ms", ms);
                    gpu.pending_add_to_map_timing
                        .store(false, std::sync::atomic::Ordering::Release);
                    add_to_map_ms = Some(ms);
                } else {
                    log::debug!("Vulkan add_to_map GPU timing not ready yet");
                }
            }

            match (planes_ms, add_to_map_ms) {
                (Some(p), Some(a)) => Some((p, a)),
                (Some(p), None) => Some((p, 0.0)),
                (None, Some(a)) => Some((0.0, a)),
                (None, None) => None,
            }
        }
    })
}

fn get_query_result(state: &VulkanGpuMapState, start_query: u32) -> Option<f64> {
    let mut timestamps = [0u64; 2];
    let result = unsafe {
        state.context.device.get_query_pool_results(
            state.timestamp_pool,
            start_query,
            &mut timestamps,
            vk::QueryResultFlags::TYPE_64,
        )
    };

    if result.is_ok() {
        let elapsed_ns = (timestamps[1] - timestamps[0]) as f64 * state.timestamp_period as f64;
        Some(elapsed_ns / 1_000_000.0)
    } else {
        None
    }
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
                unsafe {
                    if mapped_ptr.is_some() {
                        context.device.unmap_memory(*memory);
                    }
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

#[cfg(test)]
mod map_full_tests {
    use std::collections::HashMap;
    use std::mem::size_of;
    use std::sync::{Arc, Mutex};

    #[cfg(feature = "vulkan_enable")]
    #[test]
    fn downsample_shaders_compile() {
        let mut compiler = shaderc::Compiler::new().expect("failed creating shader compiler");
        let downsample_src = super::shaders::source_with_common(super::shaders::DOWNSAMPLE);
        let reorder_src = super::shaders::source_with_common(super::shaders::DOWNSAMPLE_REORDER);

        let cases = [
            (
                "downsample_build_keys",
                downsample_src.as_str(),
                vec![("BUILD_KEYS", None)],
            ),
            (
                "downsample_medoids",
                downsample_src.as_str(),
                vec![("MEDOIDS", None)],
            ),
            (
                "downsample_timestamp_keys",
                downsample_src.as_str(),
                vec![("TIMESTAMP_KEYS", None)],
            ),
            ("downsample_reorder", reorder_src.as_str(), vec![]),
        ];

        for (name, source, defines) in cases {
            super::compile_shader(&mut compiler, source, name, &defines);
        }
    }

    #[cfg(feature = "vulkan_enable")]
    #[test]
    fn portable_sort_matches_unsigned_key_value_order() {
        let stream = super::Stream::new();
        if !stream.is_vulkan() {
            return;
        }
        let super::StreamBackend::Vulkan(context) = &stream.backend else {
            unreachable!();
        };

        let count = 100_000usize;
        let capacity = count.next_power_of_two();
        let source_keys: Vec<i32> = (0..count)
            .map(|i| {
                let value = (i.wrapping_mul(2654435761usize) ^ (i / 7)) as u32;
                (value % 97).wrapping_mul(0x01010101) as i32
            })
            .collect();
        let source_values: Vec<u32> = (0..count as u32).collect();
        let mut keys = super::VulkanBuffer::new(&stream, size_of::<i32>(), capacity);
        let mut values = super::VulkanBuffer::new(&stream, size_of::<u32>(), capacity);
        keys.upload(&stream, &source_keys);
        values.upload(&stream, &source_values);
        let dummy = super::VulkanBuffer::new(&stream, size_of::<u32>(), 1);

        let compute = super::build_pipelines(Arc::clone(context));
        let resources = Mutex::new(super::ComputeCommandResources::new(context, &compute));
        let mut session = super::ComputeCommandSession::begin(
            context,
            &resources,
            super::vk_buffer_handle(&dummy),
        );
        super::record_portable_sort(
            &mut session,
            &compute,
            super::vk_buffer_handle(&dummy),
            super::vk_buffer_handle(&keys),
            super::vk_buffer_handle(&values),
            count as u32,
            capacity as u32,
        );
        session.submit();

        let sorted_keys: Vec<i32> = keys.download(&stream);
        let sorted_values: Vec<u32> = values.download(&stream);
        let mut expected: Vec<(i32, u32)> = source_keys.into_iter().zip(source_values).collect();
        expected.sort_unstable_by_key(|(key, value)| (*key as u32, *value));
        let actual: Vec<(i32, u32)> = sorted_keys.into_iter().zip(sorted_values).collect();
        assert_eq!(actual, expected);
    }

    const EMPTY_KEY: i32 = i32::MIN;
    const INVALID_INDEX: u32 = u32::MAX;

    fn assign_indices_cuda_style(
        unique_keys: &[i32],
        existing: &HashMap<i32, u32>,
        dist_cap: u32,
        add_cursor_start: u32,
        lru_mapping: &[u32],
    ) -> (Vec<u32>, Vec<u32>, Vec<i32>, u32, u32) {
        let mut target_indices = Vec::with_capacity(unique_keys.len());
        let mut update_flags = Vec::with_capacity(unique_keys.len());
        let mut dist_keys = vec![EMPTY_KEY; dist_cap as usize];
        let mut add_cursor = add_cursor_start;
        let mut lru_cursor = 0u32;

        for &key in unique_keys {
            if key == EMPTY_KEY {
                target_indices.push(INVALID_INDEX);
                update_flags.push(0);
                continue;
            }

            if let Some(&idx) = existing.get(&key) {
                target_indices.push(idx);
                update_flags.push(1);
                continue;
            }

            let old_add = add_cursor;
            add_cursor = add_cursor.saturating_add(1);

            if old_add < dist_cap {
                target_indices.push(old_add);
                update_flags.push(0);
                dist_keys[old_add as usize] = key;
                continue;
            }

            let lru_idx = lru_cursor;
            lru_cursor = lru_cursor.saturating_add(1);
            if lru_idx < dist_cap {
                let victim = lru_mapping[lru_idx as usize];
                target_indices.push(victim);
                update_flags.push(0);
                dist_keys[victim as usize] = key;
            } else {
                target_indices.push(INVALID_INDEX);
                update_flags.push(0);
            }
        }

        (
            target_indices,
            update_flags,
            dist_keys,
            add_cursor,
            lru_cursor,
        )
    }

    #[test]
    fn map_full_assign_stops_when_lru_candidates_exhausted() {
        let existing = HashMap::new();
        let lru_mapping = vec![2u32, 1u32, 0u32, 3u32];
        let (targets, flags, dist_keys, add_cursor, lru_cursor) =
            assign_indices_cuda_style(&[11, 22, 33, 44, 55, 66], &existing, 4, 4, &lru_mapping);

        assert_eq!(
            targets,
            vec![2, 1, 0, 3, INVALID_INDEX, INVALID_INDEX],
            "replacements must not wrap and overwrite earlier victims in a single scan"
        );
        assert_eq!(flags, vec![0, 0, 0, 0, 0, 0]);
        assert_eq!(add_cursor, 10);
        assert_eq!(lru_cursor, 6);
        assert_eq!(dist_keys[2], 11);
        assert_eq!(dist_keys[1], 22);
        assert_eq!(dist_keys[0], 33);
        assert_eq!(dist_keys[3], 44);
    }

    #[test]
    fn map_full_assign_resets_lru_consumption_each_scan() {
        let existing = HashMap::new();
        let lru_mapping = vec![3u32, 2u32, 1u32, 0u32];

        let (first_targets, _, _, _, first_lru_cursor) =
            assign_indices_cuda_style(&[10, 20], &existing, 4, 4, &lru_mapping);
        let (second_targets, _, _, _, second_lru_cursor) =
            assign_indices_cuda_style(&[30, 40], &existing, 4, 4, &lru_mapping);

        assert_eq!(first_targets, vec![3, 2]);
        assert_eq!(second_targets, vec![3, 2]);
        assert_eq!(first_lru_cursor, 2);
        assert_eq!(second_lru_cursor, 2);
    }
}
