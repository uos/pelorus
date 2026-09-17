use ash::{Device, Entry, Instance, khr::push_descriptor, vk};
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) struct VulkanContext {
    #[allow(dead_code)]
    entry: Entry,
    pub(crate) instance: Instance,
    pub(crate) device: Device,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) queue: vk::Queue,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(crate) submit_lock: Mutex<()>,
    pub(crate) precise_multiresolution: bool,
    /// Set by `shutdown`. Teardown cannot wait for the `Arc` refcount to reach zero.
    destroyed: std::sync::atomic::AtomicBool,
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

            if feat12.buffer_device_address == vk::FALSE || feat12.scalar_block_layout == vk::FALSE
            {
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
                "no Vulkan device with compute + bufferDeviceAddress + scalarBlockLayout + VK_KHR_push_descriptor support available"
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
        // Without this, an out-of-range SSBO index is undefined behaviour. Software
        // implementations (lavapipe, SwiftShader) JIT shaders to host code, so such an
        // index dereferences a host pointer and takes the whole process down. With it,
        // the access is defined: reads return zero, writes are discarded.
        let supported_features = unsafe { instance.get_physical_device_features(physical_device) };
        let precise_multiresolution = supported_features.shader_float64 == vk::TRUE
            && supported_features.shader_int64 == vk::TRUE;
        let enabled_features = vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(true)
            .shader_float64(precise_multiresolution)
            .shader_int64(precise_multiresolution);
        let enabled_device_extensions = [push_descriptor::NAME.as_ptr()];
        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_ci))
            .enabled_extension_names(&enabled_device_extensions)
            .enabled_features(&enabled_features)
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
            precise_multiresolution,
            destroyed: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// False once `shutdown` has destroyed the device.
    pub(crate) fn is_alive(&self) -> bool {
        !self.destroyed.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Destroy the device and instance now, regardless of the `Arc` refcount.
    /// Returns false if another caller already did it.
    pub(crate) fn destroy_now(&self) -> bool {
        if self
            .destroyed
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            return false;
        }
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        true
    }

    pub(crate) fn sync(&self) {
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
        // Normally `shutdown` already did this. This only covers contexts that
        // were never handed to `shutdown` (tests, probing, host fallback paths).
        if self.destroy_now() {
            log::debug!("Vulkan device destroyed without an explicit shutdown");
        }
    }
}

static GLOBAL_VULKAN_CONTEXT: OnceLock<Mutex<Option<Arc<VulkanContext>>>> = OnceLock::new();
static VULKAN_UNAVAILABLE: OnceLock<String> = OnceLock::new();

#[cfg(feature = "vulkan_enable")]
pub(crate) fn global_vulkan_slot() -> &'static Mutex<Option<Arc<VulkanContext>>> {
    GLOBAL_VULKAN_CONTEXT.get_or_init(|| Mutex::new(None))
}

/// The shared context, creating or re-creating it if needed.
#[cfg(feature = "vulkan_enable")]
pub(crate) fn global_vulkan_context() -> Result<Arc<VulkanContext>, &'static str> {
    if let Some(e) = VULKAN_UNAVAILABLE.get() {
        return Err(e.as_str());
    }

    let mut guard = global_vulkan_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    if let Some(ctx) = guard.as_ref() {
        if ctx.is_alive() {
            return Ok(Arc::clone(ctx));
        }
        // Released by `shutdown`. The handles are dead, so build a new device.
        *guard = None;
    }

    match VulkanContext::try_new() {
        Ok(ctx) => {
            let ctx = Arc::new(ctx);
            *guard = Some(Arc::clone(&ctx));
            Ok(ctx)
        }
        Err(e) => {
            let _ = VULKAN_UNAVAILABLE.set(e);
            Err(VULKAN_UNAVAILABLE
                .get()
                .map(String::as_str)
                .unwrap_or("Vulkan unavailable"))
        }
    }
}

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
    global_vulkan_context().is_ok()
}

pub fn get_vulkan_metadata() -> Option<VulkanMetadata> {
    let context = global_vulkan_context().ok()?;

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

/// Whether the selected Vulkan device supports the multiresolution arithmetic.
pub fn is_multiresolution_available() -> bool {
    global_vulkan_context().is_ok_and(|context| context.precise_multiresolution)
}
