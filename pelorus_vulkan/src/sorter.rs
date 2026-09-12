use ash::vk;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct VrdxSorter(pub(crate) u64);

#[repr(C)]
pub(crate) struct VrdxSorterCreateInfo {
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: vk::Device,
    pub(crate) pipeline_cache: vk::PipelineCache,
}

#[repr(C)]
pub(crate) struct VrdxSorterStorageRequirements {
    pub(crate) size: vk::DeviceSize,
    pub(crate) usage: vk::BufferUsageFlags,
}

#[link(name = "vk_radix_sort", kind = "static")]
unsafe extern "C" {
    pub(crate) fn vrdxCreateSorter(
        p_create_info: *const VrdxSorterCreateInfo,
        p_sorter: *mut VrdxSorter,
    ) -> i32;
    pub(crate) fn vrdxDestroySorter(sorter: VrdxSorter);
    pub(crate) fn vrdxGetSorterKeyValueStorageRequirements(
        sorter: VrdxSorter,
        max_element_count: u32,
        requirements: *mut VrdxSorterStorageRequirements,
    );
    pub(crate) fn vrdxCmdSortKeyValue(
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
