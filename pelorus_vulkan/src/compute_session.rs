use crate::context::VulkanContext;
use crate::pipelines::{
    DESCRIPTOR_BINDING_COUNT, MAX_DISPATCHES_PER_SESSION, PushConstants, SUBMIT_SPIN_ITERATIONS,
    THREADS_PER_GROUP, VulkanComputePipelines,
};
use ash::vk;
use std::sync::{Arc, Mutex};

pub(crate) fn scan_work_items(elements: u32) -> u32 {
    elements.div_ceil(2)
}

fn div_ceil_u32(v: u32, d: u32) -> u32 {
    if v == 0 { 0 } else { 1 + (v - 1) / d }
}

pub(crate) struct ComputeCommandResources {
    context: Arc<VulkanContext>,
    command_buffer: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    fence: vk::Fence,
}

impl ComputeCommandResources {
    pub(crate) fn new(context: &Arc<VulkanContext>, pipelines: &VulkanComputePipelines) -> Self {
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(context.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // VkCommandPool is externally synchronized. All Vulkan streams share this
        // pool, so allocation must use the same lock as reset, submission, and free.
        let command_pool_lock = context
            .submit_lock
            .lock()
            .expect("Vulkan submission lock poisoned");
        let command_buffer = unsafe {
            context
                .device
                .allocate_command_buffers(&alloc)
                .expect("failed to allocate command buffer")[0]
        };
        drop(command_pool_lock);

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

        let fence = unsafe {
            context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("failed creating compute fence")
        };

        Self {
            context: Arc::clone(context),
            command_buffer,
            descriptor_pool,
            descriptor_sets,
            fence,
        }
    }
}

impl Drop for ComputeCommandResources {
    fn drop(&mut self) {
        if !self.context.is_alive() {
            return;
        }
        // VkCommandPool is shared across streams and requires externally
        // synchronized host access, including command-buffer deallocation.
        let command_pool_lock = self
            .context
            .submit_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            self.context.device.free_command_buffers(
                self.context.command_pool,
                std::slice::from_ref(&self.command_buffer),
            );
        }
        drop(command_pool_lock);

        unsafe {
            self.context.device.destroy_fence(self.fence, None);
            self.context
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

pub(crate) struct ComputeCommandSession<'a> {
    pub(crate) context: &'a Arc<VulkanContext>,
    resources: std::sync::MutexGuard<'a, ComputeCommandResources>,
    next_set_idx: usize,
    dummy_buffer: vk::Buffer,
    _submit_lock: std::sync::MutexGuard<'a, ()>,
}

impl<'a> ComputeCommandSession<'a> {
    pub(crate) fn begin(
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

    pub(crate) fn dispatch(
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

        let infos: [vk::DescriptorBufferInfo; DESCRIPTOR_BINDING_COUNT as usize] =
            std::array::from_fn(|idx| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffers.get(idx).copied().unwrap_or(self.dummy_buffer))
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            });
        let writes: [vk::WriteDescriptorSet; DESCRIPTOR_BINDING_COUNT as usize] =
            std::array::from_fn(|idx| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(idx as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&infos[idx]))
            });
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

    pub(crate) fn barrier(&self) {
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

    /// Full compute<->transfer barrier.
    ///
    /// `vrdxCmdSortKeyValue` starts by writing its scratch buffer with
    /// `vkCmdUpdateBuffer`/`vkCmdFillBuffer` (transfer writes) and ends with
    /// compute dispatches. The plain `barrier()` only covers
    /// COMPUTE_SHADER -> COMPUTE_SHADER, so back-to-back sorts recorded into the
    /// same command buffer race: the next sort's transfer writes to the shared
    /// scratch buffer overlap the previous sort's still-running shaders.
    pub(crate) fn sort_barrier(&self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::TRANSFER_READ,
            )
            .dst_access_mask(
                vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::TRANSFER_READ,
            );
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.resources.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    pub(crate) fn shader_to_transfer_barrier(&self) {
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

    /// Publish device results both to staging copies and mapped host counters.
    pub(crate) fn shader_to_readback_barrier(&self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ);
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.resources.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    pub(crate) fn transfer_to_host_barrier(&self) {
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

    pub(crate) fn transfer_to_shader_barrier(&self) {
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

    pub(crate) fn copy_buffer(&self, src: vk::Buffer, dst: vk::Buffer, size: vk::DeviceSize) {
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

    pub(crate) fn write_timestamp(&self, timestamp_pool: vk::QueryPool, query_index: u32) {
        unsafe {
            self.context.device.cmd_write_timestamp(
                self.resources.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                timestamp_pool,
                query_index,
            );
        }
    }

    pub(crate) fn command_buffer(&self) -> vk::CommandBuffer {
        self.resources.command_buffer
    }

    pub(crate) fn submit(self) {
        unsafe {
            self.context
                .device
                .end_command_buffer(self.resources.command_buffer)
                .expect("failed to end command buffer");
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.resources.command_buffer));
            let fence = self.resources.fence;
            self.context
                .device
                .reset_fences(std::slice::from_ref(&fence))
                .expect("failed resetting compute fence");
            self.context
                .device
                .queue_submit(self.context.queue, std::slice::from_ref(&submit), fence)
                .expect("failed submitting compute work");

            // The EKF blocks on this result, so the submit round-trip is on the
            // critical path once per iteration. `vkWaitForFences` parks in the
            // kernel and pays a wake-up. These submissions finish in ~100us, so poll
            // briefly first and only block if the work turns out to be long.
            let mut signalled = false;
            for _ in 0..SUBMIT_SPIN_ITERATIONS {
                match self.context.device.get_fence_status(fence) {
                    Ok(true) => {
                        signalled = true;
                        break;
                    }
                    Ok(false) => std::hint::spin_loop(),
                    Err(e) => panic!("failed querying compute fence: {e}"),
                }
            }
            if !signalled {
                self.context
                    .device
                    .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                    .expect("failed waiting for compute fence");
            }
        }
    }
}

pub(crate) fn record_portable_sort(
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
