use crate::buffer::VulkanBuffer;
use crate::compute_session::{ComputeCommandResources, ComputeCommandSession, scan_work_items};
use crate::context::VulkanContext;
use crate::map::vk_buffer_handle;
use crate::pipelines::{PushConstants, VulkanComputePipelines, build_pipelines};
use crate::sorter::{
    VrdxSorter, VrdxSorterCreateInfo, VrdxSorterStorageRequirements, vrdxCmdSortKeyValue,
    vrdxCreateSorter, vrdxGetSorterKeyValueStorageRequirements,
};
use crate::stream::{Stream, StreamBackend};
use ash::vk;
use pelorus_core::PointXYZIT;
use std::mem::size_of;
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) struct DownsampleScratchBuffers {
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
    // See ScratchBuffers::sorter: the driver reclaims this at process exit.
    pub(crate) sorter: VrdxSorter,
    capacity: usize,
}

pub(crate) struct VulkanDownsampleState {
    pub(crate) context: Arc<VulkanContext>,
    compute: Arc<VulkanComputePipelines>,
    command_resources: Mutex<ComputeCommandResources>,
    dummy_buf: VulkanBuffer,
    pub(crate) scratch: DownsampleScratchBuffers,
    timestamp_pool: vk::QueryPool,
    timestamp_period: f32,
}

impl Drop for VulkanDownsampleState {
    fn drop(&mut self) {
        if !self.context.is_alive() {
            return;
        }
        unsafe {
            self.context
                .device
                .destroy_query_pool(self.timestamp_pool, None);
        }
    }
}

static DOWNSAMPLE_STATE: OnceLock<Mutex<Option<VulkanDownsampleState>>> = OnceLock::new();

pub(crate) fn downsample_state() -> &'static Mutex<Option<VulkanDownsampleState>> {
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
    let sorter_result = unsafe { vrdxCreateSorter(&sorter_ci, &mut sorter) };
    assert!(
        sorter_result == vk::Result::SUCCESS.as_raw() && sorter.0 != 0,
        "vk_radix_sort initialization failed for Vulkan downsampling with {}",
        vk::Result::from_raw(sorter_result)
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
        output_download: VulkanBuffer::new_host_readback(stream, size_of::<PointXYZIT>(), new_cap),
        run_starts: VulkanBuffer::new(stream, size_of::<u32>(), new_cap),
        run_ends: VulkanBuffer::new(stream, size_of::<u32>(), new_cap),
        unique_keys: VulkanBuffer::new(stream, size_of::<i32>(), new_cap),
        start_flags: VulkanBuffer::new(stream, size_of::<u32>(), new_cap),
        start_offsets: VulkanBuffer::new(stream, size_of::<u32>(), new_cap),
        block_sums: VulkanBuffer::new(stream, size_of::<u32>(), block_sums_cap),
        counter_upload: VulkanBuffer::new_host_visible(stream, size_of::<u32>(), 3),
        counter: VulkanBuffer::new(stream, size_of::<u32>(), 3),
        counter_download: VulkanBuffer::new_host_readback(stream, size_of::<u32>(), 3),
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

fn compare_downsample_points(a: &PointXYZIT, b: &PointXYZIT) -> std::cmp::Ordering {
    let compare = |lhs: f32, rhs: f32| lhs.partial_cmp(&rhs).unwrap_or(std::cmp::Ordering::Equal);
    compare(a.timestamp, b.timestamp)
        .then_with(|| compare(a.x, b.x))
        .then_with(|| compare(a.y, b.y))
        .then_with(|| compare(a.z, b.z))
        .then_with(|| compare(a.intensity, b.intensity))
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
        std::mem::size_of_val(filtered_cloud) as vk::DeviceSize,
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

    // Match the CPU tuple ordering within each voxel. The radix sorter is
    // stable, so sort secondary point fields from least to most significant,
    // then sort by the signed voxel hash.
    for component in [4u32, 3, 2, 1, 0] {
        params.j = component;
        session.dispatch(
            &state.compute,
            state.compute.downsample_timestamp_keys,
            &all_buffers,
            &params,
            selected_size as u32,
        );
        session.sort_barrier();
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
        session.sort_barrier();
    }

    params.j = 5;
    session.dispatch(
        &state.compute,
        state.compute.downsample_timestamp_keys,
        &all_buffers,
        &params,
        selected_size as u32,
    );
    session.sort_barrier();
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
    session.sort_barrier();
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
        scan_work_items(selected_size as u32),
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
        scan_work_items(selected_size as u32),
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
    out.sort_unstable_by(compare_downsample_points);
    let timestamp_sort_ms = sort_start.elapsed().as_secs_f64() * 1000.0;

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
            timestamp_sort_ms,
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
    out.sort_unstable_by(compare_downsample_points);

    (
        out,
        VulkanDownsampleTimings {
            total_ms: start.elapsed().as_secs_f64() * 1000.0,
            ..Default::default()
        },
    )
}
