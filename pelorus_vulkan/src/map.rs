use crate::buffer::{BufferStorage, VulkanBuffer};
use crate::compute_session::{
    ComputeCommandResources, ComputeCommandSession, record_portable_sort, scan_work_items,
};
use crate::context::{VulkanContext, global_vulkan_slot};
use crate::pipelines::{
    EMPTY_KEY, GpuVoxelDist, PushConstants, VulkanComputePipelines, build_pipelines,
};
use crate::sorter::{
    VrdxSorter, VrdxSorterCreateInfo, VrdxSorterStorageRequirements, vrdxCmdSortKeyValue,
    vrdxCreateSorter, vrdxDestroySorter, vrdxGetSorterKeyValueStorageRequirements,
};
use crate::stream::{Stream, StreamBackend};
use crate::types::{NeighborType, Quaternion, Vector3};
use ash::vk;
use pelorus_core::PointXYZIT;
use pelorus_localmap::IVoxHashMap;
use std::mem::size_of;
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) struct ScratchBuffers {
    pub(crate) keys: VulkanBuffer,
    pub(crate) indices: VulkanBuffer,
    packed: VulkanBuffer,
    pub(crate) run_starts: VulkanBuffer,
    pub(crate) run_ends: VulkanBuffer,
    pub(crate) unique_keys: VulkanBuffer,
    pub(crate) target_indices: VulkanBuffer,
    pub(crate) update_flags: VulkanBuffer,
    pub(crate) sorted_points: VulkanBuffer,
    pub(crate) counter: VulkanBuffer,
    pub(crate) sorter_scratch: VulkanBuffer,
    pub(crate) lru_mapping: VulkanBuffer,
    pub(crate) lru_keys: VulkanBuffer,
    pub(crate) is_new: VulkanBuffer,
    pub(crate) new_offset: VulkanBuffer,
    pub(crate) block_sums: VulkanBuffer,
    pub(crate) plane_compact: VulkanBuffer,
    pub(crate) plane_readback: VulkanBuffer,
    // Keep the native sorter process-lifetime. Some NVIDIA drivers crash when
    // its pipelines are explicitly destroyed during coordinated shutdown.
    pub(crate) sorter: VrdxSorter,
    pub(crate) capacity: usize,
}

pub(crate) struct VulkanGpuMapState {
    pub(crate) context: Arc<VulkanContext>,
    pub(crate) compute: Arc<VulkanComputePipelines>,
    pub(crate) command_resources: Mutex<ComputeCommandResources>,
    pub(crate) keys_buf: VulkanBuffer,
    pub(crate) values_buf: VulkanBuffer,
    pub(crate) dist_buf: VulkanBuffer,
    pub(crate) dist_keys_buf: VulkanBuffer,
    pub(crate) dummy_buf: VulkanBuffer,
    pub(crate) scratch: ScratchBuffers,
    pub(crate) dist_cap: u32,
    pub(crate) table_cap: u32,
    pub(crate) resolution: f32,
    pub(crate) sensitivity: f32,
    pub(crate) min_dist: f32,
    pub(crate) scan_number: u32,
    pub(crate) nbtype: u32,
    pub(crate) add_cursor: u32,
    pub(crate) lru_cursor: u32,
    pub(crate) timestamp_pool: vk::QueryPool,
    pub(crate) timestamp_period: f32,
    pub(crate) pending_planes_timing: std::sync::atomic::AtomicBool,
    pub(crate) pending_add_to_map_timing: std::sync::atomic::AtomicBool,
}

impl Drop for VulkanGpuMapState {
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

pub(crate) struct HostMapState {
    pub(crate) map: IVoxHashMap,
}

pub(crate) enum MapBackendState {
    Vulkan(VulkanGpuMapState),
    Host(HostMapState),
}

pub(crate) struct VulkanMapState {
    pub(crate) backend: MapBackendState,
}

static MAP_STATE: OnceLock<Mutex<Option<VulkanMapState>>> = OnceLock::new();

fn global_state() -> &'static Mutex<Option<VulkanMapState>> {
    MAP_STATE.get_or_init(|| Mutex::new(None))
}

pub(crate) fn with_state<R>(f: impl FnOnce(&mut VulkanMapState) -> R) -> R {
    let mut guard = global_state().lock().expect("vulkan map mutex poisoned");
    let state = guard
        .as_mut()
        .expect("vulkan map is not initialized; call init_map first");
    f(state)
}

pub(crate) fn requires_portable_map_sort(context: &VulkanContext) -> bool {
    let properties = unsafe {
        context
            .instance
            .get_physical_device_properties(context.physical_device)
    };
    properties.vendor_id == 0x8086
        && properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
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

    // Hold the map lock for the whole initialization. Releasing it between the
    // emptiness check and the store lets two callers both observe an empty map
    // and build a full state each; the loser's state is then dropped out from
    // under whoever already looked it up, and the winner can be stored over a
    // state that is mid-use. `ensure_downsample_state` already holds its guard
    // across construction; this path was the odd one out.
    let mut guard = global_state().lock().expect("vulkan map mutex poisoned");
    if guard.is_some() {
        return;
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
            let sorter_result = unsafe { vrdxCreateSorter(&sorter_ci, &mut sorter) };
            assert!(
                sorter_result == vk::Result::SUCCESS.as_raw() && sorter.0 != 0,
                "vk_radix_sort initialization failed with {}. \
                 This backend requires VK_KHR_push_descriptor support and a compatible Vulkan implementation.",
                vk::Result::from_raw(sorter_result)
            );

            let mut requirements = VrdxSorterStorageRequirements {
                size: 0,
                usage: vk::BufferUsageFlags::empty(),
            };
            unsafe {
                vrdxGetSorterKeyValueStorageRequirements(sorter, dist_cap, &mut requirements)
            };
            // The key/value layout is two element-sized in/out regions plus a
            // histogram, so anything at or below `dist_cap * 4` bytes means the
            // sorter did not report real requirements.
            assert!(
                requirements.size > u64::from(dist_cap) * 4
                    && requirements
                        .usage
                        .contains(vk::BufferUsageFlags::STORAGE_BUFFER),
                "vk_radix_sort reported invalid storage requirements ({} bytes for {dist_cap} elements). \
                 Refusing to continue to avoid undefined behavior.",
                requirements.size
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
                counter: VulkanBuffer::new_host_readback(stream, size_of::<u32>(), 3),
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
                plane_readback: VulkanBuffer::new_host_readback(
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

    *guard = Some(VulkanMapState { backend });
}

pub(crate) fn vk_buffer_handle(buf: &VulkanBuffer) -> vk::Buffer {
    match &buf.storage {
        BufferStorage::Vulkan { buffer, .. } => *buffer,
        BufferStorage::Host { .. } => panic!("expected Vulkan buffer"),
    }
}

pub(crate) fn ensure_scratch_capacity(
    state: &mut VulkanGpuMapState,
    stream: &Stream,
    n_points: usize,
) {
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
        VulkanBuffer::new_host_readback(stream, size_of::<crate::types::PointNormal>(), new_cap);
    // block_sums needs ceil(new_cap / 512) elements, round up to next power of 2 for safety
    let block_sums_cap = (new_cap / 512 + 1).next_power_of_two().max(1024);
    state.scratch.block_sums = VulkanBuffer::new(stream, size_of::<u32>(), block_sums_cap);

    let mut requirements = VrdxSorterStorageRequirements {
        size: 0,
        usage: vk::BufferUsageFlags::empty(),
    };
    let sort_capacity = new_cap.max(state.dist_cap as usize);
    assert!(
        state.scratch.sorter.0 != 0,
        "Vulkan map scratch has a null radix sorter; the map state was not initialized by init_map"
    );
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
        session.sort_barrier();
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
        session.sort_barrier();
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
            unsafe {
                state.context.device.cmd_reset_query_pool(
                    session.command_buffer(),
                    state.timestamp_pool,
                    2,
                    2,
                );
            }
            session.write_timestamp(state.timestamp_pool, 2);

            session.dispatch(
                &state.compute,
                state.compute.calculate_keys,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

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
                session.sort_barrier();
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
                session.sort_barrier();
            }

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
                scan_work_items(n_points as u32),
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
                scan_work_items(n_points as u32),
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

            session.dispatch(
                &state.compute,
                state.compute.classify_keys,
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
                scan_work_items(n_points as u32),
            );
            session.barrier();

            session.dispatch(
                &state.compute,
                state.compute.scan_block_sums,
                &all_buffers,
                &params,
                1, // Single workgroup
            );
            session.barrier();

            session.dispatch(
                &state.compute,
                state.compute.add_block_prefix,
                &all_buffers,
                &params,
                scan_work_items(n_points as u32),
            );
            session.barrier();

            session.dispatch(
                &state.compute,
                state.compute.assign_new_targets,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            session.dispatch(
                &state.compute,
                state.compute.finalize_cursors,
                &all_buffers,
                &params,
                1, // Single thread
            );
            session.barrier();

            session.dispatch(
                &state.compute,
                state.compute.reorder_points,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

            session.dispatch(
                &state.compute,
                state.compute.build_voxels,
                &all_buffers,
                &params,
                n_points as u32,
            );
            session.barrier();

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

/// Live embedder sessions, so `session_release` only tears down at the last one.
static LIVE_SESSIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Register a session that is using the Vulkan backend.
///
/// Embedders that create and drop handles (the C API, the Python module) pair this
/// with [`session_release`], so that dropping one handle cannot destroy the device
/// out from under another that is still running.
pub fn session_retain() {
    LIVE_SESSIONS.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

/// End a session registered with [`session_retain`], running [`shutdown`] when the
/// last one ends.
pub fn session_release() {
    if LIVE_SESSIONS
        .fetch_sub(1, std::sync::atomic::Ordering::AcqRel)
        .saturating_sub(1)
        == 0
    {
        shutdown();
    }
}

/// Release every Vulkan object this crate owns, in order, before the process exits.
///
/// `MAP_STATE` and `DOWNSAMPLE_STATE` are `OnceLock` statics, and Rust never drops
/// statics. Each holds an `Arc<VulkanContext>`, so without this the refcount never
/// reaches zero, `VulkanContext::drop` never runs, and the process reaches `exit()`
/// with a live `VkInstance`. The NVIDIA driver then tears itself down from an
/// `atexit` handler while its own worker threads are still parked in the driver,
/// and one of them faults on state the handler already freed.
///
/// Call this once, at the end of the run, after backend work has been quiesced.
/// Any Vulkan call afterwards will panic; the map has to be re-initialized with
/// `init_map` first.
pub fn shutdown() {
    let map = global_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    let downsample = crate::downsample::downsample_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();

    let context = match (&map, &downsample) {
        (
            Some(VulkanMapState {
                backend: MapBackendState::Vulkan(gpu),
            }),
            _,
        ) => Some(Arc::clone(&gpu.context)),
        (_, Some(state)) => Some(Arc::clone(&state.context)),
        // Neither state exists, but a context may still be live: `Stream`s are handed
        // out for downsampling before any map is initialized.
        _ => global_vulkan_slot()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone(),
    };

    let Some(context) = context else {
        return;
    };

    // The sorters own device pipelines and layouts, so they have to go while the
    // device is still alive, and only once the GPU is finished with them.
    unsafe {
        let _ = context.device.device_wait_idle();
    }

    let mut map = map;
    if let Some(VulkanMapState {
        backend: MapBackendState::Vulkan(gpu),
    }) = &mut map
    {
        unsafe { vrdxDestroySorter(gpu.scratch.sorter) };
        gpu.scratch.sorter = VrdxSorter(0);
    }
    let mut downsample = downsample;
    if let Some(state) = &mut downsample {
        unsafe { vrdxDestroySorter(state.scratch.sorter) };
        state.scratch.sorter = VrdxSorter(0);
    }

    // Buffers, pipelines, descriptor pools and query pools go here, while the
    // device is still alive.
    drop(map);
    drop(downsample);

    global_vulkan_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();

    // Destroy the device explicitly rather than waiting for the refcount: other
    // threads may still hold `Stream`s or `VulkanBuffer`s when `main` returns, and
    // those objects would otherwise keep the instance alive past `exit()`. Their
    // `Drop` impls check `VulkanContext::is_alive` and become no-ops from here on.
    context.destroy_now();
    drop(context);
}
