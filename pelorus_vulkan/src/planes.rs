use crate::buffer::VulkanBuffer;
use crate::compute_session::{ComputeCommandSession, scan_work_items};
use crate::map::{
    MapBackendState, VulkanGpuMapState, ensure_scratch_capacity, vk_buffer_handle, with_state,
};
use crate::pipelines::PushConstants;
use crate::stream::Stream;
use crate::types::{NeighborType, PointNormal, Quaternion, Vector3};
use ash::vk;
use pelorus_core::PointXYZIT;
use pelorus_localmap::IVoxNearbyType;
use ros_pointcloud2::points::PointXYZINormal;
use std::mem::size_of;

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
        MapBackendState::Multiresolution(map) => map.planes(
            stream,
            cloud,
            ekf_pose_t,
            ekf_pose_r,
            ekf_imu_offset_t,
            ekf_imu_offset_r,
            out_buff,
            out_buff_gpu,
            download_buff,
        ),
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
                std::sync::Arc::ptr_eq(&cloud_ctx, &state.context),
                "cloud buffer and Vulkan map context mismatch"
            );
            assert!(
                out_buff_gpu.capacity() >= cloud.n,
                "GPU output buffer must fit cloud size"
            );

            ensure_scratch_capacity(state, stream, cloud.n);

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
                scan_work_items(cloud.n as u32),
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
                scan_work_items(cloud.n as u32),
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

/// Try to retrieve GPU timing measurements from previous operations.
/// (planes_ms, add_to_map_ms)
pub fn try_get_gpu_timings() -> Option<(f64, f64)> {
    with_state(|state| match &mut state.backend {
        MapBackendState::Host(_) | MapBackendState::Multiresolution(_) => None,
        MapBackendState::Vulkan(gpu) => {
            let mut planes_ms = None;
            let mut add_to_map_ms = None;

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
