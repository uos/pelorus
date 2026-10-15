//! Device-resident distribution pyramid.
use crate::buffer::VulkanBuffer;
use crate::compute_session::{ComputeCommandResources, ComputeCommandSession, scan_work_items};
use crate::context::VulkanContext;
use crate::map::vk_buffer_handle as handle;
use crate::pipelines::{PushConstants, THREADS_PER_GROUP, VulkanComputePipelines, build_pipelines};
use crate::planes::VulkanPlaneTimings;
use crate::stream::{Stream, StreamBackend};
use crate::types::{PointNormal, Pose, Quaternion, Vector3};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use ros_pointcloud2::points::PointXYZINormal;
use std::mem::size_of;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct Cell {
    key: [i32; 3],
    scan: u32,
    count: u64,
    mean: [f32; 3],
    cov: [f32; 9],
}
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct Fit {
    status: u32,
    mean: [f32; 3],
    normal: [f32; 3],
    tangent: [f32; 6],
    extent: [f32; 2],
    quality: f32,
    d: f32,
}
const _: () = assert!(size_of::<Cell>() == 72 && size_of::<Fit>() == 68);
// MRMissSlot (state, key, fit) and MRMissWork (level, slot, key) in scalar layout.
const MISS_SLOT_SIZE: usize = 4 + 12 + size_of::<Fit>();
const MISS_WORK_SIZE: usize = 4 + 4 + 12;
struct Level {
    cells: VulkanBuffer,
    fits: VulkanBuffer,
    table: VulkanBuffer,
    // The *_next buffers hold the level while a scan is merged, so unchanged
    // fits can be carried over from the previous version instead of refitted.
    cells_next: VulkanBuffer,
    fits_next: VulkanBuffer,
    table_next: VulkanBuffer,
    n: u32,
    budget: u32,
}
struct Scratch {
    work: VulkanBuffer,
    miss_work: VulkanBuffer,
    reduced: VulkanBuffer,
    batches: VulkanBuffer,
    combined: VulkanBuffer,
    indices: VulkanBuffer,
    flags: VulkanBuffer,
    offsets: VulkanBuffer,
    sums: VulkanBuffer,
    planes: VulkanBuffer,
    plane_readback: VulkanBuffer,
    counter: VulkanBuffer,
    poses: VulkanBuffer,
    cap: usize,
}

/// Lidar-frame query points and the poses that move them into the map frame.
pub(crate) struct PlaneQuery<'a> {
    pub(crate) cloud: &'a VulkanBuffer,
    pub(crate) pose: &'a Pose,
    pub(crate) lidar_to_imu: &'a Pose,
}

pub(crate) struct MultiResolutionMap {
    pub(crate) context: Arc<VulkanContext>,
    compute: Arc<VulkanComputePipelines>,
    resources: Mutex<ComputeCommandResources>,
    levels: [Level; 4],
    scratch: Scratch,
    // Fits of absent cells for all levels, valid for one map version.
    miss: VulkanBuffer,
    // Number of listed misses.
    miss_counter: VulkanBuffer,
    // Map version for the miss cache, advanced after every map update.
    version: u32,
    // Tests skip filling the cache, so the query fits every absent cell inline.
    #[cfg(test)]
    bypass_miss_cache: bool,
    base: f32,
    sensitivity: f32,
}

impl MultiResolutionMap {
    pub(crate) fn new(stream: &Stream, capacity: usize, base: f32, sensitivity: f32) -> Self {
        assert!(capacity >= 4 && capacity <= (u32::MAX / 2) as usize);
        assert!(base.is_finite() && base > 0.0);
        let StreamBackend::Vulkan(context) = &stream.backend else {
            panic!("multiresolution requires a Vulkan device");
        };
        assert!(
            context.precise_multiresolution,
            "Vulkan multiresolution requires shaderFloat64 and shaderInt64"
        );
        let compute = build_pipelines(Arc::clone(context));
        let alloc = |size, n| VulkanBuffer::new_device_local_required(stream, size, n);
        let levels: [Level; 4] = std::array::from_fn(|i| {
            let budget = capacity / 4 + usize::from(i < capacity % 4);
            let table_n = budget.next_power_of_two() * 2;
            Level {
                cells: alloc(size_of::<Cell>(), budget),
                fits: alloc(size_of::<Fit>(), budget),
                table: alloc(4, table_n),
                cells_next: alloc(size_of::<Cell>(), budget),
                fits_next: alloc(size_of::<Fit>(), budget),
                table_next: alloc(4, table_n),
                n: 0,
                budget: budget as u32,
            }
        });
        // Each level caches as many absent cells as its hash table has slots.
        let miss_n = levels.iter().map(|l| l.table.n).sum::<usize>();
        let map = Self {
            context: Arc::clone(context),
            resources: Mutex::new(ComputeCommandResources::new(context, &compute)),
            compute,
            levels,
            miss: alloc(MISS_SLOT_SIZE, miss_n),
            miss_counter: alloc(4, 1),
            version: 1,
            #[cfg(test)]
            bypass_miss_cache: false,
            base,
            sensitivity,
            scratch: Scratch {
                work: alloc(size_of::<Cell>(), 1),
                miss_work: alloc(MISS_WORK_SIZE, 1),
                reduced: alloc(size_of::<Cell>(), 1),
                batches: alloc(size_of::<Cell>(), 1),
                combined: alloc(size_of::<Cell>(), 1),
                indices: alloc(4, 1),
                flags: alloc(4, 1),
                offsets: alloc(4, 1),
                sums: alloc(4, 1),
                planes: alloc(size_of::<PointNormal>(), 1),
                plane_readback: VulkanBuffer::new_host_readback(
                    stream,
                    size_of::<PointNormal>(),
                    1,
                ),
                counter: VulkanBuffer::new_host_visible(stream, 4, 1),
                poses: VulkanBuffer::new_host_visible(stream, 8, 7),
                cap: 1,
            },
        };
        // A dispatch for version 0, which is never current, clears the cache.
        let mut b = map.buffers();
        b[16] = handle(&map.miss);
        b[17] = handle(&map.miss_counter);
        let p = map.params(miss_n as u32, 0);
        let mut session = map.begin();
        map.dispatch(&mut session, 13, &b, &p, p.n);
        session.submit();
        map
    }
    fn ensure(&mut self, stream: &Stream, n: usize) {
        let cap = n.max(1).next_power_of_two();
        if cap <= self.scratch.cap {
            return;
        }
        let alloc = |size| VulkanBuffer::new_device_local_required(stream, size, cap);
        self.scratch.work = alloc(size_of::<Cell>());
        // Up to four absent cells per query point.
        self.scratch.miss_work =
            VulkanBuffer::new_device_local_required(stream, MISS_WORK_SIZE, 4 * cap);
        self.scratch.reduced = alloc(size_of::<Cell>());
        self.scratch.batches = alloc(size_of::<Cell>());
        self.scratch.combined = alloc(size_of::<Cell>());
        self.scratch.indices = alloc(4);
        self.scratch.flags = alloc(4);
        self.scratch.offsets = alloc(4);
        self.scratch.sums = VulkanBuffer::new_device_local_required(stream, 4, cap.div_ceil(512));
        self.scratch.planes = alloc(size_of::<PointNormal>());
        self.scratch.plane_readback =
            VulkanBuffer::new_host_readback(stream, size_of::<PointNormal>(), cap);
        self.scratch.cap = cap;
    }
    fn params(&self, n: u32, scan: u32) -> PushConstants {
        PushConstants {
            resolution: self.base,
            sensitivity: self.sensitivity,
            n,
            cap: n.max(1).next_power_of_two(),
            scan_number: scan,
            ..Default::default()
        }
    }
    fn buffers(&self) -> [vk::Buffer; 18] {
        let dummy = handle(&self.scratch.counter);
        let mut b = [dummy; 18];
        b[2] = handle(&self.scratch.indices);
        b[10] = dummy;
        b[14] = handle(&self.scratch.poses);
        b[15] = handle(&self.scratch.flags);
        b[16] = handle(&self.scratch.offsets);
        b[17] = handle(&self.scratch.sums);
        b
    }
    fn begin(&self) -> ComputeCommandSession<'_> {
        ComputeCommandSession::begin(
            &self.context,
            &self.resources,
            handle(&self.scratch.counter),
        )
    }
    fn dispatch(
        &self,
        session: &mut ComputeCommandSession<'_>,
        kernel: usize,
        b: &[vk::Buffer],
        p: &PushConstants,
        n: u32,
    ) {
        session.dispatch(&self.compute, self.compute.multiresolution[kernel], b, p, n);
        session.barrier();
    }
    fn sort(
        &self,
        session: &mut ComputeCommandSession<'_>,
        b: &[vk::Buffer],
        params: &PushConstants,
        eviction: bool,
    ) {
        let mut p = *params;
        p.nbtype = u32::from(eviction);
        p.k = 0;
        self.dispatch(session, 1, b, &p, p.cap);
        // The initialization dispatch sorts each workgroup in shared memory.
        // Only comparisons spanning workgroups need separate dispatches.
        let mut k = THREADS_PER_GROUP * 2;
        while k <= p.cap {
            p.k = k;
            let mut j = k / 2;
            while j > 0 {
                p.j = j;
                self.dispatch(session, 1, b, &p, p.cap);
                if j < THREADS_PER_GROUP {
                    // This dispatch finishes every remaining stride in its tile.
                    break;
                }
                j /= 2;
            }
            k *= 2;
        }
    }
    fn prefix(&self, session: &mut ComputeCommandSession<'_>, b: &[vk::Buffer], p: &PushConstants) {
        for (pipeline, items) in [
            (self.compute.exclusive_scan, scan_work_items(p.n)),
            (self.compute.scan_block_sums, 1),
            (self.compute.add_block_prefix, scan_work_items(p.n)),
        ] {
            session.dispatch(&self.compute, pipeline, b, p, items);
            session.barrier();
        }
    }
    fn read_count(&self, stream: &Stream) -> u32 {
        let mut count = [0];
        self.scratch.counter.download_into(stream, &mut count);
        count[0]
    }
    // Modes: 2 raw two-pass statistics, 0 parent pooling, 3 temporal update.
    fn record_reduce(
        &self,
        session: &mut ComputeCommandSession<'_>,
        n: u32,
        scan: u32,
        mode: u32,
        output: vk::Buffer,
        sorted: bool,
    ) {
        let mut p = self.params(n, scan);
        let mut b = self.buffers();
        b[0] = handle(&self.scratch.work);
        b[1] = handle(&self.scratch.reduced);
        if !sorted {
            self.sort(session, &b, &p, false);
        }
        p.nbtype = mode;
        self.dispatch(session, 2, &b, &p, n);
        self.prefix(session, &b, &p);
        b[0] = handle(&self.scratch.reduced);
        b[1] = output;
        self.dispatch(session, 3, &b, &p, n);
        session.dispatch(&self.compute, self.compute.plane_count, &b, &p, 1);
        session.shader_to_readback_barrier();
    }
    pub(crate) fn add(
        &mut self,
        stream: &Stream,
        cloud: &VulkanBuffer,
        t: &Vector3,
        q: &Quaternion,
        scan: u32,
    ) {
        let n = u32::try_from(cloud.n).expect("too many points");
        if n == 0 {
            return;
        }
        let max_budget = self.levels.iter().map(|l| l.budget).max().unwrap() as usize;
        self.ensure(stream, cloud.n + max_budget);
        self.scratch
            .poses
            .upload(stream, &[q.x, q.y, q.z, q.w, t.x, t.y, t.z]);
        let mut b = self.buffers();
        b[0] = handle(&self.scratch.work);
        b[13] = handle(cloud);
        let p = self.params(n, scan);
        self.scratch.counter.upload(stream, &[n]);
        let mut session = self.begin();
        self.dispatch(&mut session, 0, &b, &p, n);
        self.record_reduce(
            &mut session,
            n,
            scan,
            2,
            handle(&self.scratch.batches),
            false,
        );
        session.submit();
        let mut batch_n = self.read_count(stream);
        if batch_n == 0 {
            return;
        }
        for level in 0..4 {
            if level > 0 {
                let mut b = self.buffers();
                b[0] = handle(&self.scratch.batches);
                b[1] = handle(&self.scratch.work);
                let p = self.params(batch_n, scan);
                self.scratch.counter.upload(stream, &[batch_n]);
                let mut session = self.begin();
                self.dispatch(&mut session, 4, &b, &p, batch_n);
                self.record_reduce(
                    &mut session,
                    batch_n,
                    scan,
                    0,
                    handle(&self.scratch.batches),
                    false,
                );
                session.submit();
                batch_n = self.read_count(stream);
            }
            let old_n = self.levels[level].n;
            let total = old_n + batch_n;
            let mut p = self.params(total, scan);
            p.valid_size = old_n;
            let mut b = self.buffers();
            b[0] = handle(&self.levels[level].cells);
            b[3] = handle(&self.scratch.batches);
            b[1] = handle(&self.scratch.work);
            self.scratch.counter.upload(stream, &[total]);
            let mut session = self.begin();
            // Both inputs are sorted; equal keys keep old cells first. The merge
            // writes identity indices, so temporal reduction needs no key sort.
            self.dispatch(&mut session, 10, &b, &p, total);
            self.record_reduce(
                &mut session,
                total,
                scan,
                3,
                handle(&self.scratch.combined),
                true,
            );
            session.submit();
            let merged_n = self.read_count(stream);
            let excess = merged_n.saturating_sub(self.levels[level].budget);
            let retained = merged_n - excess;
            let l = &self.levels[level];
            let mut b = self.buffers();
            b[0] = handle(&self.scratch.combined);
            b[1] = handle(&l.cells_next);
            let mut p = self.params(merged_n, scan);
            let mut session = self.begin();
            if excess > 0 {
                // Rank the merged cells by eviction order, then drop the evicted
                // ones with an order-preserving compaction: the merge is already
                // key-sorted, so no second sort is needed. The counter still
                // holds merged_n for the prefix sum.
                self.sort(&mut session, &b, &p, true);
                p.valid_size = excess;
                let mut evict = b;
                evict[1] = handle(&self.scratch.work);
                self.dispatch(&mut session, 11, &evict, &p, merged_n);
                self.prefix(&mut session, &b, &p);
                self.dispatch(&mut session, 3, &b, &p, merged_n);
            } else {
                p.valid_size = merged_n;
                self.dispatch(&mut session, 5, &b, &p, merged_n);
            }
            p = self.params(retained, scan);
            p.cap = l.table_next.n as u32;
            let mut index_buffers = b;
            index_buffers[0] = handle(&l.cells_next);
            index_buffers[1] = handle(&l.table_next);
            self.dispatch(&mut session, 9, &index_buffers, &p, p.cap);
            p.k = 1;
            self.dispatch(&mut session, 9, &index_buffers, &p, retained);
            // Only fits next to this scan's updates, insertions and evictions
            // change; flags marks them dirty.
            let mut fit = self.buffers();
            fit[0] = handle(&l.cells_next);
            fit[3] = handle(&l.table_next);
            fit[8] = handle(&l.fits_next);
            fit[5] = handle(&l.cells);
            fit[9] = handle(&l.fits);
            fit[16] = handle(&l.table);
            p = self.params(retained, scan);
            p.table_cap = l.table_next.n as u32;
            p.j = 1;
            self.dispatch(&mut session, 7, &fit, &p, retained);
            p.j = 2;
            for (changed, n) in [
                (&self.scratch.batches, batch_n),
                (&self.scratch.work, excess),
            ] {
                fit[4] = handle(changed);
                p.n = n;
                self.dispatch(&mut session, 7, &fit, &p, n);
            }
            p.j = 3;
            p.n = retained;
            p.valid_size = l.n;
            self.dispatch(&mut session, 7, &fit, &p, retained);
            session.shader_to_readback_barrier();
            session.submit();
            let l = &mut self.levels[level];
            std::mem::swap(&mut l.cells, &mut l.cells_next);
            std::mem::swap(&mut l.fits, &mut l.fits_next);
            std::mem::swap(&mut l.table, &mut l.table_next);
            l.n = retained;
        }
        // The map changed, so cached fits of absent cells are stale.
        self.version += 1;
    }
    pub(crate) fn planes(
        &mut self,
        stream: &Stream,
        query: PlaneQuery<'_>,
        out: &mut Vec<PointXYZINormal>,
        out_gpu: &mut VulkanBuffer,
        download: &mut [PointNormal],
    ) -> VulkanPlaneTimings {
        let PlaneQuery {
            cloud,
            pose: Pose {
                position: t,
                orientation: q,
            },
            lidar_to_imu:
                Pose {
                    position: ext_t,
                    orientation: ext_q,
                },
        } = query;
        let n = u32::try_from(cloud.n).expect("too many query points");
        if n == 0 {
            out.clear();
            return VulkanPlaneTimings::default();
        }
        assert!(out_gpu.capacity() >= cloud.n && download.len() >= cloud.n);
        out_gpu.n = cloud.n;
        self.ensure(stream, cloud.n);
        self.scratch.counter.upload(stream, &[n]);
        let mut p = self.params(n, 0);
        p.pose_quat = [q.x as f32, q.y as f32, q.z as f32, q.w as f32];
        p.pose_trans = [t.x as f32, t.y as f32, t.z as f32];
        p.lidar_to_imu_quat = [
            ext_q.x as f32,
            ext_q.y as f32,
            ext_q.z as f32,
            ext_q.w as f32,
        ];
        p.lidar_to_imu_pos = [ext_t.x as f32, ext_t.y as f32, ext_t.z as f32];
        p.dist_cap = self.levels[0].n;
        p.table_cap = self.levels[1].n;
        p.valid_size = self.levels[2].n;
        p.unique_key_count = self.levels[3].n;
        // Query uses the otherwise-unused sort fields for hash capacities;
        // level sizes above remain separate from these power-of-two tables.
        p.cap = self.levels[0].table.n as u32;
        p.nbtype = self.levels[1].table.n as u32;
        p.k = self.levels[2].table.n as u32;
        p.j = self.levels[3].table.n as u32;
        let mut b = self.buffers();
        b[13] = handle(cloud);
        b[1] = handle(&self.scratch.planes);
        b[3] = handle(&self.levels[0].table);
        b[0] = handle(&self.levels[1].table);
        b[2] = handle(&self.levels[2].table);
        b[14] = handle(&self.levels[3].table);
        for i in 0..4 {
            b[4 + i] = handle(&self.levels[i].cells);
            b[[8, 9, 11, 12][i]] = handle(&self.levels[i].fits);
        }
        // Bindings 16 and 17 are free in the query itself and hold the miss
        // cache; the version selects its valid slots.
        let mut query = b;
        query[16] = handle(&self.miss);
        query[17] = handle(&self.miss_counter);
        let mut collect = query;
        collect[1] = handle(&self.scratch.miss_work);
        p.scan_number = self.version;
        let start = Instant::now();
        let mut session = self.begin();
        // Absent cells are listed and fitted once per map version, so the query
        // does not diverge on inline fits in every registration iteration.
        #[cfg(test)]
        let fill_cache = !self.bypass_miss_cache;
        #[cfg(not(test))]
        let fill_cache = true;
        if fill_cache {
            self.dispatch(&mut session, 12, &collect, &p, n);
            self.dispatch(&mut session, 13, &collect, &p, 4 * n);
        }
        self.dispatch(&mut session, 8, &query, &p, n);
        self.prefix(&mut session, &b, &p);
        let mut compact = b;
        compact[15] = handle(&self.scratch.planes);
        compact[17] = handle(out_gpu);
        session.dispatch(&self.compute, self.compute.compact_planes, &compact, &p, n);
        session.barrier();
        session.dispatch(&self.compute, self.compute.plane_count, &b, &p, 1);
        session.shader_to_readback_barrier();
        // Copy candidates in the same submission as compaction. The count is
        // read only after its fence signals, avoiding a second transfer submit
        // and a staging allocation on every registration iteration.
        session.copy_buffer(
            handle(out_gpu),
            handle(&self.scratch.plane_readback),
            (cloud.n * size_of::<PointNormal>()) as vk::DeviceSize,
        );
        session.transfer_to_host_barrier();
        session.submit();
        let kernel_compact_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let count = self.read_count(stream) as usize;
        assert!(count <= cloud.n && count <= out_gpu.n);
        self.scratch
            .plane_readback
            .download_into(stream, &mut download[..count]);
        out.clear();
        out.extend(download[..count].iter().copied().map(PointXYZINormal::from));
        VulkanPlaneTimings {
            kernel_compact_ms,
            download_ms: start.elapsed().as_secs_f64() * 1000.,
        }
    }
    pub(crate) fn capacity(&self) -> u32 {
        self.levels.iter().map(|l| l.budget).sum()
    }
    pub(crate) fn size(&self) -> u32 {
        self.levels.iter().map(|l| l.n).sum()
    }
    pub(crate) fn get_map(&self, stream: &Stream) -> Vec<nalgebra::Point3<f32>> {
        self.get_map_with_recency(stream)
            .into_iter()
            .map(|(p, _, _)| p)
            .collect()
    }
    /// Downloads every resolution level, tagging each point with the level
    /// it lives at (0 = finest, 3 = coarsest) and the scan number it was
    /// last touched on.
    pub(crate) fn get_map_with_recency(
        &self,
        stream: &Stream,
    ) -> Vec<(nalgebra::Point3<f32>, u8, u32)> {
        let mut out = Vec::with_capacity(self.levels.iter().map(|l| l.n as usize).sum());
        for (level, l) in self.levels.iter().enumerate() {
            if l.n == 0 {
                continue;
            }
            let mut cells = vec![Cell::default(); l.n as usize];
            l.cells.download_into(stream, &mut cells);
            out.extend(cells.into_iter().map(|c| {
                (
                    nalgebra::Point3::new(c.mean[0], c.mean[1], c.mean[2]),
                    level as u8,
                    c.scan,
                )
            }));
        }
        out
    }
}

#[cfg(all(test, feature = "vulkan_enable"))]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn nearly_isotropic_covariance_preserves_f64_tangent_split() {
        let stream = Stream::new();
        let StreamBackend::Vulkan(context) = &stream.backend else {
            return;
        };
        if !context.precise_multiresolution {
            return;
        }
        let mut map = MultiResolutionMap::new(&stream, 37, 0.5, 10.0);
        map.levels[0].cells.upload(
            &stream,
            &[Cell {
                count: 20,
                cov: [0.1, 1e-8, 0.0, 1e-8, 0.1, 0.0, 0.0, 0.0, 1e-7],
                ..Cell::default()
            }],
        );
        let mut b = map.buffers();
        b[0] = handle(&map.levels[0].cells);
        b[1] = handle(&map.levels[0].table);
        let mut p = map.params(1, 1);
        p.cap = map.levels[0].table.n as u32;
        let mut session = map.begin();
        map.dispatch(&mut session, 9, &b, &p, p.cap);
        p.k = 1;
        map.dispatch(&mut session, 9, &b, &p, 1);
        b[3] = handle(&map.levels[0].table);
        b[8] = handle(&map.levels[0].fits);
        p.table_cap = p.cap;
        map.dispatch(&mut session, 7, &b, &p, 1);
        session.shader_to_readback_barrier();
        session.submit();
        map.levels[0].fits.n = 1;
        let mut fits = vec![Fit::default(); 1];
        map.levels[0].fits.download_into(&stream, &mut fits);
        let fit = fits[0];
        assert_eq!(fit.status, 2);
        assert!(fit.normal[2].abs() > 0.999999);
        let contains = |delta: [f32; 3]| {
            (0..2).all(|i| {
                let t = &fit.tangent[3 * i..3 * i + 3];
                ((t[0] * delta[0] + t[1] * delta[1]) + t[2] * delta[2]).abs() <= fit.extent[i]
            })
        };
        assert!(!contains([0.9, 0.9, 0.0]));
        assert!(contains([0.9, 0.0, 0.0]));
    }

    #[test]
    fn empty_map_queries_do_not_read_uninitialized_hash_tables() {
        let stream = Stream::new();
        let StreamBackend::Vulkan(context) = &stream.backend else {
            return;
        };
        if !context.precise_multiresolution {
            return;
        }
        let mut map = MultiResolutionMap::new(&stream, 37, 0.5, 10.0);
        let points = vec![
            pelorus_core::PointXYZIT {
                x: -1.0,
                y: 2.0,
                z: 0.0,
                ..Default::default()
            };
            257
        ];
        let cloud = VulkanBuffer::from_cloud(&stream, &points);
        let mut output = VulkanBuffer::new(&stream, size_of::<PointNormal>(), points.len());
        let mut download = vec![PointNormal::default(); points.len()];
        let mut planes = vec![PointXYZINormal::default(); points.len()];
        let pose = Pose {
            position: Vector3::new(0., 0., 0.),
            orientation: Quaternion::default(),
        };
        map.planes(
            &stream,
            PlaneQuery {
                cloud: &cloud,
                pose: &pose,
                lidar_to_imu: &pose,
            },
            &mut planes,
            &mut output,
            &mut download,
        );
        assert!(planes.is_empty());
    }

    #[test]
    fn hash_index_preserves_colliding_signed_keys() {
        let stream = Stream::new();
        let StreamBackend::Vulkan(context) = &stream.backend else {
            return;
        };
        if !context.precise_multiresolution {
            return;
        }
        let hash = |key: [i32; 3]| {
            let mut h = (key[0] as u32)
                .wrapping_mul(73856093)
                .wrapping_add((key[1] as u32).wrapping_mul(471943))
                .wrapping_add((key[2] as u32).wrapping_mul(83492791));
            h ^= h >> 16;
            h = h.wrapping_mul(0x85ebca6b);
            h ^= h >> 13;
            h = h.wrapping_mul(0xc2b2ae35);
            h ^ (h >> 16)
        };
        let mut map = MultiResolutionMap::new(&stream, 1025, 0.5, 10.0);
        let capacity = map.levels[0].table.n as u32;
        let mask = capacity - 1;
        let cells: Vec<_> = (i32::MIN..)
            .map(|x| [x, -17, i32::MAX])
            .filter(|&key| hash(key) & mask == mask)
            .take(128)
            .map(|key| Cell {
                key,
                count: 1,
                ..Cell::default()
            })
            .collect();
        map.ensure(&stream, cells.len());
        map.levels[0].cells.upload(&stream, &cells);
        let mut buffers = map.buffers();
        buffers[0] = handle(&map.levels[0].cells);
        buffers[1] = handle(&map.levels[0].table);
        let mut params = map.params(cells.len() as u32, 1);
        params.cap = capacity;
        let mut session = map.begin();
        map.dispatch(&mut session, 9, &buffers, &params, capacity);
        params.k = 1;
        map.dispatch(&mut session, 9, &buffers, &params, params.n);
        session.shader_to_readback_barrier();
        session.submit();
        let mut table = vec![0u32; capacity as usize];
        map.levels[0].table.download_into(&stream, &mut table);
        assert_eq!(
            table.iter().filter(|&&i| i != u32::MAX).count(),
            cells.len()
        );
        for (i, cell) in cells.iter().enumerate() {
            let mut slot = hash(cell.key) & mask;
            loop {
                let found = table[slot as usize];
                assert_ne!(found, u32::MAX, "missing colliding key {:?}", cell.key);
                if found == i as u32 {
                    break;
                }
                slot = (slot + 1) & mask;
            }
        }
    }

    #[test]
    fn stable_merge_preserves_equal_key_order_and_payloads() {
        let stream = Stream::new();
        let StreamBackend::Vulkan(context) = &stream.backend else {
            return;
        };
        if !context.precise_multiresolution {
            return;
        }
        let mut map = MultiResolutionMap::new(&stream, 37, 0.5, 10.0);
        for (old_n, batch_n) in [
            (0, 17),
            (17, 0),
            (1, 1),
            (127, 129),
            (257, 255),
            (1023, 2049),
        ] {
            let n = old_n + batch_n;
            map.ensure(&stream, n);
            let cells = |count, offset| {
                let mut cells: Vec<_> = (0..count)
                    .map(|i| Cell {
                        key: [
                            if i % 17 == 0 {
                                i32::MIN
                            } else if i % 29 == 0 {
                                i32::MAX
                            } else {
                                (i % 13) as i32 - 7
                            },
                            (i % 3) as i32 - 1,
                            0,
                        ],
                        scan: (i + offset) as u32,
                        count: (1u64 << 40) + (i + offset) as u64,
                        mean: [i as f32, offset as f32, 0.],
                        ..Cell::default()
                    })
                    .collect();
                cells.sort_by_key(|c| c.key);
                cells
            };
            let old = cells(old_n, 0);
            let batch = cells(batch_n, old_n);
            map.scratch.combined.upload(&stream, &old);
            map.scratch.batches.upload(&stream, &batch);
            let mut buffers = map.buffers();
            buffers[0] = handle(&map.scratch.combined);
            buffers[3] = handle(&map.scratch.batches);
            buffers[1] = handle(&map.scratch.work);
            let mut params = map.params(n as u32, 1);
            params.valid_size = old_n as u32;
            let mut session = map.begin();
            map.dispatch(&mut session, 10, &buffers, &params, n as u32);
            session.shader_to_readback_barrier();
            session.submit();
            let mut merged = vec![Cell::default(); n];
            map.scratch.work.download_into(&stream, &mut merged);
            let mut expected = old;
            expected.extend(batch);
            expected.sort_by_key(|c| c.key);
            assert_eq!(
                bytemuck::cast_slice::<Cell, u8>(&merged),
                bytemuck::cast_slice::<Cell, u8>(&expected),
                "merge mismatch at {old_n}/{batch_n}"
            );
            let mut indices = vec![0u32; n];
            map.scratch.indices.download_into(&stream, &mut indices);
            assert_eq!(indices, (0..n as u32).collect::<Vec<_>>());
        }
    }

    #[test]
    fn tiled_sort_preserves_stable_keys_and_exact_eviction_order() {
        let stream = Stream::new();
        let StreamBackend::Vulkan(context) = &stream.backend else {
            return;
        };
        if !context.precise_multiresolution {
            return;
        }
        let mut map = MultiResolutionMap::new(&stream, 40003, 0.5, 10.0);
        let scan = u32::MAX;
        for n in [1, 17, 255, 256, 257, 1023, 4097, 20037] {
            map.ensure(&stream, n);
            let cells: Vec<_> = (0..n)
                .map(|i| Cell {
                    key: [
                        (i.wrapping_mul(2654435761) % 31) as i32 - 15,
                        (i % 7) as i32 - 3,
                        (i % 3) as i32 - 1,
                    ],
                    scan: (i as u32).wrapping_mul(2246822519),
                    count: if i % 97 == 0 {
                        0
                    } else {
                        (1u64 << 60) + i as u64
                    },
                    ..Cell::default()
                })
                .collect();
            map.scratch.work.upload(&stream, &cells);
            for eviction in [false, true] {
                let compare = |a: u32, b: u32| {
                    let (ca, cb) = (&cells[a as usize], &cells[b as usize]);
                    match (ca.count != 0, cb.count != 0) {
                        (true, false) => Ordering::Less,
                        (false, true) => Ordering::Greater,
                        _ => {
                            if eviction {
                                let pa = u128::from(scan - ca.scan) * u128::from(cb.count);
                                let pb = u128::from(scan - cb.scan) * u128::from(ca.count);
                                pb.cmp(&pa).then(ca.key.cmp(&cb.key))
                            } else {
                                ca.key.cmp(&cb.key).then(a.cmp(&b))
                            }
                        }
                    }
                };
                let mut buffers = map.buffers();
                buffers[0] = handle(&map.scratch.work);
                let params = map.params(n as u32, scan);
                let mut session = map.begin();
                map.sort(&mut session, &buffers, &params, eviction);
                session.shader_to_readback_barrier();
                session.submit();
                let mut sorted = vec![0u32; params.cap as usize];
                map.scratch.indices.download_into(&stream, &mut sorted);
                assert!(sorted[n..].iter().all(|&i| i == u32::MAX));
                let mut permutation = sorted[..n].to_vec();
                permutation.sort_unstable();
                assert_eq!(permutation, (0..n as u32).collect::<Vec<_>>());
                for pair in sorted[..n].windows(2) {
                    assert_ne!(
                        compare(pair[0], pair[1]),
                        Ordering::Greater,
                        "wrong order at n={n}, eviction={eviction}: {pair:?}"
                    );
                }
                if !eviction {
                    let mut expected: Vec<_> = (0..n as u32).collect();
                    expected.sort_by(|&a, &b| compare(a, b));
                    assert_eq!(&sorted[..n], expected);
                }
            }
        }
    }

    #[test]
    fn carried_fits_evictions_and_cached_misses_match_full_recompute() {
        let stream = Stream::new();
        let StreamBackend::Vulkan(context) = &stream.backend else {
            return;
        };
        if !context.precise_multiresolution {
            return;
        }
        let mut map = MultiResolutionMap::new(&stream, 3001, 0.5, 10.0);
        // A floor and two walls with deterministic noise, seen from a sensor
        // that moves, so cells are updated, inserted and evicted every scan.
        // Every other scan only covers a small patch, so most fits carry over.
        let mut seed = 0x9e3779b9u32;
        let mut noise = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32 - 0.5
        };
        let pose = Pose {
            position: Vector3::new(0., 0., 0.),
            orientation: Quaternion::default(),
        };
        for scan in 1..=12u32 {
            let points: Vec<_> = (0..6000)
                .map(|i| {
                    let extent = if scan % 2 == 0 { 3. } else { 20. };
                    let (u, v) = (extent * noise(), extent * noise());
                    let e = 0.02 * noise();
                    let (x, y, z) = match i % 3 {
                        0 => (u, v, -1.5 + e),
                        1 => (6. + e, u, 2. * v.abs() / 10. - 1.5),
                        _ => (u, -7. + e, 2. * v.abs() / 10. - 1.5),
                    };
                    pelorus_core::PointXYZIT {
                        x,
                        y,
                        z,
                        ..Default::default()
                    }
                })
                .collect();
            let cloud = VulkanBuffer::from_cloud(&stream, &points);
            let t = Vector3::new(1.5 * f64::from(scan), 0.2 * f64::from(scan), 0.);
            let q = Quaternion::new(
                0.,
                0.,
                (0.05 * f64::from(scan)).sin(),
                (0.05 * f64::from(scan)).cos(),
            );
            map.add(&stream, &cloud, &t, &q, scan);
            // flags still holds the dirty cells of the last level.
            let mut dirty = vec![0u32; map.levels[3].n as usize];
            map.scratch.flags.download_into(&stream, &mut dirty);
            let clean = dirty.iter().filter(|&&d| d == 0).count();
            assert!(
                scan % 2 == 1 || clean > 0,
                "nothing carried over in scan {scan}"
            );
            // Independent of the comparison kernels: a full refit of every level.
            for l in &map.levels {
                let n = l.n as usize;
                let full = VulkanBuffer::new(&stream, size_of::<Fit>(), n.max(1));
                let mut b = map.buffers();
                b[0] = handle(&l.cells);
                b[3] = handle(&l.table);
                b[8] = handle(&full);
                let mut p = map.params(l.n, scan);
                p.table_cap = l.table.n as u32;
                let mut session = map.begin();
                map.dispatch(&mut session, 7, &b, &p, l.n);
                session.shader_to_readback_barrier();
                session.submit();
                let (mut carried, mut expected) =
                    (vec![Fit::default(); n], vec![Fit::default(); n]);
                l.fits.download_into(&stream, &mut carried);
                full.download_into(&stream, &mut expected);
                assert_eq!(
                    bytemuck::cast_slice::<Fit, u8>(&carried),
                    bytemuck::cast_slice::<Fit, u8>(&expected),
                    "carried fits differ after scan {scan}"
                );
                let mut cells = vec![Cell::default(); n];
                l.cells.download_into(&stream, &mut cells);
                assert!(cells.windows(2).all(|w| w[0].key < w[1].key));
            }
            // Repeated queries within one map version read the cached fits. The
            // last one runs in a version without cached fits and fits inline.
            let query = VulkanBuffer::from_cloud(&stream, &points[..2000]);
            let mut results = Vec::new();
            for inline in [false, false, true] {
                if inline {
                    map.version += 1;
                    map.bypass_miss_cache = true;
                }
                let mut output = VulkanBuffer::new(&stream, size_of::<PointNormal>(), 2000);
                let mut download = vec![PointNormal::default(); 2000];
                let mut planes = Vec::new();
                let sensor = Pose {
                    position: t,
                    orientation: q,
                };
                map.planes(
                    &stream,
                    PlaneQuery {
                        cloud: &query,
                        pose: &sensor,
                        lidar_to_imu: &pose,
                    },
                    &mut planes,
                    &mut output,
                    &mut download,
                );
                map.bypass_miss_cache = false;
                download.truncate(planes.len());
                results.push(download);
            }
            for cached in &results[..2] {
                assert_eq!(
                    bytemuck::cast_slice::<PointNormal, u8>(cached),
                    bytemuck::cast_slice::<PointNormal, u8>(&results[2]),
                    "cached absent-cell fits differ after scan {scan}"
                );
            }
            assert!(!results[0].is_empty());
        }
        // The budget was reached, so cells were evicted.
        assert!(map.levels.iter().any(|l| l.n == l.budget));
    }
}
