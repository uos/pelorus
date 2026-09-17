//! Device-resident distribution pyramid. Counts cross to the host between
//! passes. Observations, statistics, sorting, fitting and selection stay on GPU.
use crate::buffer::VulkanBuffer;
use crate::compute_session::{ComputeCommandResources, ComputeCommandSession, scan_work_items};
use crate::context::VulkanContext;
use crate::map::vk_buffer_handle as handle;
use crate::pipelines::{PushConstants, VulkanComputePipelines, build_pipelines};
use crate::planes::VulkanPlaneTimings;
use crate::stream::{Stream, StreamBackend};
use crate::types::{PointNormal, Quaternion, Vector3};
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
struct Level {
    cells: VulkanBuffer,
    fits: VulkanBuffer,
    n: u32,
    budget: u32,
}
struct Scratch {
    work: VulkanBuffer,
    reduced: VulkanBuffer,
    batches: VulkanBuffer,
    combined: VulkanBuffer,
    indices: VulkanBuffer,
    flags: VulkanBuffer,
    offsets: VulkanBuffer,
    sums: VulkanBuffer,
    planes: VulkanBuffer,
    counter: VulkanBuffer,
    poses: VulkanBuffer,
    cap: usize,
}
pub(crate) struct MultiResolutionMap {
    pub(crate) context: Arc<VulkanContext>,
    compute: Arc<VulkanComputePipelines>,
    resources: Mutex<ComputeCommandResources>,
    levels: [Level; 4],
    scratch: Scratch,
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
        let levels = std::array::from_fn(|i| {
            let budget = capacity / 4 + usize::from(i < capacity % 4);
            Level {
                cells: alloc(size_of::<Cell>(), budget),
                fits: alloc(size_of::<Fit>(), budget),
                n: 0,
                budget: budget as u32,
            }
        });
        Self {
            context: Arc::clone(context),
            resources: Mutex::new(ComputeCommandResources::new(context, &compute)),
            compute,
            levels,
            base,
            sensitivity,
            scratch: Scratch {
                work: alloc(size_of::<Cell>(), 1),
                reduced: alloc(size_of::<Cell>(), 1),
                batches: alloc(size_of::<Cell>(), 1),
                combined: alloc(size_of::<Cell>(), 1),
                indices: alloc(4, 1),
                flags: alloc(4, 1),
                offsets: alloc(4, 1),
                sums: alloc(4, 1),
                planes: alloc(size_of::<PointNormal>(), 1),
                counter: VulkanBuffer::new_host_visible(stream, 4, 1),
                poses: VulkanBuffer::new_host_visible(stream, 8, 7),
                cap: 1,
            },
        }
    }
    fn ensure(&mut self, stream: &Stream, n: usize) {
        let cap = n.max(1).next_power_of_two();
        if cap <= self.scratch.cap {
            return;
        }
        let alloc = |size| VulkanBuffer::new_device_local_required(stream, size, cap);
        self.scratch.work = alloc(size_of::<Cell>());
        self.scratch.reduced = alloc(size_of::<Cell>());
        self.scratch.batches = alloc(size_of::<Cell>());
        self.scratch.combined = alloc(size_of::<Cell>());
        self.scratch.indices = alloc(4);
        self.scratch.flags = alloc(4);
        self.scratch.offsets = alloc(4);
        self.scratch.sums = VulkanBuffer::new_device_local_required(stream, 4, cap.div_ceil(512));
        self.scratch.planes = alloc(size_of::<PointNormal>());
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
        let mut k = 2;
        while k <= p.cap {
            p.k = k;
            let mut j = k / 2;
            while j > 0 {
                p.j = j;
                self.dispatch(session, 1, b, &p, p.cap);
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
    fn reduce(&mut self, stream: &Stream, n: u32, scan: u32, mode: u32, output: vk::Buffer) -> u32 {
        if n == 0 {
            return 0;
        }
        self.scratch.counter.upload(stream, &[n]);
        let mut p = self.params(n, scan);
        let mut b = self.buffers();
        b[0] = handle(&self.scratch.work);
        b[1] = handle(&self.scratch.reduced);
        let mut session = self.begin();
        self.sort(&mut session, &b, &p, false);
        p.nbtype = mode;
        self.dispatch(&mut session, 2, &b, &p, n);
        self.prefix(&mut session, &b, &p);
        b[0] = handle(&self.scratch.reduced);
        b[1] = output;
        self.dispatch(&mut session, 3, &b, &p, n);
        session.dispatch(&self.compute, self.compute.plane_count, &b, &p, 1);
        session.shader_to_readback_barrier();
        session.submit();
        self.read_count(stream)
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
        let mut session = self.begin();
        self.dispatch(&mut session, 0, &b, &p, n);
        session.submit();
        let mut batch_n = self.reduce(stream, n, scan, 2, handle(&self.scratch.batches));
        if batch_n == 0 {
            return;
        }
        for level in 0..4 {
            if level > 0 {
                let mut b = self.buffers();
                b[0] = handle(&self.scratch.batches);
                b[1] = handle(&self.scratch.work);
                let p = self.params(batch_n, scan);
                let mut session = self.begin();
                self.dispatch(&mut session, 4, &b, &p, batch_n);
                session.submit();
                batch_n = self.reduce(stream, batch_n, scan, 0, handle(&self.scratch.batches));
            }
            let old_n = self.levels[level].n;
            let total = old_n + batch_n;
            let mut p = self.params(total, scan);
            p.valid_size = old_n;
            let mut b = self.buffers();
            b[0] = handle(&self.levels[level].cells);
            b[3] = handle(&self.scratch.batches);
            b[1] = handle(&self.scratch.work);
            let mut session = self.begin();
            self.dispatch(&mut session, 5, &b, &p, total);
            session.submit();
            let merged_n = self.reduce(stream, total, scan, 3, handle(&self.scratch.combined));
            let excess = merged_n.saturating_sub(self.levels[level].budget);
            let retained = merged_n - excess;
            let mut b = self.buffers();
            b[0] = handle(&self.scratch.combined);
            b[1] = handle(&self.scratch.work);
            let mut p = self.params(merged_n, scan);
            let mut session = self.begin();
            if excess > 0 {
                self.sort(&mut session, &b, &p, true);
                p.valid_size = excess;
                self.dispatch(&mut session, 6, &b, &p, retained);
                p = self.params(retained, scan);
                b[0] = handle(&self.scratch.work);
                b[1] = handle(&self.levels[level].cells);
                self.sort(&mut session, &b, &p, false);
                self.dispatch(&mut session, 6, &b, &p, retained);
            } else {
                p.valid_size = merged_n;
                b[1] = handle(&self.levels[level].cells);
                self.dispatch(&mut session, 5, &b, &p, merged_n);
            }
            b[0] = handle(&self.levels[level].cells);
            b[8] = handle(&self.levels[level].fits);
            p = self.params(retained, scan);
            self.dispatch(&mut session, 7, &b, &p, retained);
            session.shader_to_readback_barrier();
            session.submit();
            self.levels[level].n = retained;
        }
    }
    pub(crate) fn planes(
        &mut self,
        stream: &Stream,
        cloud: &VulkanBuffer,
        t: &Vector3,
        q: &Quaternion,
        ext_t: &Vector3,
        ext_q: &Quaternion,
        out: &mut Vec<PointXYZINormal>,
        out_gpu: &mut VulkanBuffer,
    ) -> VulkanPlaneTimings {
        let n = u32::try_from(cloud.n).expect("too many query points");
        if n == 0 {
            out.clear();
            return VulkanPlaneTimings::default();
        }
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
        let mut b = self.buffers();
        b[13] = handle(cloud);
        b[1] = handle(&self.scratch.planes);
        for i in 0..4 {
            b[4 + i] = handle(&self.levels[i].cells);
            b[[8, 9, 11, 12][i]] = handle(&self.levels[i].fits);
        }
        let start = Instant::now();
        let mut session = self.begin();
        self.dispatch(&mut session, 8, &b, &p, n);
        self.prefix(&mut session, &b, &p);
        let mut compact = b;
        compact[15] = handle(&self.scratch.planes);
        compact[17] = handle(out_gpu);
        session.dispatch(&self.compute, self.compute.compact_planes, &compact, &p, n);
        session.barrier();
        session.dispatch(&self.compute, self.compute.plane_count, &b, &p, 1);
        session.shader_to_readback_barrier();
        session.submit();
        let kernel_compact_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let count = self.read_count(stream) as usize;
        assert!(count <= cloud.n && count <= out_gpu.n);
        let mut downloaded = vec![PointNormal::default(); count];
        out_gpu.download_into(stream, &mut downloaded);
        out.clear();
        out.extend(downloaded.into_iter().map(PointXYZINormal::from));
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
