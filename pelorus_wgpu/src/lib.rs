//! Experimental single- and multiresolution WebGPU correspondence search.
//!
//! CPU map maintenance and the f64 filter are shared with the CPU backend.
//! Upload a snapshot and cloud once per scan, then query each filter iteration.
//! The browser must await readback; it must never block its worker event loop.

use anyhow::{Context, Result, ensure};
use bytemuck::{Pod, Zeroable};
use nalgebra::{Isometry3, UnitQuaternion, Vector3};
use pelorus_core::PointXYZIT;
use pelorus_localmap::{IVoxHashMap, VoxelDistribution};
use ros_pointcloud2::points::PointXYZINormal;
use std::sync::{Arc, Mutex};

const SHADER: &str = include_str!("planes.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Cell {
    mean_count: [f32; 4],
    covariance: [[f32; 4]; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Level {
    resolution: f32,
    inverse_resolution: f32,
    table_offset: u32,
    table_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    pose_q: [f32; 4],
    pose_t: [f32; 4],
    offset_q: [f32; 4],
    offset_t: [f32; 4],
    count: u32,
    level_count: u32,
    max_thickness_squared: f32,
    // Always zero. The shader uses it to keep products from fusing into additions.
    zero: u32,
    levels: [Level; 4],
}

fn hash_key(mut h: u32) -> u32 {
    h = (h ^ (h >> 16)).wrapping_mul(0x85ebca6b);
    h = (h ^ (h >> 13)).wrapping_mul(0xc2b2ae35);
    h ^ (h >> 16)
}

struct Buffer {
    raw: wgpu::Buffer,
    usage: wgpu::BufferUsages,
}
impl Buffer {
    fn new(device: &wgpu::Device, usage: wgpu::BufferUsages) -> Self {
        Self {
            raw: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Pelorus scratch"),
                size: 16,
                usage,
                mapped_at_creation: false,
            }),
            usage,
        }
    }
    fn reserve(&mut self, device: &wgpu::Device, bytes: u64) -> Result<()> {
        let limits = device.limits();
        let maximum = if self.usage.contains(wgpu::BufferUsages::STORAGE) {
            limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size)
        } else {
            limits.max_buffer_size
        };
        ensure!(
            bytes <= maximum,
            "WebGPU buffer needs {bytes} bytes, limit is {maximum}"
        );
        if bytes > self.raw.size() {
            self.raw = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Pelorus scratch"),
                size: bytes.next_power_of_two().min(maximum),
                usage: self.usage,
                mapped_at_creation: false,
            });
        }
        Ok(())
    }
}

/// One odometer's GPU resources. No global device or map state.
/// Errors disable acceleration; the caller can continue against its CPU map.
pub struct Correspondence {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    points: Buffer,
    cells: Buffer,
    table: Buffer,
    output: Buffer,
    readback: Buffer,
    params_buffer: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    params: Params,
    error: Arc<Mutex<Option<String>>>,
    adapter_name: String,
    host_cells: Vec<Cell>,
    host_table: Vec<[u32; 4]>,
    host_points: Vec<[f32; 4]>,
}

impl Correspondence {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .context("no WebGPU compute adapter")?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Pelorus WebGPU"),
                ..Default::default()
            })
            .await
            .context("requesting WebGPU device")?;
        let error = Arc::new(Mutex::new(None));
        let errors = error.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            *errors.lock().unwrap() = Some(e.to_string());
        }));
        let errors = error.clone();
        device.set_device_lost_callback(move |reason, message| {
            *errors.lock().unwrap() = Some(format!("device lost ({reason:?}): {message}"));
        });
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pelorus planes"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pelorus planes"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = scope.pop().await {
            anyhow::bail!("WebGPU pipeline: {error}");
        }
        use wgpu::BufferUsages as U;
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pelorus pose"),
            size: size_of::<Params>() as u64,
            usage: U::UNIFORM | U::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            points: Buffer::new(&device, U::STORAGE | U::COPY_DST),
            cells: Buffer::new(&device, U::STORAGE | U::COPY_DST),
            table: Buffer::new(&device, U::STORAGE | U::COPY_DST),
            output: Buffer::new(&device, U::STORAGE | U::COPY_SRC),
            readback: Buffer::new(&device, U::MAP_READ | U::COPY_DST),
            device,
            queue,
            pipeline,
            params_buffer,
            bind_group: None,
            params: Params::zeroed(),
            error,
            adapter_name,
            host_cells: Vec::new(),
            host_table: Vec::new(),
            host_points: Vec::new(),
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }
    pub fn failure(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }
    pub fn disable(&mut self, error: impl ToString) {
        *self.error.lock().unwrap() = Some(error.to_string());
    }
    fn check(&self) -> Result<()> {
        if let Some(error) = self.failure() {
            anyhow::bail!("{error}");
        }
        Ok(())
    }

    fn append_level(
        &mut self,
        level: usize,
        resolution: f32,
        distributions: impl Iterator<Item = ([i32; 3], VoxelDistribution)>,
    ) -> Result<()> {
        ensure!(
            resolution.is_finite() && resolution > 0.0,
            "invalid map resolution"
        );
        let entries: Vec<_> = distributions.collect();
        let capacity = (entries.len().max(1) * 2).next_power_of_two();
        let offset = self.host_table.len();
        ensure!(
            offset + capacity <= u32::MAX as usize,
            "map exceeds WebGPU table limit"
        );
        self.host_table.resize(offset + capacity, [0; 4]);
        self.params.levels[level] = Level {
            resolution,
            inverse_resolution: 1.0 / resolution,
            table_offset: offset as u32,
            table_mask: (capacity - 1) as u32,
        };
        for (key, d) in entries {
            let index = self.host_cells.len() as u32 + 1;
            // Single-resolution maps intentionally identify cells by their stored
            // hash. Pyramid levels use full coordinates and resolve hash collisions.
            let hash = if self.params.level_count == 1 {
                key[0] as u32
            } else {
                (key[0] as u32)
                    .wrapping_mul(73856093)
                    .wrapping_add((key[1] as u32).wrapping_mul(471943))
                    .wrapping_add((key[2] as u32).wrapping_mul(83492791))
            };
            let mut slot = hash_key(hash) as usize & (capacity - 1);
            while self.host_table[offset + slot][3] != 0 {
                slot = (slot + 1) & (capacity - 1);
            }
            self.host_table[offset + slot] = [key[0] as u32, key[1] as u32, key[2] as u32, index];
            self.host_cells.push(Cell {
                mean_count: [d.mean.x, d.mean.y, d.mean.z, d.point_count as f32],
                covariance: std::array::from_fn(|col| {
                    [
                        d.covariance[(0, col)],
                        d.covariance[(1, col)],
                        d.covariance[(2, col)],
                        0.0,
                    ]
                }),
            });
        }
        Ok(())
    }

    /// Upload immutable map levels and the deskewed cloud once before correction.
    pub fn prepare(&mut self, map: &IVoxHashMap, points: &[PointXYZIT]) -> Result<()> {
        self.check()?;
        self.bind_group = None;
        ensure!(
            points.len().div_ceil(64)
                <= self.device.limits().max_compute_workgroups_per_dimension as usize,
            "cloud exceeds WebGPU dispatch limit"
        );
        self.host_cells.clear();
        self.host_table.clear();
        self.params.levels = [Level::zeroed(); 4];
        if let Some(snapshot) = map.multiresolution_snapshot() {
            self.params.level_count = 4;
            self.params.max_thickness_squared = snapshot.max_thickness.powi(2);
            for (index, level) in snapshot.levels.iter().enumerate() {
                self.append_level(index, level.resolution(), level.distributions())?;
            }
        } else {
            let (resolution, distributions) = map
                .single_resolution_snapshot()
                .context("WebGPU requires a multiresolution or single-resolution Nearby26 map")?;
            self.params.level_count = 1;
            self.params.max_thickness_squared = 0.0;
            self.append_level(
                0,
                resolution,
                distributions.map(|(key, d)| ([key, 0, 0], *d)),
            )?;
        }
        // Bind a complete Cell even for an empty map.
        if self.host_cells.is_empty() {
            self.host_cells.push(Cell::zeroed());
        }
        self.host_points.clear();
        self.host_points
            .extend(points.iter().map(|p| [p.x, p.y, p.z, 0.0]));
        self.params.count = points.len() as u32;
        let bytes = (points.len().max(1) * 16) as u64;
        self.points.reserve(&self.device, bytes)?;
        self.output.reserve(&self.device, bytes)?;
        self.readback.reserve(&self.device, bytes)?;
        self.cells.reserve(
            &self.device,
            (self.host_cells.len() * size_of::<Cell>()) as u64,
        )?;
        self.table
            .reserve(&self.device, (self.host_table.len() * 16) as u64)?;
        if !points.is_empty() {
            self.queue
                .write_buffer(&self.points.raw, 0, bytemuck::cast_slice(&self.host_points));
        }
        self.queue
            .write_buffer(&self.cells.raw, 0, bytemuck::cast_slice(&self.host_cells));
        self.queue
            .write_buffer(&self.table.raw, 0, bytemuck::cast_slice(&self.host_table));
        let buffers = [
            &self.points.raw,
            &self.cells.raw,
            &self.table.raw,
            &self.output.raw,
            &self.params_buffer,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pelorus scan"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        }));
        self.check()
    }

    /// Recompute correspondences for the current filter pose, preserving input order.
    pub async fn query(
        &mut self,
        pose: &Isometry3<f64>,
        offset: &Isometry3<f64>,
        out: &mut Vec<PointXYZINormal>,
    ) -> Result<()> {
        self.check()?;
        out.clear();
        if self.params.count == 0 {
            return Ok(());
        }
        let group = self
            .bind_group
            .as_ref()
            .context("prepare a scan before querying")?;
        fn quaternion(q: &UnitQuaternion<f64>) -> [f32; 4] {
            [q.i as f32, q.j as f32, q.k as f32, q.w as f32]
        }
        fn translation(v: &Vector3<f64>) -> [f32; 4] {
            [v.x as f32, v.y as f32, v.z as f32, 0.0]
        }
        self.params.pose_q = quaternion(&pose.rotation);
        self.params.pose_t = translation(&pose.translation.vector);
        self.params.offset_q = quaternion(&offset.rotation);
        self.params.offset_t = translation(&offset.translation.vector);
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(self.params.count.div_ceil(64), 1, 1);
        }
        let bytes = self.params.count as u64 * 16;
        encoder.copy_buffer_to_buffer(&self.output.raw, 0, &self.readback.raw, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let slice = self.readback.raw.slice(..bytes);
        let (sender, receiver) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        // Browser callbacks are driven by its event loop. Only native tests poll.
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .context("waiting for WebGPU")?;
        receiver
            .await
            .context("WebGPU readback cancelled")?
            .context("WebGPU readback")?;
        {
            let data = slice
                .get_mapped_range()
                .context("mapping the WebGPU readback")?;
            // read_unaligned avoids relying on the browser's mapped allocation alignment.
            for (p, bytes) in self.host_points.iter().zip(data.as_chunks::<16>().0) {
                let plane: [f32; 4] = bytemuck::pod_read_unaligned(bytes);
                if plane.iter().all(|v| v.is_finite()) && plane[..3].iter().any(|v| *v != 0.0) {
                    out.push(PointXYZINormal::new(
                        p[0], p[1], p[2], plane[3], plane[0], plane[1], plane[2],
                    ));
                }
            }
        }
        self.readback.raw.unmap();
        self.check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;
    use pelorus_localmap::IVoxNearbyType;
    use ros_pointcloud2::points::PointXYZ;

    // Some software adapters crash during concurrent device initialization.
    static ADAPTER_TEST: Mutex<()> = Mutex::new(());

    #[test]
    fn shader_validates_against_portable_webgpu() {
        let module = naga::front::wgsl::parse_str(SHADER)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(SHADER)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(size_of::<Cell>(), 64);
        assert_eq!(size_of::<Level>(), 16);
        assert_eq!(size_of::<Params>(), 144);
    }

    fn point(x: f32, y: f32, z: f32) -> PointXYZIT {
        PointXYZIT {
            x,
            y,
            z,
            ..Default::default()
        }
    }

    #[test]
    #[ignore = "requires a WebGPU/Vulkan/Metal/DX12 compute adapter"]
    fn gpu_matches_cpu_across_poses_and_buffer_reuse() {
        let _adapter_guard = ADAPTER_TEST.lock().unwrap();
        futures_lite::future::block_on(async {
            let mut gpu = Correspondence::new()
                .await
                .expect("compute adapter required");
            eprintln!("adapter: {}", gpu.adapter_name());
            let mut map = IVoxHashMap::new(0.5, 2000, IVoxNearbyType::Nearby26, 0.01);
            let scene: Vec<_> = (-35..=35)
                .flat_map(|x| {
                    (-25..=25).map(move |y| {
                        let x = x as f32 * 0.09;
                        let y = y as f32 * 0.09;
                        point(x, y, 1.0 + 0.15 * x - 0.07 * y)
                    })
                })
                .collect();
            map.add(&scene, &Isometry3::identity(), 1);
            let mut out = Vec::new();
            for count in [13, scene.len(), 71, 0, 200] {
                let mut query = scene[..count].to_vec();
                // Far-away points must not inherit an earlier invocation's result.
                query.extend([point(100.0, -100.0, 50.0), point(0.1, 0.2, 7.0)]);
                gpu.prepare(&map, &query).unwrap();
                for pose in [
                    Isometry3::identity(),
                    Isometry3::new(
                        Vector3::new(0.013, -0.017, 0.005),
                        Vector3::new(0.002, -0.003, 0.001),
                    ),
                ] {
                    let offset = Isometry3::new(
                        Vector3::new(0.01, 0.02, -0.01),
                        Vector3::new(-0.001, 0.002, 0.003),
                    );
                    gpu.query(&pose, &offset, &mut out).await.unwrap();
                    let pose32: Isometry3<f32> = pose.cast();
                    let offset32: Isometry3<f32> = offset.cast();
                    let expected: Vec<_> = query
                        .iter()
                        .filter_map(|p| {
                            let local = Point3::new(p.x, p.y, p.z);
                            let world = pose32 * (offset32 * local);
                            let plane = map.plane_fit_from_distributions(&PointXYZ::new(
                                world.x, world.y, world.z,
                            ))?;
                            let residual = plane[0] * world.x
                                + plane[1] * world.y
                                + plane[2] * world.z
                                + plane[3];
                            let score = 1.0
                                - 0.9 * residual.abs()
                                    / local.coords.norm_squared().max(1e-9).sqrt().sqrt();
                            (score > 0.9).then_some((p, plane))
                        })
                        .collect();
                    assert_eq!(
                        out.len(),
                        expected.len(),
                        "validity differs for {count} points"
                    );
                    for (actual, (point, plane)) in out.iter().zip(expected) {
                        assert_eq!([actual.x, actual.y, actual.z], [point.x, point.y, point.z]);
                        let sign = if actual.normal_x * plane[0]
                            + actual.normal_y * plane[1]
                            + actual.normal_z * plane[2]
                            < 0.0
                        {
                            -1.0
                        } else {
                            1.0
                        };
                        for (a, b) in [
                            actual.normal_x,
                            actual.normal_y,
                            actual.normal_z,
                            actual.intensity,
                        ]
                        .iter()
                        .zip(plane)
                        {
                            assert!(
                                (a * sign - b).abs() < 0.003,
                                "plane mismatch: GPU {actual:?}, CPU {plane:?}"
                            );
                        }
                    }
                }
            }
            gpu.prepare(&map, &[]).unwrap();
            gpu.query(&Isometry3::identity(), &Isometry3::identity(), &mut out)
                .await
                .unwrap();
            assert!(out.is_empty());
            let empty = IVoxHashMap::new(0.5, 100, IVoxNearbyType::Nearby26, 0.01);
            gpu.prepare(&empty, &scene[..100]).unwrap();
            gpu.query(&Isometry3::identity(), &Isometry3::identity(), &mut out)
                .await
                .unwrap();
            assert!(out.is_empty());
            let unsupported = IVoxHashMap::new(0.5, 100, IVoxNearbyType::Nearby6, 0.01);
            assert!(gpu.prepare(&unsupported, &scene).is_err());
            gpu.disable("test device loss");
            assert!(gpu.prepare(&map, &scene).is_err());
        });
    }
    async fn compare_multiresolution(
        gpu: &mut Correspondence,
        map: &IVoxHashMap,
        query: &[PointXYZIT],
        pose: &Isometry3<f64>,
    ) -> usize {
        gpu.prepare(map, query).unwrap();
        let mut out = Vec::new();
        let offset = Isometry3::identity();
        gpu.query(pose, &offset, &mut out).await.unwrap();
        let pose32: Isometry3<f32> = pose.cast();
        let expected: Vec<_> = query
            .iter()
            .filter_map(|p| {
                let local = Point3::new(p.x, p.y, p.z);
                let world = pose32 * local;
                let plane =
                    map.plane_fit_from_distributions(&PointXYZ::new(world.x, world.y, world.z))?;
                let residual =
                    plane[0] * world.x + plane[1] * world.y + plane[2] * world.z + plane[3];
                let score = 1.0
                    - 0.9 * residual.abs() / local.coords.norm_squared().max(1e-9).sqrt().sqrt();
                (score > 0.9).then_some((p, plane))
            })
            .collect();
        assert_eq!(
            out.len(),
            expected.len(),
            "multiresolution acceptance differs"
        );
        for (actual, (point, plane)) in out.iter().zip(&expected) {
            assert_eq!([actual.x, actual.y, actual.z], [point.x, point.y, point.z]);
            let sign = if actual.normal_x * plane[0]
                + actual.normal_y * plane[1]
                + actual.normal_z * plane[2]
                < 0.0
            {
                -1.0
            } else {
                1.0
            };
            for (a, b) in [
                actual.normal_x,
                actual.normal_y,
                actual.normal_z,
                actual.intensity,
            ]
            .iter()
            .zip(plane)
            {
                assert!(
                    (a * sign - b).abs() < 0.002,
                    "multiresolution plane mismatch: {actual:?} vs {plane:?}"
                );
            }
        }
        expected.len()
    }

    #[test]
    #[ignore = "requires a compute adapter"]
    fn multiresolution_matches_cpu_for_sparse_patches_updates_and_rejections() {
        let _adapter_guard = ADAPTER_TEST.lock().unwrap();
        futures_lite::future::block_on(async {
            let mut gpu = Correspondence::new().await.unwrap();
            // The widest spacing needs coarse levels: a fine neighborhood contains
            // fewer than six samples. Non-power-of-two resolution checks grid division.
            for (resolution, spacing) in [
                (0.5, 0.073),
                (0.5, 0.31),
                (0.5, 0.67),
                (0.7, 0.67),
                (0.5, 1.31),
            ] {
                let mut map = IVoxHashMap::new(resolution, 2048, IVoxNearbyType::Nearby26, 0.01);
                map.enable_multiresolution(2048, 0.01);
                let scene: Vec<_> = (-8..=8)
                    .flat_map(|x| {
                        (-8..=8).map(move |y| {
                            let x = x as f32 * spacing + 0.031;
                            let y = y as f32 * spacing - 0.047;
                            point(x, y, 1.123 + 0.12 * x - 0.07 * y)
                        })
                    })
                    .collect();
                for scan in 0..3 {
                    map.add(
                        &scene,
                        &Isometry3::translation(0.004 * scan as f64, 0.0, 0.002 * scan as f64),
                        scan,
                    );
                    let mut query: Vec<_> = scene.iter().step_by(3).copied().collect();
                    query.extend([point(100.0, -100.0, 1.0), point(0.0, 0.0, 7.0)]);
                    for pose in [
                        Isometry3::identity(),
                        Isometry3::new(
                            Vector3::new(0.013, -0.017, 0.005),
                            Vector3::new(0.002, -0.003, 0.001),
                        ),
                    ] {
                        assert!(compare_multiresolution(&mut gpu, &map, &query, &pose).await > 20);
                    }
                }
                // Reusing GPU buffers after clearing a populated pyramid must not
                // resurrect stale cells from any of the four levels.
                map.clear();
                assert_eq!(
                    compare_multiresolution(&mut gpu, &map, &scene, &Isometry3::identity()).await,
                    0
                );
            }
            let mut map = IVoxHashMap::new(0.5, 64, IVoxNearbyType::Nearby26, 0.01);
            map.enable_multiresolution(64, 0.01);
            let patch: Vec<_> = (-4..=4)
                .flat_map(|x| {
                    (-4..=4).map(move |y| point(x as f32 * 0.023, y as f32 * 0.029, 1.123))
                })
                .collect();
            map.add(&patch, &Isometry3::identity(), 0);
            // Nearby coarse cells must not extend the finite patch indefinitely.
            let query = [
                point(0.01, 0.02, 1.123),
                point(0.7, 0.0, 1.123),
                point(0.0, 0.0, 1.8),
            ];
            assert_eq!(
                compare_multiresolution(&mut gpu, &map, &query, &Isometry3::identity()).await,
                1
            );
            // Move many patches through a small map to exercise eviction and reuse.
            for scan in 1..8 {
                let translation = Isometry3::translation(scan as f64 * 3.13, 0.0, 0.0);
                map.add(&patch, &translation, scan);
                assert!(compare_multiresolution(&mut gpu, &map, &patch, &translation).await > 0);
            }
            for scene in [
                vec![point(0.0, 0.0, 1.123); 5], // insufficient support
                (-10..=10)
                    .map(|x| point(x as f32 * 0.03, 0.0, 1.123))
                    .collect(), // line
                (-2..=2)
                    .flat_map(|x| {
                        (-2..=2).flat_map(move |y| {
                            (-2..=2).map(move |z| {
                                point(x as f32 * 0.06, y as f32 * 0.06, 1.123 + z as f32 * 0.06)
                            })
                        })
                    })
                    .collect(), // thick volume
            ] {
                map.clear();
                map.add(&scene, &Isometry3::identity(), 0);
                assert_eq!(
                    compare_multiresolution(
                        &mut gpu,
                        &map,
                        &[point(0.01, 0.02, 1.123)],
                        &Isometry3::identity()
                    )
                    .await,
                    0
                );
            }

            // These different full grid keys have the same 32-bit spatial hash.
            // Their patches have perpendicular normals and must remain distinct.
            let keys = [[-567, 384, -160], [-146, 702, -277]];
            let mut collision_map = IVoxHashMap::new(2.0, 128, IVoxNearbyType::Nearby26, 0.01);
            collision_map.enable_multiresolution(128, 0.01);
            let mut scene = Vec::new();
            for (index, key) in keys.iter().enumerate() {
                for i in -2..=2 {
                    for j in -2..=2 {
                        let [x, y, z] = key.map(|v| v as f32 + 0.5);
                        scene.push(if index == 0 {
                            point(x + i as f32 * 0.07, y + j as f32 * 0.07, z)
                        } else {
                            point(x, y + i as f32 * 0.07, z + j as f32 * 0.07)
                        });
                    }
                }
            }
            collision_map.add(&scene, &Isometry3::identity(), 0);
            assert_eq!(
                compare_multiresolution(&mut gpu, &collision_map, &scene, &Isometry3::identity())
                    .await,
                scene.len()
            );

            // A fine patch remains valid even when larger neighborhoods include
            // thick clutter. Selecting the coarsest plane would lose this match.
            let mut fine_map = IVoxHashMap::new(0.5, 4096, IVoxNearbyType::Nearby26, 0.01);
            fine_map.enable_multiresolution(4096, 0.01);
            let mut scene = patch;
            for x in [-1.2, -0.7, 0.7, 1.2] {
                for y in -5..=5 {
                    for z in -5..=5 {
                        scene.push(point(x, y as f32 * 0.13, 1.123 + z as f32 * 0.13));
                    }
                }
            }
            fine_map.add(&scene, &Isometry3::identity(), 0);
            assert_eq!(
                compare_multiresolution(
                    &mut gpu,
                    &fine_map,
                    &[point(0.01, 0.02, 1.123)],
                    &Isometry3::identity()
                )
                .await,
                1
            );
        });
    }
}
