use nalgebra::Isometry3;
use pelorus_core::timing::RuntimeEvaluator;
use pelorus_core::{ImuParameters, PointXYZIT};
use pelorus_reg::{ActiveBackend, Registration, registration::RegistrationState};
use pelorus_sync::LocatedBufferView;
use serde::{Deserialize, Serialize};
use pelorus_localmap::{IVoxHashMap, IVoxNearbyType};
use std::fs;
use std::path::PathBuf;
#[cfg(any(feature = "metal", feature = "cuda", feature = "vulkan"))]
use std::sync::Mutex;

#[cfg(feature = "metal")]
static METAL_MAP_MUTEX: Mutex<()> = Mutex::new(());
#[cfg(feature = "cuda")]
static CUDA_MAP_MUTEX: Mutex<()> = Mutex::new(());
#[cfg(feature = "vulkan")]
static VULKAN_MAP_MUTEX: Mutex<()> = Mutex::new(());

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct DumpedState {
    pub translation: [f64; 3],
    pub rotation: [f64; 4],
    pub velocity: [f64; 3],
    pub bg: [f64; 3],
    pub ba: [f64; 3],
}

impl DumpedState {
    fn from_ekf(ekf: &pelorus_reg::ImuEkfPrediction) -> Self {
        let t = ekf.pose.translation.vector;
        let r = ekf.pose.rotation.coords;
        let v = ekf.linear_velocity;
        Self {
            translation: [t.x, t.y, t.z],
            rotation: [r.x, r.y, r.z, r.w],
            velocity: [v.x, v.y, v.z],
            bg: [
                ekf.angular_velocity.x,
                ekf.angular_velocity.y,
                ekf.angular_velocity.z,
            ],
            ba: [ekf.accel_bias.x, ekf.accel_bias.y, ekf.accel_bias.z],
        }
    }
}

fn create_simple_cloud() -> Vec<PointXYZIT> {
    let mut cloud = Vec::new();
    for x in 0..10 {
        for y in 0..10 {
            cloud.push(PointXYZIT {
                x: x as f32 * 0.1,
                y: y as f32 * 0.1,
                z: 1.0,
                intensity: 1.0,
                timestamp: 0.0,
            });
        }
    }
    cloud
}

fn create_complex_cloud() -> Vec<PointXYZIT> {
    let mut cloud = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            cloud.push(PointXYZIT {
                x: i as f32 * 0.1,
                y: j as f32 * 0.1,
                z: 1.0,
                intensity: 1.0,
                timestamp: 0.0,
            });

            cloud.push(PointXYZIT {
                x: i as f32 * 0.1,
                y: 1.0,
                z: j as f32 * 0.1,
                intensity: 1.0,
                timestamp: 0.0,
            });

            cloud.push(PointXYZIT {
                x: 1.0,
                y: i as f32 * 0.1,
                z: j as f32 * 0.1,
                intensity: 1.0,
                timestamp: 0.0,
            });
        }
    }
    cloud
}

#[test]
fn test_backends() {
    #[cfg(feature = "metal")]
    let _lock = METAL_MAP_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(feature = "cuda")]
    let _lock = CUDA_MAP_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(feature = "vulkan")]
    let _lock = VULKAN_MAP_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let params = ImuParameters::default();
    let map_resolution = 0.5;
    let map = IVoxHashMap::new(map_resolution as f32, 10000, IVoxNearbyType::Nearby26, 0.1);
    let lidar_to_imu = Isometry3::identity();
    let base_to_lidar = Isometry3::identity();
    let mut state = RegistrationState::new(
        map,
        &params,
        &lidar_to_imu,
        &base_to_lidar,
        9.81,
        0.001,
        5,
        None,
    );

    let simple_cloud = create_simple_cloud();
    let complex_cloud = create_complex_cloud();

    let imu_transform = Isometry3::identity();
    let mut eval = RuntimeEvaluator::new();

    #[cfg(feature = "metal")]
    let stream = pelorus_metal::Stream::new();
    #[cfg(feature = "metal")]
    pelorus_metal::init_map(
        &stream,
        10000,
        map_resolution as f32,
        0.1,
        (map_resolution / 4.0) as f32,
        pelorus_metal::NeighborType::Neighbor26,
    );

    #[cfg(feature = "cuda")]
    let stream = pelorus_cuda::Stream::new();
    #[cfg(feature = "cuda")]
    pelorus_cuda::init_map(
        &stream,
        10000,
        map_resolution as f32,
        0.1,
        (map_resolution / 4.0) as f32,
        pelorus_cuda::NeighborType::Neighbor26,
    );
    #[cfg(feature = "vulkan")]
    let stream = pelorus_vulkan::Stream::new();
    #[cfg(feature = "vulkan")]
    pelorus_vulkan::init_map(
        &stream,
        10000,
        map_resolution as f32,
        0.1,
        (map_resolution / 4.0) as f32,
        pelorus_vulkan::NeighborType::Neighbor26,
    );

    #[cfg(not(any(feature = "metal", feature = "cuda", feature = "vulkan")))]
    let (simple_view, complex_view, active_backend) = {
        (
            LocatedBufferView::from_vec_cpu(simple_cloud.clone()),
            LocatedBufferView::from_vec_cpu(complex_cloud.clone()),
            ActiveBackend::Cpu,
        )
    };

    #[cfg(feature = "metal")]
    let (simple_view, complex_view, active_backend) = {
        (
            LocatedBufferView::copy_from_slice_metal(&stream, &simple_cloud),
            LocatedBufferView::copy_from_slice_metal(&stream, &complex_cloud),
            ActiveBackend::Metal(stream),
        )
    };

    #[cfg(feature = "cuda")]
    let (simple_view, complex_view, active_backend) = {
        (
            LocatedBufferView::copy_from_slice_cuda(&stream, &simple_cloud),
            LocatedBufferView::copy_from_slice_cuda(&stream, &complex_cloud),
            ActiveBackend::Cuda(stream),
        )
    };
    #[cfg(feature = "vulkan")]
    let (simple_view, complex_view, active_backend) = {
        (
            LocatedBufferView::copy_from_slice_vulkan(&stream, &simple_cloud),
            LocatedBufferView::copy_from_slice_vulkan(&stream, &complex_cloud),
            ActiveBackend::Vulkan(stream),
        )
    };

    let _ = state
        .register_and_map(
            true,
            imu_transform,
            &simple_view,
            0.0,
            1,
            None,
            &mut eval,
            &active_backend,
        )
        .unwrap();

    let _ = state
        .register_and_map(
            false,
            imu_transform,
            &complex_view,
            0.1,
            2,
            None,
            &mut eval,
            &active_backend,
        )
        .unwrap();

    let output_state = DumpedState::from_ekf(&state.get_state());

    let mut gt_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    gt_path.push("tests");
    gt_path.push("backend_gt.json");

    if cfg!(not(any(
        feature = "metal",
        feature = "cuda",
        feature = "vulkan"
    ))) {
        println!("Saving CPU ground truth to {:?}", gt_path);
        let s = serde_json::to_string_pretty(&output_state).expect("Failed to serialize");
        fs::write(&gt_path, s).expect("Failed to write GT");
    } else {
        println!("Checking against CPU ground truth at {:?}", gt_path);
        let s = fs::read_to_string(&gt_path)
            .expect("Ground truth file missing. Run without GPU features first to generate it!");
        let expected_state: DumpedState = serde_json::from_str(&s).expect("Failed to parse GT");

        let tolerance_t = 1e-6; // Back to strict tolerance
        let tolerance_r = 1e-6;
        assert!(
            (output_state.translation[0] - expected_state.translation[0]).abs() < tolerance_t,
            "tx diff: {} vs {}",
            output_state.translation[0],
            expected_state.translation[0]
        );
        assert!(
            (output_state.translation[1] - expected_state.translation[1]).abs() < tolerance_t,
            "ty diff: {} vs {}",
            output_state.translation[1],
            expected_state.translation[1]
        );
        assert!(
            (output_state.translation[2] - expected_state.translation[2]).abs() < tolerance_t,
            "tz diff: {} vs {}",
            output_state.translation[2],
            expected_state.translation[2]
        );

        assert!(
            (output_state.rotation[0] - expected_state.rotation[0]).abs() < tolerance_r,
            "qx diff: {} vs {}",
            output_state.rotation[0],
            expected_state.rotation[0]
        );
        assert!(
            (output_state.rotation[1] - expected_state.rotation[1]).abs() < tolerance_r,
            "qy diff: {} vs {}",
            output_state.rotation[1],
            expected_state.rotation[1]
        );
        assert!(
            (output_state.rotation[2] - expected_state.rotation[2]).abs() < tolerance_r,
            "qz diff: {} vs {}",
            output_state.rotation[2],
            expected_state.rotation[2]
        );
        assert!(
            (output_state.rotation[3] - expected_state.rotation[3]).abs() < tolerance_r,
            "qw diff: {} vs {}",
            output_state.rotation[3],
            expected_state.rotation[3]
        );
    }

    #[cfg(feature = "cuda")]
    pelorus_cuda::cleanup_map();
    #[cfg(feature = "metal")]
    pelorus_metal::cleanup_map();
    #[cfg(feature = "vulkan")]
    pelorus_vulkan::cleanup_map();
}

#[cfg(feature = "cuda")]
#[test]
fn test_planes_direct_equivalency_cuda() {
    use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};
    let _lock = CUDA_MAP_MUTEX.lock().unwrap();
    let resolution = 0.5_f32;
    let map_cloud = create_simple_cloud();
    let query_cloud = create_complex_cloud();

    let mut cpu_map = pelorus_localmap::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &Isometry3::identity(), 1);

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    for pt in &query_cloud {
        let q = PointXYZ::new(pt.x, pt.y, pt.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * pt.x + plane[1] * pt.y + plane[2] * pt.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, plane[3], plane[0], plane[1], plane[2],
                ));
            }
        }
    }

    let stream = pelorus_cuda::Stream::new();
    pelorus_cuda::init_map(
        &stream,
        10000,
        resolution,
        0.1,
        resolution / 4.0,
        pelorus_cuda::NeighborType::Neighbor26,
    );

    let gpu_map_cloud: LocatedBufferView<PointXYZIT> =
        LocatedBufferView::copy_from_slice_cuda(&stream, &map_cloud);
    let LocatedBufferView::Cuda {
        buffer: ref map_buf,
    } = gpu_map_cloud
    else {
        unreachable!()
    };
    pelorus_cuda::add_to_map(
        &stream,
        map_buf,
        &pelorus_cuda::Vector3::new(0.0, 0.0, 0.0),
        &pelorus_cuda::Quaternion::default(),
        1,
    );

    let gpu_query_cloud: LocatedBufferView<PointXYZIT> =
        LocatedBufferView::copy_from_slice_cuda(&stream, &query_cloud);
    let LocatedBufferView::Cuda {
        buffer: ref query_buf,
    } = gpu_query_cloud
    else {
        unreachable!()
    };

    let mut cuda_planes_out: Vec<PointXYZINormal> =
        vec![PointXYZINormal::default(); query_cloud.len()];
    let mut cuda_planes_gpu = pelorus_cuda::CudaBuffer::new(
        &stream,
        std::mem::size_of::<PointXYZINormal>(),
        query_cloud.len(),
    );
    pelorus_cuda::planes_cuda(
        &stream,
        query_buf,
        &pelorus_cuda::Vector3::new(0.0, 0.0, 0.0),
        &pelorus_cuda::Quaternion::default(),
        &pelorus_cuda::Vector3::new(0.0, 0.0, 0.0),
        &pelorus_cuda::Quaternion::default(),
        &mut cuda_planes_out,
        &mut cuda_planes_gpu,
    );
    pelorus_cuda::cleanup_map();

    assert_eq!(
        cpu_planes.len(),
        cuda_planes_out.len(),
        "Different number of valid planes: CPU={} CUDA={}",
        cpu_planes.len(),
        cuda_planes_out.len()
    );

    let sort_key = |p: &PointXYZINormal| {
        (
            (p.x * 1000.0) as i32,
            (p.y * 1000.0) as i32,
            (p.z * 1000.0) as i32,
        )
    };
    let mut cpu_sorted = cpu_planes.clone();
    let mut cuda_sorted = cuda_planes_out.clone();
    cpu_sorted.sort_by_key(&sort_key);
    cuda_sorted.sort_by_key(&sort_key);

    let tol = 1e-6_f32;
    for (c, g) in cpu_sorted.iter().zip(cuda_sorted.iter()) {
        let max_diff = (c.normal_x - g.normal_x)
            .abs()
            .max((c.normal_y - g.normal_y).abs())
            .max((c.normal_z - g.normal_z).abs())
            .max((c.intensity - g.intensity).abs());
        assert!(
            max_diff < tol,
            "Plane mismatch at ({},{},{}): CPU=({:.4},{:.4},{:.4},{:.4}) CUDA=({:.4},{:.4},{:.4},{:.4}) diff={:.2e}",
            c.x,
            c.y,
            c.z,
            c.normal_x,
            c.normal_y,
            c.normal_z,
            c.intensity,
            g.normal_x,
            g.normal_y,
            g.normal_z,
            g.intensity,
            max_diff
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_planes_after_voxel_update_cuda() {
    use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};
    let _lock = CUDA_MAP_MUTEX.lock().unwrap();
    let resolution = 0.5_f32;
    let map_cloud = create_simple_cloud();

    let mut cpu_map = pelorus_localmap::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &Isometry3::identity(), 1);
    cpu_map.add(&map_cloud, &Isometry3::identity(), 2);

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    for pt in &map_cloud {
        let q = PointXYZ::new(pt.x, pt.y, pt.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * pt.x + plane[1] * pt.y + plane[2] * pt.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, plane[3], plane[0], plane[1], plane[2],
                ));
            }
        }
    }

    let stream = pelorus_cuda::Stream::new();
    pelorus_cuda::init_map(
        &stream,
        10000,
        resolution,
        0.1,
        resolution / 4.0,
        pelorus_cuda::NeighborType::Neighbor26,
    );

    let gpu_cloud: LocatedBufferView<PointXYZIT> =
        LocatedBufferView::copy_from_slice_cuda(&stream, &map_cloud);
    let LocatedBufferView::Cuda {
        buffer: ref cloud_buf,
    } = gpu_cloud
    else {
        unreachable!()
    };

    let zero_t = pelorus_cuda::Vector3::new(0.0, 0.0, 0.0);
    let identity_q = pelorus_cuda::Quaternion::default();
    pelorus_cuda::add_to_map(&stream, cloud_buf, &zero_t, &identity_q, 1);
    pelorus_cuda::add_to_map(&stream, cloud_buf, &zero_t, &identity_q, 2);

    let mut cuda_out: Vec<PointXYZINormal> = vec![PointXYZINormal::default(); map_cloud.len()];
    let mut cuda_gpu_out = pelorus_cuda::CudaBuffer::new(
        &stream,
        std::mem::size_of::<PointXYZINormal>(),
        map_cloud.len(),
    );
    pelorus_cuda::planes_cuda(
        &stream,
        cloud_buf,
        &zero_t,
        &identity_q,
        &zero_t,
        &identity_q,
        &mut cuda_out,
        &mut cuda_gpu_out,
    );
    pelorus_cuda::cleanup_map();

    assert_eq!(
        cpu_planes.len(),
        cuda_out.len(),
        "After voxel update: different plane count CPU={} CUDA={}",
        cpu_planes.len(),
        cuda_out.len()
    );

    let sort_key = |p: &PointXYZINormal| {
        (
            (p.x * 1000.0) as i32,
            (p.y * 1000.0) as i32,
            (p.z * 1000.0) as i32,
        )
    };
    let mut cpu_sorted = cpu_planes.clone();
    let mut cuda_sorted = cuda_out.clone();
    cpu_sorted.sort_by_key(&sort_key);
    cuda_sorted.sort_by_key(&sort_key);

    let tol = 1e-6_f32;
    for (c, g) in cpu_sorted.iter().zip(cuda_sorted.iter()) {
        let d = (c.normal_x - g.normal_x)
            .abs()
            .max((c.normal_y - g.normal_y).abs())
            .max((c.normal_z - g.normal_z).abs())
            .max((c.intensity - g.intensity).abs());
        assert!(
            d < tol,
            "After update — mismatch at ({:.2},{:.2},{:.2}): CPU=({:.4},{:.4},{:.4},{:.4}) CUDA=({:.4},{:.4},{:.4},{:.4}) diff={:.2e}",
            c.x,
            c.y,
            c.z,
            c.normal_x,
            c.normal_y,
            c.normal_z,
            c.intensity,
            g.normal_x,
            g.normal_y,
            g.normal_z,
            g.intensity,
            d
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_lru_eviction_equivalency_cuda() {
    let _lock = CUDA_MAP_MUTEX.lock().unwrap();
    let resolution = 0.5_f32;
    let capacity = 4;
    let point = |x: f32| PointXYZIT {
        x,
        y: 0.0,
        z: 0.0,
        intensity: 1.0,
        timestamp: 0.0,
    };

    // Fill the map with different point counts so the eviction ratios are
    // distinct after updating only the first voxel.
    let initial_cloud = vec![
        point(0.0),
        point(2.0),
        point(2.0),
        point(4.0),
        point(4.0),
        point(4.0),
        point(6.0),
        point(6.0),
        point(6.0),
        point(6.0),
    ];
    let update_cloud = vec![point(0.0)];
    let eviction_cloud = vec![point(8.0)];

    let mut cpu_map = pelorus_localmap::IVoxHashMap::new(resolution, capacity, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&initial_cloud, &Isometry3::identity(), 1);
    cpu_map.add(&update_cloud, &Isometry3::identity(), 2);
    cpu_map.add(&eviction_cloud, &Isometry3::identity(), 3);
    let mut cpu_map_points = cpu_map.get_entire_map();

    let stream = pelorus_cuda::Stream::new();
    pelorus_cuda::init_map(
        &stream,
        capacity,
        resolution,
        0.1,
        resolution / 4.0,
        pelorus_cuda::NeighborType::Neighbor26,
    );
    let zero_t = pelorus_cuda::Vector3::new(0.0, 0.0, 0.0);
    let identity_q = pelorus_cuda::Quaternion::default();

    for (scan_id, cloud) in [
        (1, &initial_cloud),
        (2, &update_cloud),
        (3, &eviction_cloud),
    ] {
        let gpu_cloud: LocatedBufferView<PointXYZIT> =
            LocatedBufferView::copy_from_slice_cuda(&stream, cloud);
        let LocatedBufferView::Cuda { buffer } = &gpu_cloud else {
            unreachable!()
        };
        pelorus_cuda::add_to_map(&stream, buffer, &zero_t, &identity_q, scan_id);
        stream.sync();
    }
    let mut cuda_map_points = pelorus_cuda::get_map();
    pelorus_cuda::cleanup_map();

    cpu_map_points.sort_by(|a, b| a.x.total_cmp(&b.x));
    cuda_map_points.sort_by(|a, b| a.x.total_cmp(&b.x));

    assert_eq!(cpu_map_points.len(), cuda_map_points.len());
    for (cpu, cuda) in cpu_map_points.iter().zip(cuda_map_points.iter()) {
        assert!(
            (cpu - cuda).norm() < 1e-6,
            "LRU map mismatch: CPU={cpu:?}, CUDA={cuda:?}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_capacity_crossing_does_not_reuse_empty_cuda_slot() {
    let _lock = CUDA_MAP_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let resolution = 0.5_f32;
    let capacity = 4;
    let point = |x: f32| PointXYZIT {
        x,
        y: 0.0,
        z: 0.0,
        intensity: 1.0,
        timestamp: 0.0,
    };

    // Start below capacity, then add enough new voxels in one scan to fill the
    // remaining slots and require an eviction. Empty slots must not also occur
    // at the front of the LRU victim order for that scan.
    let partial_cloud = vec![point(0.0), point(2.0), point(2.0)];
    let crossing_cloud = vec![point(4.0), point(6.0), point(8.0)];

    let mut cpu_map = pelorus_localmap::IVoxHashMap::new(resolution, capacity, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&partial_cloud, &Isometry3::identity(), 1);
    cpu_map.add(&crossing_cloud, &Isometry3::identity(), 2);
    let mut cpu_map_points = cpu_map.get_entire_map();

    let stream = pelorus_cuda::Stream::new();
    pelorus_cuda::init_map(
        &stream,
        capacity,
        resolution,
        0.1,
        resolution / 4.0,
        pelorus_cuda::NeighborType::Neighbor26,
    );
    let zero_t = pelorus_cuda::Vector3::new(0.0, 0.0, 0.0);
    let identity_q = pelorus_cuda::Quaternion::default();

    for (scan_id, cloud) in [(1, &partial_cloud), (2, &crossing_cloud)] {
        let gpu_cloud: LocatedBufferView<PointXYZIT> =
            LocatedBufferView::copy_from_slice_cuda(&stream, cloud);
        let LocatedBufferView::Cuda { buffer } = &gpu_cloud else {
            unreachable!()
        };
        pelorus_cuda::add_to_map(&stream, buffer, &zero_t, &identity_q, scan_id);
        stream.sync();
    }
    let mut cuda_map_points = pelorus_cuda::get_map();
    pelorus_cuda::cleanup_map();

    cpu_map_points.sort_by(|a, b| a.x.total_cmp(&b.x));
    cuda_map_points.sort_by(|a, b| a.x.total_cmp(&b.x));

    assert_eq!(cpu_map_points.len(), cuda_map_points.len());
    for (cpu, cuda) in cpu_map_points.iter().zip(cuda_map_points.iter()) {
        assert!(
            (cpu - cuda).norm() < 1e-6,
            "capacity-crossing map mismatch: CPU={cpu:?}, CUDA={cuda:?}"
        );
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn test_planes_direct_equivalency_vulkan() {
    use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};
    let _lock = VULKAN_MAP_MUTEX.lock().unwrap();
    let resolution = 0.5_f32;
    let map_cloud = create_simple_cloud();
    let query_cloud = create_complex_cloud();

    let mut cpu_map = pelorus_localmap::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &Isometry3::identity(), 1);

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    for pt in &query_cloud {
        let q = PointXYZ::new(pt.x, pt.y, pt.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * pt.x + plane[1] * pt.y + plane[2] * pt.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, plane[3], plane[0], plane[1], plane[2],
                ));
            }
        }
    }

    let stream = pelorus_vulkan::Stream::new();
    pelorus_vulkan::init_map(
        &stream,
        10000,
        resolution,
        0.1,
        resolution / 4.0,
        pelorus_vulkan::NeighborType::Neighbor26,
    );

    let gpu_map_cloud: LocatedBufferView<PointXYZIT> =
        LocatedBufferView::copy_from_slice_vulkan(&stream, &map_cloud);
    let LocatedBufferView::Vulkan {
        buffer: ref map_buf,
    } = gpu_map_cloud
    else {
        unreachable!()
    };
    pelorus_vulkan::add_to_map(
        &stream,
        map_buf,
        &pelorus_vulkan::Vector3::new(0.0, 0.0, 0.0),
        &pelorus_vulkan::Quaternion::default(),
        1,
    );

    let gpu_query_cloud: LocatedBufferView<PointXYZIT> =
        LocatedBufferView::copy_from_slice_vulkan(&stream, &query_cloud);
    let LocatedBufferView::Vulkan {
        buffer: ref query_buf,
    } = gpu_query_cloud
    else {
        unreachable!()
    };

    let mut vulkan_planes_out: Vec<PointXYZINormal> =
        vec![PointXYZINormal::default(); query_cloud.len()];
    let mut vulkan_planes_gpu = pelorus_vulkan::VulkanBuffer::new(
        &stream,
        std::mem::size_of::<pelorus_vulkan::PointNormal>(),
        query_cloud.len(),
    );
    let mut vulkan_download = vec![pelorus_vulkan::PointNormal::default(); query_cloud.len()];
    let mut vulkan_counter = [0u32; 3];
    pelorus_vulkan::planes_vulkan(
        &stream,
        query_buf,
        &pelorus_vulkan::Vector3::new(0.0, 0.0, 0.0),
        &pelorus_vulkan::Quaternion::default(),
        &pelorus_vulkan::Vector3::new(0.0, 0.0, 0.0),
        &pelorus_vulkan::Quaternion::default(),
        &mut vulkan_planes_out,
        &mut vulkan_planes_gpu,
        &mut vulkan_download,
        &mut vulkan_counter,
    );
    pelorus_vulkan::cleanup_map();

    assert_eq!(
        cpu_planes.len(),
        vulkan_planes_out.len(),
        "Different number of valid planes: CPU={} Vulkan={}",
        cpu_planes.len(),
        vulkan_planes_out.len()
    );

    let sort_key = |p: &PointXYZINormal| {
        (
            (p.x * 1000.0) as i32,
            (p.y * 1000.0) as i32,
            (p.z * 1000.0) as i32,
        )
    };
    let mut cpu_sorted = cpu_planes.clone();
    let mut vulkan_sorted = vulkan_planes_out.clone();
    cpu_sorted.sort_by_key(&sort_key);
    vulkan_sorted.sort_by_key(&sort_key);

    let tol = 1e-6_f32;
    for (c, g) in cpu_sorted.iter().zip(vulkan_sorted.iter()) {
        let max_diff = (c.normal_x - g.normal_x)
            .abs()
            .max((c.normal_y - g.normal_y).abs())
            .max((c.normal_z - g.normal_z).abs())
            .max((c.intensity - g.intensity).abs());
        assert!(
            max_diff < tol,
            "Plane mismatch at ({},{},{}): CPU=({:.4},{:.4},{:.4},{:.4}) Vulkan=({:.4},{:.4},{:.4},{:.4}) diff={:.2e}",
            c.x,
            c.y,
            c.z,
            c.normal_x,
            c.normal_y,
            c.normal_z,
            c.intensity,
            g.normal_x,
            g.normal_y,
            g.normal_z,
            g.intensity,
            max_diff
        );
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn test_planes_after_voxel_update_vulkan() {
    use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};
    let _lock = VULKAN_MAP_MUTEX.lock().unwrap();
    let resolution = 0.5_f32;
    let map_cloud = create_simple_cloud();

    let mut cpu_map = pelorus_localmap::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &Isometry3::identity(), 1);
    cpu_map.add(&map_cloud, &Isometry3::identity(), 2);

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    for pt in &map_cloud {
        let q = PointXYZ::new(pt.x, pt.y, pt.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * pt.x + plane[1] * pt.y + plane[2] * pt.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, plane[3], plane[0], plane[1], plane[2],
                ));
            }
        }
    }

    let stream = pelorus_vulkan::Stream::new();
    pelorus_vulkan::init_map(
        &stream,
        10000,
        resolution,
        0.1,
        resolution / 4.0,
        pelorus_vulkan::NeighborType::Neighbor26,
    );

    let gpu_cloud: LocatedBufferView<PointXYZIT> =
        LocatedBufferView::copy_from_slice_vulkan(&stream, &map_cloud);
    let LocatedBufferView::Vulkan {
        buffer: ref cloud_buf,
    } = gpu_cloud
    else {
        unreachable!()
    };

    let zero_t = pelorus_vulkan::Vector3::new(0.0, 0.0, 0.0);
    let identity_q = pelorus_vulkan::Quaternion::default();
    pelorus_vulkan::add_to_map(&stream, cloud_buf, &zero_t, &identity_q, 1);
    pelorus_vulkan::add_to_map(&stream, cloud_buf, &zero_t, &identity_q, 2);

    let mut vulkan_out: Vec<PointXYZINormal> = vec![PointXYZINormal::default(); map_cloud.len()];
    let mut vulkan_gpu_out = pelorus_vulkan::VulkanBuffer::new(
        &stream,
        std::mem::size_of::<pelorus_vulkan::PointNormal>(),
        map_cloud.len(),
    );
    let mut vulkan_download = vec![pelorus_vulkan::PointNormal::default(); map_cloud.len()];
    let mut vulkan_counter = [0u32; 3];
    pelorus_vulkan::planes_vulkan(
        &stream,
        cloud_buf,
        &zero_t,
        &identity_q,
        &zero_t,
        &identity_q,
        &mut vulkan_out,
        &mut vulkan_gpu_out,
        &mut vulkan_download,
        &mut vulkan_counter,
    );
    pelorus_vulkan::cleanup_map();

    assert_eq!(
        cpu_planes.len(),
        vulkan_out.len(),
        "After voxel update: different plane count CPU={} Vulkan={}",
        cpu_planes.len(),
        vulkan_out.len()
    );

    let sort_key = |p: &PointXYZINormal| {
        (
            (p.x * 1000.0) as i32,
            (p.y * 1000.0) as i32,
            (p.z * 1000.0) as i32,
        )
    };
    let mut cpu_sorted = cpu_planes.clone();
    let mut vulkan_sorted = vulkan_out.clone();
    cpu_sorted.sort_by_key(&sort_key);
    vulkan_sorted.sort_by_key(&sort_key);

    let tol = 1e-6_f32;
    for (c, g) in cpu_sorted.iter().zip(vulkan_sorted.iter()) {
        let d = (c.normal_x - g.normal_x)
            .abs()
            .max((c.normal_y - g.normal_y).abs())
            .max((c.normal_z - g.normal_z).abs())
            .max((c.intensity - g.intensity).abs());
        assert!(
            d < tol,
            "After update — mismatch at ({:.2},{:.2},{:.2}): CPU=({:.4},{:.4},{:.4},{:.4}) Vulkan=({:.4},{:.4},{:.4},{:.4}) diff={:.2e}",
            c.x,
            c.y,
            c.z,
            c.normal_x,
            c.normal_y,
            c.normal_z,
            c.intensity,
            g.normal_x,
            g.normal_y,
            g.normal_z,
            g.intensity,
            d
        );
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn test_planes_transformed_extrinsic_equivalency_vulkan() {
    use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};
    let _lock = VULKAN_MAP_MUTEX.lock().unwrap();
    let resolution = 0.5_f32;
    let map_cloud = create_simple_cloud();
    let query_cloud = create_complex_cloud();

    let map_pose = Isometry3::from_parts(
        nalgebra::Translation3::new(1.25, -0.75, 0.5),
        nalgebra::UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3),
    );
    let query_pose = Isometry3::from_parts(
        nalgebra::Translation3::new(1.4, -0.6, 0.7),
        nalgebra::UnitQuaternion::from_euler_angles(0.08, -0.18, 0.32),
    );
    let lidar_to_imu_q =
        nalgebra::UnitQuaternion::from_euler_angles(std::f64::consts::PI, 0.0, 0.0);
    let lidar_to_imu_t = nalgebra::Vector3::new(0.1, -0.05, 0.02);

    let mut cpu_map = pelorus_localmap::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &map_pose, 1);

    let offset_q_f32 = lidar_to_imu_q.cast::<f32>();
    let offset_t_f32 = lidar_to_imu_t.cast::<f32>();
    let pose_q_f32 = query_pose.rotation.cast::<f32>();
    let pose_t_f32 = query_pose.translation.vector.cast::<f32>();

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    for pt in &query_cloud {
        let p_l = nalgebra::Point3::new(pt.x, pt.y, pt.z);
        let p_imu = offset_q_f32 * p_l + offset_t_f32;
        let p_world = pose_q_f32 * p_imu + pose_t_f32;
        let q = PointXYZ::new(p_world.x, p_world.y, p_world.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * p_world.x + plane[1] * p_world.y + plane[2] * p_world.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, plane[3], plane[0], plane[1], plane[2],
                ));
            }
        }
    }

    let stream = pelorus_vulkan::Stream::new();
    pelorus_vulkan::init_map(
        &stream,
        10000,
        resolution,
        0.1,
        resolution / 4.0,
        pelorus_vulkan::NeighborType::Neighbor26,
    );

    let gpu_map_cloud: LocatedBufferView<PointXYZIT> =
        LocatedBufferView::copy_from_slice_vulkan(&stream, &map_cloud);
    let LocatedBufferView::Vulkan {
        buffer: ref map_buf,
    } = gpu_map_cloud
    else {
        unreachable!()
    };
    let map_pose_t = pelorus_vulkan::Vector3::from(map_pose.translation.vector);
    let map_pose_q = pelorus_vulkan::Quaternion::from(map_pose.rotation);
    pelorus_vulkan::add_to_map(&stream, map_buf, &map_pose_t, &map_pose_q, 1);

    let gpu_query_cloud: LocatedBufferView<PointXYZIT> =
        LocatedBufferView::copy_from_slice_vulkan(&stream, &query_cloud);
    let LocatedBufferView::Vulkan {
        buffer: ref query_buf,
    } = gpu_query_cloud
    else {
        unreachable!()
    };

    let query_pose_t = pelorus_vulkan::Vector3::from(query_pose.translation.vector);
    let query_pose_q = pelorus_vulkan::Quaternion::from(query_pose.rotation);
    let offset_t = pelorus_vulkan::Vector3::from(lidar_to_imu_t);
    let offset_q = pelorus_vulkan::Quaternion::from(lidar_to_imu_q);
    let mut vulkan_out: Vec<PointXYZINormal> = vec![PointXYZINormal::default(); query_cloud.len()];
    let mut vulkan_gpu_out = pelorus_vulkan::VulkanBuffer::new(
        &stream,
        std::mem::size_of::<pelorus_vulkan::PointNormal>(),
        query_cloud.len(),
    );
    let mut vulkan_download = vec![pelorus_vulkan::PointNormal::default(); query_cloud.len()];
    let mut vulkan_counter = [0u32; 3];
    pelorus_vulkan::planes_vulkan(
        &stream,
        query_buf,
        &query_pose_t,
        &query_pose_q,
        &offset_t,
        &offset_q,
        &mut vulkan_out,
        &mut vulkan_gpu_out,
        &mut vulkan_download,
        &mut vulkan_counter,
    );
    pelorus_vulkan::cleanup_map();

    assert_eq!(
        cpu_planes.len(),
        vulkan_out.len(),
        "Transformed extrinsic: different plane count CPU={} Vulkan={}",
        cpu_planes.len(),
        vulkan_out.len()
    );

    let sort_key = |p: &PointXYZINormal| {
        (
            (p.x * 1000.0) as i32,
            (p.y * 1000.0) as i32,
            (p.z * 1000.0) as i32,
        )
    };
    let mut cpu_sorted = cpu_planes.clone();
    let mut vulkan_sorted = vulkan_out.clone();
    cpu_sorted.sort_by_key(&sort_key);
    vulkan_sorted.sort_by_key(&sort_key);

    let tol = 1e-5_f32;
    for (c, g) in cpu_sorted.iter().zip(vulkan_sorted.iter()) {
        let d = (c.normal_x - g.normal_x)
            .abs()
            .max((c.normal_y - g.normal_y).abs())
            .max((c.normal_z - g.normal_z).abs())
            .max((c.intensity - g.intensity).abs());
        assert!(
            d < tol,
            "Transformed extrinsic mismatch at ({:.2},{:.2},{:.2}): CPU=({:.4},{:.4},{:.4},{:.4}) Vulkan=({:.4},{:.4},{:.4},{:.4}) diff={:.2e}",
            c.x,
            c.y,
            c.z,
            c.normal_x,
            c.normal_y,
            c.normal_z,
            c.intensity,
            g.normal_x,
            g.normal_y,
            g.normal_z,
            g.intensity,
            d
        );
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn test_planes_after_moved_voxel_update_vulkan() {
    use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};
    let _lock = VULKAN_MAP_MUTEX.lock().unwrap();
    let resolution = 0.5_f32;
    let map_cloud = create_simple_cloud();

    let first_pose = Isometry3::from_parts(
        nalgebra::Translation3::new(1.25, -0.75, 0.5),
        nalgebra::UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3),
    );
    let second_pose = Isometry3::from_parts(
        nalgebra::Translation3::new(1.28, -0.72, 0.53),
        nalgebra::UnitQuaternion::from_euler_angles(0.11, -0.19, 0.31),
    );

    let mut cpu_map = pelorus_localmap::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &first_pose, 1);
    cpu_map.add(&map_cloud, &second_pose, 2);

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    let second_pose_f32 = second_pose.cast::<f32>();
    for pt in &map_cloud {
        let p_l = nalgebra::Point3::new(pt.x, pt.y, pt.z);
        let p_world = second_pose_f32.transform_point(&p_l);
        let q = PointXYZ::new(p_world.x, p_world.y, p_world.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * p_world.x + plane[1] * p_world.y + plane[2] * p_world.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, plane[3], plane[0], plane[1], plane[2],
                ));
            }
        }
    }

    let stream = pelorus_vulkan::Stream::new();
    pelorus_vulkan::init_map(
        &stream,
        10000,
        resolution,
        0.1,
        resolution / 4.0,
        pelorus_vulkan::NeighborType::Neighbor26,
    );

    let gpu_cloud: LocatedBufferView<PointXYZIT> =
        LocatedBufferView::copy_from_slice_vulkan(&stream, &map_cloud);
    let LocatedBufferView::Vulkan {
        buffer: ref cloud_buf,
    } = gpu_cloud
    else {
        unreachable!()
    };

    let first_t = pelorus_vulkan::Vector3::from(first_pose.translation.vector);
    let first_q = pelorus_vulkan::Quaternion::from(first_pose.rotation);
    let second_t = pelorus_vulkan::Vector3::from(second_pose.translation.vector);
    let second_q = pelorus_vulkan::Quaternion::from(second_pose.rotation);
    pelorus_vulkan::add_to_map(&stream, cloud_buf, &first_t, &first_q, 1);
    pelorus_vulkan::add_to_map(&stream, cloud_buf, &second_t, &second_q, 2);

    let zero_t = pelorus_vulkan::Vector3::new(0.0, 0.0, 0.0);
    let identity_q = pelorus_vulkan::Quaternion::default();
    let mut vulkan_out: Vec<PointXYZINormal> = vec![PointXYZINormal::default(); map_cloud.len()];
    let mut vulkan_gpu_out = pelorus_vulkan::VulkanBuffer::new(
        &stream,
        std::mem::size_of::<pelorus_vulkan::PointNormal>(),
        map_cloud.len(),
    );
    let mut vulkan_download = vec![pelorus_vulkan::PointNormal::default(); map_cloud.len()];
    let mut vulkan_counter = [0u32; 3];
    pelorus_vulkan::planes_vulkan(
        &stream,
        cloud_buf,
        &second_t,
        &second_q,
        &zero_t,
        &identity_q,
        &mut vulkan_out,
        &mut vulkan_gpu_out,
        &mut vulkan_download,
        &mut vulkan_counter,
    );
    pelorus_vulkan::cleanup_map();

    assert_eq!(
        cpu_planes.len(),
        vulkan_out.len(),
        "Moved update: different plane count CPU={} Vulkan={}",
        cpu_planes.len(),
        vulkan_out.len()
    );

    let sort_key = |p: &PointXYZINormal| {
        (
            (p.x * 1000.0) as i32,
            (p.y * 1000.0) as i32,
            (p.z * 1000.0) as i32,
        )
    };
    let mut cpu_sorted = cpu_planes.clone();
    let mut vulkan_sorted = vulkan_out.clone();
    cpu_sorted.sort_by_key(&sort_key);
    vulkan_sorted.sort_by_key(&sort_key);

    let tol = 1e-5_f32;
    for (c, g) in cpu_sorted.iter().zip(vulkan_sorted.iter()) {
        let d = (c.normal_x - g.normal_x)
            .abs()
            .max((c.normal_y - g.normal_y).abs())
            .max((c.normal_z - g.normal_z).abs())
            .max((c.intensity - g.intensity).abs());
        assert!(
            d < tol,
            "Moved update mismatch at ({:.2},{:.2},{:.2}): CPU=({:.4},{:.4},{:.4},{:.4}) Vulkan=({:.4},{:.4},{:.4},{:.4}) diff={:.2e}",
            c.x,
            c.y,
            c.z,
            c.normal_x,
            c.normal_y,
            c.normal_z,
            c.intensity,
            g.normal_x,
            g.normal_y,
            g.normal_z,
            g.intensity,
            d
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn test_planes_direct_equivalency() {
    use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};
    let _lock = METAL_MAP_MUTEX.lock().unwrap();
    let resolution = 0.5_f32;
    let map_cloud = create_simple_cloud();
    let query_cloud = create_complex_cloud();

    // ── CPU path ────────────────────────────────────────────────────────────────
    let mut cpu_map = pelorus_localmap::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &Isometry3::identity(), 1);

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    for pt in &query_cloud {
        let q = PointXYZ::new(pt.x, pt.y, pt.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * pt.x + plane[1] * pt.y + plane[2] * pt.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, plane[3], plane[0], plane[1], plane[2],
                ));
            }
        }
    }

    // ── Metal path ──────────────────────────────────────────────────────────────
    let stream = pelorus_metal::Stream::new();
    pelorus_metal::init_map(
        &stream,
        10000,
        resolution,
        0.1,
        resolution / 4.0,
        pelorus_metal::NeighborType::Neighbor26,
    );

    let gpu_map_cloud: LocatedBufferView<PointXYZIT> =
        LocatedBufferView::copy_from_slice_metal(&stream, &map_cloud);
    let LocatedBufferView::Metal {
        buffer: ref map_buf,
    } = gpu_map_cloud
    else {
        unreachable!()
    };
    pelorus_metal::add_to_map(
        &stream,
        map_buf,
        &pelorus_metal::Vector3::new(0.0, 0.0, 0.0),
        &pelorus_metal::Quaternion::default(),
        1,
    );

    let gpu_query_cloud: LocatedBufferView<PointXYZIT> =
        LocatedBufferView::copy_from_slice_metal(&stream, &query_cloud);
    let LocatedBufferView::Metal {
        buffer: ref query_buf,
    } = gpu_query_cloud
    else {
        unreachable!()
    };

    let mut metal_planes_out: Vec<PointXYZINormal> =
        vec![PointXYZINormal::default(); query_cloud.len()];
    let mut metal_planes_gpu = pelorus_metal::MetalBuffer::new(
        &stream,
        std::mem::size_of::<pelorus_metal::PointNormal>(),
        query_cloud.len(),
    );

    pelorus_metal::planes_metal(
        &stream,
        query_buf,
        &pelorus_metal::Vector3::new(0.0, 0.0, 0.0),
        &pelorus_metal::Quaternion::default(),
        &pelorus_metal::Vector3::new(0.0, 0.0, 0.0),
        &pelorus_metal::Quaternion::default(),
        &mut metal_planes_out,
        &mut metal_planes_gpu,
    );

    pelorus_metal::cleanup_map();

    // ── Compare ─────────────────────────────────────────────────────────────────
    println!(
        "Plane counts — CPU: {}, Metal: {}",
        cpu_planes.len(),
        metal_planes_out.len()
    );

    assert_eq!(
        cpu_planes.len(),
        metal_planes_out.len(),
        "Different number of valid planes: CPU={} Metal={}",
        cpu_planes.len(),
        metal_planes_out.len()
    );

    // Sort both by (x, y, z) to make comparison order-independent
    let sort_key = |p: &PointXYZINormal| {
        (
            (p.x * 1000.0) as i32,
            (p.y * 1000.0) as i32,
            (p.z * 1000.0) as i32,
        )
    };
    let mut cpu_sorted = cpu_planes.clone();
    let mut metal_sorted = metal_planes_out.clone();
    cpu_sorted.sort_by_key(&sort_key);
    metal_sorted.sort_by_key(&sort_key);

    let tol = 1e-6_f32;
    let mut max_normal_diff: f32 = 0.0;
    for (c, m) in cpu_sorted.iter().zip(metal_sorted.iter()) {
        let diff_nx = (c.normal_x - m.normal_x).abs();
        let diff_ny = (c.normal_y - m.normal_y).abs();
        let diff_nz = (c.normal_z - m.normal_z).abs();
        let diff_i = (c.intensity - m.intensity).abs();
        let max_diff = diff_nx.max(diff_ny).max(diff_nz).max(diff_i);
        if max_diff > max_normal_diff {
            max_normal_diff = max_diff;
        }
        assert!(
            max_diff < tol,
            "Plane mismatch at ({},{},{}): CPU normal=({:.4},{:.4},{:.4},{:.4}) Metal=({:.4},{:.4},{:.4},{:.4}) diff={:.2e}",
            c.x,
            c.y,
            c.z,
            c.normal_x,
            c.normal_y,
            c.normal_z,
            c.intensity,
            m.normal_x,
            m.normal_y,
            m.normal_z,
            m.intensity,
            max_diff
        );
    }
    println!("Max normal component diff: {:.2e}", max_normal_diff);
}

/// Tests the voxel UPDATE path: add the same cloud twice to both backends,
/// then compare plane outputs. The second add triggers the Mahalanobis-gated
/// blend in CPU `VoxelDistribution::update()` vs Metal `build_voxels` update branch.
#[cfg(feature = "metal")]
#[test]
fn test_planes_after_voxel_update() {
    use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};
    let _lock = METAL_MAP_MUTEX.lock().unwrap();
    let resolution = 0.5_f32;
    let map_cloud = create_simple_cloud();

    // ── CPU path: add cloud twice ────────────────────────────────────────────────
    let mut cpu_map = pelorus_localmap::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &Isometry3::identity(), 1);
    cpu_map.add(&map_cloud, &Isometry3::identity(), 2); // second add triggers voxel UPDATE

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    for pt in &map_cloud {
        let q = PointXYZ::new(pt.x, pt.y, pt.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * pt.x + plane[1] * pt.y + plane[2] * pt.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, plane[3], plane[0], plane[1], plane[2],
                ));
            }
        }
    }

    // ── Metal path: add cloud twice ──────────────────────────────────────────────
    let stream = pelorus_metal::Stream::new();
    pelorus_metal::init_map(
        &stream,
        10000,
        resolution,
        0.1,
        resolution / 4.0,
        pelorus_metal::NeighborType::Neighbor26,
    );

    let gpu_cloud: LocatedBufferView<PointXYZIT> =
        LocatedBufferView::copy_from_slice_metal(&stream, &map_cloud);
    let LocatedBufferView::Metal {
        buffer: ref cloud_buf,
    } = gpu_cloud
    else {
        unreachable!()
    };

    let zero_t = pelorus_metal::Vector3::new(0.0, 0.0, 0.0);
    let identity_q = pelorus_metal::Quaternion::default();
    pelorus_metal::add_to_map(&stream, cloud_buf, &zero_t, &identity_q, 1);
    pelorus_metal::add_to_map(&stream, cloud_buf, &zero_t, &identity_q, 2); // second triggers UPDATE

    let mut metal_out: Vec<PointXYZINormal> = vec![PointXYZINormal::default(); map_cloud.len()];
    let mut metal_gpu_out = pelorus_metal::MetalBuffer::new(
        &stream,
        std::mem::size_of::<pelorus_metal::PointNormal>(),
        map_cloud.len(),
    );
    pelorus_metal::planes_metal(
        &stream,
        cloud_buf,
        &zero_t,
        &identity_q,
        &zero_t,
        &identity_q,
        &mut metal_out,
        &mut metal_gpu_out,
    );
    pelorus_metal::cleanup_map();

    // ── Compare ─────────────────────────────────────────────────────────────────
    println!(
        "[update test] Plane counts — CPU: {}, Metal: {}",
        cpu_planes.len(),
        metal_out.len()
    );

    assert_eq!(
        cpu_planes.len(),
        metal_out.len(),
        "After voxel update: different plane count CPU={} Metal={}",
        cpu_planes.len(),
        metal_out.len()
    );

    let sort_key = |p: &PointXYZINormal| {
        (
            (p.x * 1000.0) as i32,
            (p.y * 1000.0) as i32,
            (p.z * 1000.0) as i32,
        )
    };
    let mut cpu_sorted = cpu_planes.clone();
    let mut metal_sorted = metal_out.clone();
    cpu_sorted.sort_by_key(&sort_key);
    metal_sorted.sort_by_key(&sort_key);

    let tol = 1e-6_f32;
    let mut max_diff: f32 = 0.0;
    for (c, m) in cpu_sorted.iter().zip(metal_sorted.iter()) {
        let d = (c.normal_x - m.normal_x)
            .abs()
            .max((c.normal_y - m.normal_y).abs())
            .max((c.normal_z - m.normal_z).abs())
            .max((c.intensity - m.intensity).abs());
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d < tol,
            "After update — mismatch at ({:.2},{:.2},{:.2}): CPU=({:.4},{:.4},{:.4},{:.4}) Metal=({:.4},{:.4},{:.4},{:.4}) diff={:.2e}",
            c.x,
            c.y,
            c.z,
            c.normal_x,
            c.normal_y,
            c.normal_z,
            c.intensity,
            m.normal_x,
            m.normal_y,
            m.normal_z,
            m.intensity,
            d
        );
    }
    println!(
        "[update test] Max normal diff after voxel update: {:.2e}",
        max_diff
    );
}
