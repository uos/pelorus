use nalgebra::Isometry3;
use pelorus_core::timing::RuntimeEvaluator;
use pelorus_core::{ImuParameters, PointXYZIT};
use pelorus_reg::{ActiveBackend, Registration, registration::RegistrationState};
use prelio::LocatedBufferView;
use serde::{Deserialize, Serialize};
use spole::{IVoxHashMap, IVoxNearbyType};
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

    let (sender, _receiver) = std::sync::mpsc::channel();

    let _ = state
        .register_and_map(
            true,
            imu_transform,
            &simple_view,
            0.0,
            1,
            Some(&sender),
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
            Some(&sender),
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

    let mut cpu_map = spole::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &Isometry3::identity());

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    for pt in &query_cloud {
        let q = PointXYZ::new(pt.x, pt.y, pt.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * pt.x + plane[1] * pt.y + plane[2] * pt.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, pd2, plane[0], plane[1], plane[2],
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

    let mut cpu_map = spole::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &Isometry3::identity());
    cpu_map.add(&map_cloud, &Isometry3::identity());

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    for pt in &map_cloud {
        let q = PointXYZ::new(pt.x, pt.y, pt.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * pt.x + plane[1] * pt.y + plane[2] * pt.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, pd2, plane[0], plane[1], plane[2],
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
    pelorus_cuda::add_to_map(&stream, cloud_buf, &zero_t, &identity_q);
    pelorus_cuda::add_to_map(&stream, cloud_buf, &zero_t, &identity_q);

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

#[cfg(feature = "vulkan")]
#[test]
fn test_planes_direct_equivalency_vulkan() {
    use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};
    let _lock = VULKAN_MAP_MUTEX.lock().unwrap();
    let resolution = 0.5_f32;
    let map_cloud = create_simple_cloud();
    let query_cloud = create_complex_cloud();

    let mut cpu_map = spole::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &Isometry3::identity());

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    for pt in &query_cloud {
        let q = PointXYZ::new(pt.x, pt.y, pt.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * pt.x + plane[1] * pt.y + plane[2] * pt.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, pd2, plane[0], plane[1], plane[2],
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
    pelorus_vulkan::planes_vulkan(
        &stream,
        query_buf,
        &pelorus_vulkan::Vector3::new(0.0, 0.0, 0.0),
        &pelorus_vulkan::Quaternion::default(),
        &pelorus_vulkan::Vector3::new(0.0, 0.0, 0.0),
        &pelorus_vulkan::Quaternion::default(),
        &mut vulkan_planes_out,
        &mut vulkan_planes_gpu,
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

    let mut cpu_map = spole::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &Isometry3::identity());
    cpu_map.add(&map_cloud, &Isometry3::identity());

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    for pt in &map_cloud {
        let q = PointXYZ::new(pt.x, pt.y, pt.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * pt.x + plane[1] * pt.y + plane[2] * pt.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, pd2, plane[0], plane[1], plane[2],
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
    pelorus_vulkan::add_to_map(&stream, cloud_buf, &zero_t, &identity_q);
    pelorus_vulkan::add_to_map(&stream, cloud_buf, &zero_t, &identity_q);

    let mut vulkan_out: Vec<PointXYZINormal> = vec![PointXYZINormal::default(); map_cloud.len()];
    let mut vulkan_gpu_out = pelorus_vulkan::VulkanBuffer::new(
        &stream,
        std::mem::size_of::<pelorus_vulkan::PointNormal>(),
        map_cloud.len(),
    );
    pelorus_vulkan::planes_vulkan(
        &stream,
        cloud_buf,
        &zero_t,
        &identity_q,
        &zero_t,
        &identity_q,
        &mut vulkan_out,
        &mut vulkan_gpu_out,
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

#[cfg(feature = "metal")]
#[test]
fn test_planes_direct_equivalency() {
    use ros_pointcloud2::points::{PointXYZ, PointXYZINormal};
    let _lock = METAL_MAP_MUTEX.lock().unwrap();
    let resolution = 0.5_f32;
    let map_cloud = create_simple_cloud();
    let query_cloud = create_complex_cloud();

    // ── CPU path ────────────────────────────────────────────────────────────────
    let mut cpu_map = spole::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &Isometry3::identity());

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    for pt in &query_cloud {
        let q = PointXYZ::new(pt.x, pt.y, pt.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * pt.x + plane[1] * pt.y + plane[2] * pt.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, pd2, plane[0], plane[1], plane[2],
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
    let mut cpu_map = spole::IVoxHashMap::new(resolution, 10000, IVoxNearbyType::Nearby26, 0.1);
    cpu_map.add(&map_cloud, &Isometry3::identity());
    cpu_map.add(&map_cloud, &Isometry3::identity()); // second add → voxel UPDATE

    let mut cpu_planes: Vec<PointXYZINormal> = Vec::new();
    for pt in &map_cloud {
        let q = PointXYZ::new(pt.x, pt.y, pt.z);
        if let Some(plane) = cpu_map.plane_fit_from_distributions(&q) {
            let pd2 = plane[0] * pt.x + plane[1] * pt.y + plane[2] * pt.z + plane[3];
            let l2_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            let s = 1.0_f32 - 0.9 * pd2.abs() / l2_sq.max(1e-9_f32).sqrt().sqrt();
            if s > 0.9 {
                cpu_planes.push(PointXYZINormal::new(
                    pt.x, pt.y, pt.z, pd2, plane[0], plane[1], plane[2],
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
    pelorus_metal::add_to_map(&stream, cloud_buf, &zero_t, &identity_q);
    pelorus_metal::add_to_map(&stream, cloud_buf, &zero_t, &identity_q); // second → UPDATE

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
