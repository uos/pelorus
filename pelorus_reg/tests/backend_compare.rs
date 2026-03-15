use nalgebra::Isometry3;
use pelorus_core::timing::RuntimeEvaluator;
use pelorus_core::{ImuParameters, PointXYZIT};
use pelorus_reg::{Registration, registration::RegistrationState};
use prelio::LocatedBufferView;
use serde::{Deserialize, Serialize};
use spole::IVoxNearbyType;
use std::fs;
use std::path::PathBuf;

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
    let params = ImuParameters::default();
    let map_resolution = 0.5;
    let mut state = RegistrationState::new(
        map_resolution,
        10000,
        9.81,
        &params,
        0.001,
        1.0,
        Isometry3::identity(),
        Isometry3::identity(),
        IVoxNearbyType::Nearby26,
        0.1,
        5,
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

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    let (simple_view, complex_view) = {
        (
            LocatedBufferView::from_vec_cpu(simple_cloud.clone()),
            LocatedBufferView::from_vec_cpu(complex_cloud.clone()),
        )
    };

    #[cfg(feature = "metal")]
    let (simple_view, complex_view) = {
        (
            LocatedBufferView::copy_from_slice_metal(&stream, &simple_cloud),
            LocatedBufferView::copy_from_slice_metal(&stream, &complex_cloud),
        )
    };

    #[cfg(feature = "cuda")]
    let (simple_view, complex_view) = {
        (
            LocatedBufferView::copy_from_slice_cuda(&stream, &simple_cloud),
            LocatedBufferView::copy_from_slice_cuda(&stream, &complex_cloud),
        )
    };

    let (sender, _receiver) = std::sync::mpsc::channel();

    let _ = state
        .register_and_map(
            true,
            imu_transform,
            &simple_view,
            0.0,
            Some(&sender),
            &mut eval,
            #[cfg(feature = "cuda")]
            &stream,
            #[cfg(feature = "metal")]
            &stream,
        )
        .unwrap();

    let _ = state
        .register_and_map(
            false,
            imu_transform,
            &complex_view,
            0.1,
            Some(&sender),
            &mut eval,
            #[cfg(feature = "cuda")]
            &stream,
            #[cfg(feature = "metal")]
            &stream,
        )
        .unwrap();

    let output_state = DumpedState::from_ekf(&state.get_state());

    let mut gt_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    gt_path.push("tests");
    gt_path.push("backend_gt.json");

    if cfg!(not(any(feature = "metal", feature = "cuda"))) {
        println!("Saving CPU ground truth to {:?}", gt_path);
        let s = serde_json::to_string_pretty(&output_state).expect("Failed to serialize");
        fs::write(&gt_path, s).expect("Failed to write GT");
    } else {
        println!("Checking against CPU ground truth at {:?}", gt_path);
        let s = fs::read_to_string(&gt_path)
            .expect("Ground truth file missing. Run without GPU features first to generate it!");
        let expected_state: DumpedState = serde_json::from_str(&s).expect("Failed to parse GT");

        let tolerance_t = 1e-4; // Back to strict tolerance
        let tolerance_r = 1e-4;
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
}
