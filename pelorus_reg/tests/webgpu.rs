//! Exercises the actual iterated filter, map updates and failure fallback.
#![cfg(feature = "webgpu")]

use nalgebra::{Isometry3, Point3, Vector3};
use pelorus_core::{ImuParameters, PointXYZIT, timing::RuntimeEvaluator};
use pelorus_localmap::{IVoxHashMap, IVoxNearbyType};
use pelorus_reg::{
    ActiveBackend, Registration, RegistrationScan,
    registration::{RegistrationSettings, RegistrationState},
};
use pelorus_sync::LocatedBufferView;

fn state(multiresolution: bool) -> RegistrationState {
    let mut map = IVoxHashMap::new(0.5, 10000, IVoxNearbyType::Nearby26, 0.1);
    if multiresolution {
        map.enable_multiresolution(10000, 0.1);
    }
    RegistrationState::new(
        map,
        &ImuParameters::default(),
        &Isometry3::identity(),
        &Isometry3::identity(),
        RegistrationSettings::new(9.81, 0.001, 5),
        None,
    )
}

#[test]
#[ignore = "requires a compute adapter"]
fn async_registration_matches_cpu_and_survives_gpu_failure() {
    futures_lite::future::block_on(async {
        for multiresolution in [false, true] {
            let mut gpu = pelorus_wgpu::Correspondence::new().await.unwrap();
            let mut accelerated = state(multiresolution);
            let mut cpu = state(multiresolution);
            let mut eval_cpu = RuntimeEvaluator::new();
            let mut eval_gpu = RuntimeEvaluator::new();
            for scan_id in 0..8 {
                let movement = Isometry3::new(
                    Vector3::new(
                        scan_id as f64 * 0.004,
                        -0.003 * scan_id as f64,
                        0.002 * scan_id as f64,
                    ),
                    Vector3::new(0.0, 0.001 * scan_id as f64, 0.0),
                );
                let mut points = Vec::new();
                for i in -15..=15 {
                    for j in -15..=15 {
                        let a = i as f64 * 0.09;
                        let b = j as f64 * 0.09;
                        for p in [
                            Point3::new(a, b, 2.0),
                            Point3::new(a, 2.0, b),
                            Point3::new(2.0, a, b),
                        ] {
                            let p = movement.inverse_transform_point(&p);
                            points.push(PointXYZIT {
                                x: p.x as f32,
                                y: p.y as f32,
                                z: p.z as f32,
                                ..Default::default()
                            });
                        }
                    }
                }
                let cloud = LocatedBufferView::Ram(points);
                let scan = || RegistrationScan {
                    cloud: &cloud,
                    imu_transform: Isometry3::identity(),
                    lidar_timestamp: scan_id as f64 * 0.1,
                    scan_id,
                    map_only: scan_id == 0,
                };
                // Failure occurs after GPU-backed scans, so fallback must use an up-to-date CPU map.
                if scan_id == 5 {
                    gpu.disable("simulated device loss");
                }
                let a = accelerated
                    .register_and_map_async(
                        scan(),
                        None,
                        &mut eval_gpu,
                        &ActiveBackend::Cpu,
                        Some(&mut gpu),
                    )
                    .await
                    .unwrap();
                let b = cpu
                    .register_and_map(scan(), None, &mut eval_cpu, &ActiveBackend::Cpu)
                    .unwrap();
                assert_eq!(a.is_some(), b.is_some());
                if scan_id > 0 {
                    assert!(
                        !accelerated.planes_cpu_buffer.is_empty(),
                        "must actually register correspondences"
                    );
                    let a = accelerated.get_state().pose;
                    let b = cpu.get_state().pose;
                    assert!(
                        (a.translation.vector - b.translation.vector).norm() < 0.002,
                        "translation differs on scan {scan_id}"
                    );
                    assert!(
                        a.rotation.angle_to(&b.rotation) < 0.002,
                        "rotation differs on scan {scan_id}"
                    );
                }
                if scan_id < 5 {
                    assert!(gpu.failure().is_none(), "GPU must remain active");
                }
            }
        }
    });
}
