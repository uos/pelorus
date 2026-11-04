use alloc::boxed::Box;
use alloc::sync::Arc;
use pelorus_core::sync::Mutex;

use crate::{
    Registration,
    registration::{RegistrationSettings, RegistrationState},
};
use either::Either;
use nalgebra::{Isometry3, Translation3, UnitQuaternion};
#[cfg(feature = "threads")]
use pelorus_core::SyncProcessedCloud;
use pelorus_core::{
    ImuMsg, ImuPosePrediction, ImuPredictor, ImuPredictorUndistort, KinematicModel, LidarVendor,
    LioConfig, OdomMsg,
};
use pelorus_localmap::{IVoxHashMap, IVoxNearbyType};
use pelorus_sync::{ImuInitCallback, ImuSyncMode, SteppedSynchronizer, SteppedSynchronizerHooks};
#[cfg(feature = "threads")]
use pelorus_sync::{Synchronizer, SynchronizerHooks};

/// How densely the constant-twist model is sampled into IMU messages. It only
/// has to be fine enough that the rotation between two samples stays small.
const IMU_SYNTHESIS_RATE: f64 = 200.0;

pub struct Pipeline {
    map: Arc<Mutex<RegistrationState>>,
    pub config: LioConfig,
}

impl Pipeline {
    pub fn new(config: &LioConfig) -> Self {
        let ext_lidar_to_imu = Isometry3::from_parts(
            Translation3::new(
                config.imu_params.initial_translation_lidar_to_imu.x,
                config.imu_params.initial_translation_lidar_to_imu.y,
                config.imu_params.initial_translation_lidar_to_imu.z,
            ),
            UnitQuaternion::from_matrix(&config.imu_params.initial_rotation_lidar_to_imu),
        );

        let mut ivox = IVoxHashMap::new(
            config.map_resolution as f32,
            config.map_capacity,
            IVoxNearbyType::Nearby26,
            config.dist_update_sensitivity,
        );
        if config.map_multiresolution {
            ivox.enable_multiresolution(config.map_capacity, config.dist_update_sensitivity);
        }

        let mut map = RegistrationState::new(
            ivox,
            &config.imu_params,
            &ext_lidar_to_imu,
            &config.base_to_lidar,
            RegistrationSettings {
                gravity: config.sync_params.imu.gravity,
                laser_point_noise_cov: config.lidar_odom_cov,
                ekf_max_iterations: config.ekf_max_iterations,
                parallelism_decision_threshold_stddevs: config
                    .sync_params
                    .parallelism_decision_threshold_stddevs,
                parallelism_decision_samples_per_mode: config
                    .sync_params
                    .parallelism_decision_samples_per_mode,
                parallelism_execution_policy: config.sync_params.parallelism_execution_policy,
            },
            None,
        );

        if config.imu_params.lidar_only {
            let synthetic = crate::lidar_only::SyntheticImu::new(
                config.sync_params.imu.gravity,
                IMU_SYNTHESIS_RATE,
            );
            let _ = map.init_imu_ekf(&synthetic.initialization_buffer(2));
        }

        Self {
            map: Arc::new(Mutex::new(map)),
            config: config.clone(),
        }
    }

    pub fn map(&self) -> Arc<Mutex<RegistrationState>> {
        Arc::clone(&self.map)
    }

    pub fn set_base_to_lidar(&mut self, base_to_lidar: Isometry3<f64>) {
        self.config.base_to_lidar = base_to_lidar;
        self.map.lock().set_base_to_lidar(base_to_lidar);
    }

    pub fn ext(&self) -> Isometry3<f64> {
        self.map.lock().get_state().lidar_imu_extrinsic
    }

    /// Drop the GPU-resident buffer the registration state owns.
    ///
    /// Backend teardown destroys the device, and every child object has to be gone
    /// before that. The plane buffer outlives the map and downsample state, so it
    /// has to be released explicitly. Safe to call more than once. The registration
    /// reallocates it on the next scan.
    pub fn release_gpu_buffers(&mut self) {
        self.map.lock_ignoring_poison().planes_gpu_buffer = None;
    }

    pub fn lidar_vendor(&self) -> LidarVendor {
        self.config.sync_params.lidar_vendor
    }

    pub fn kinematic_model(&self) -> KinematicModel {
        self.config.imu_params.kinematic_model
    }

    pub fn build_init_callback(&self) -> ImuInitCallback {
        let map = self.map();
        let lidar_only = self.config.imu_params.lidar_only;
        Arc::new(Mutex::new(move |msg: &[ImuMsg]| {
            if lidar_only {
                return;
            }
            let mut map = map.lock();
            let _ = map.init_imu_ekf(msg);
        }))
    }

    pub fn build_undistort(&self) -> Box<ImuPredictorUndistort> {
        let map = self.map();
        let kinematic_model = self.config.imu_params.kinematic_model;
        let constraint_std = self.config.imu_params.kinematic_constraint_std;
        let lidar_vendor = self.config.sync_params.lidar_vendor;
        let lidar_only = self.config.imu_params.lidar_only;
        let synthetic = Mutex::new(crate::lidar_only::SyntheticImu::new(
            self.config.sync_params.imu.gravity,
            IMU_SYNTHESIS_RATE,
        ));

        Box::new(
            move |imu_msgs: &[ImuMsg],
                  odom_msgs: &[OdomMsg],
                  last_cloud_end_time,
                  cloud_begin_time,
                  cloud_end_time,
                  cloud,
                  cloud_passthrough| {
                let mut map = map.lock();
                // Feed synthetic samples through the normal estimation path.
                let generated;
                let imu_msgs: &[ImuMsg] = if lidar_only {
                    let pose = map.get_state().pose;
                    let estimates = crate::lidar_only::ImuEstimates {
                        gyro_bias: map.ekf.state.bg,
                        accel_bias: map.ekf.state.ba,
                        gravity: map.ekf.state.grav,
                    };
                    let (messages, linear_velocity) = synthetic.lock().generate(
                        &pose,
                        &estimates,
                        last_cloud_end_time,
                        cloud_end_time,
                    );
                    if let Some(linear_velocity) = linear_velocity {
                        map.set_linear_velocity(linear_velocity);
                    }
                    generated = messages;
                    &generated
                } else {
                    imu_msgs
                };
                map.undistort(
                    crate::UndistortScan {
                        imu_msgs,
                        odom_msgs,
                        last_cloud_end_time,
                        cloud_begin_time,
                        cloud_end_time,
                        cloud,
                        cloud_passthrough,
                        // The synthetic twist is measured over the previous
                        // scan. Under motion that changes from scan to scan it
                        // misplaces points by more than skipping deskew does.
                        deskew: !lidar_only && !matches!(lidar_vendor, LidarVendor::NoTimestamp),
                    },
                    kinematic_model,
                    constraint_std,
                );
                let pose = map.get_state().pose;

                Either::Right(pose)
            },
        )
    }

    pub fn build_init_guess(&self) -> Box<ImuPredictor> {
        let map = self.map();

        Box::new(move |_, _, _, _| {
            let ekfstate = map.lock().get_state();
            ImuPosePrediction {
                pose: ekfstate.pose,
                linear_velocity: ekfstate.linear_velocity,
                angular_velocity: nalgebra::Vector3::default(),
            }
        })
    }

    /// `lossless` applies backpressure to the caller instead of dropping input.
    fn imu_sync_mode(&self, lossless: bool) -> ImuSyncMode {
        if self.config.imu_params.lidar_only {
            ImuSyncMode::Disabled { lossless }
        } else if self.config.sync_params.lidar_as_sync {
            ImuSyncMode::LidarDriven
        } else {
            ImuSyncMode::Continuous { lossless }
        }
    }

    /// Synchronizer that runs on its own threads and hands each synchronized
    /// scan to the returned channel.
    ///
    /// No input is dropped. When the queues are full, adding input blocks until
    /// the threads catch up, so the same input sequence produces the same scans
    /// as a [`SteppedSynchronizer`].
    #[cfg(feature = "threads")]
    pub fn create_sync(
        &self,
        ready_for_next: Option<std::sync::mpsc::Receiver<u64>>,
    ) -> anyhow::Result<(Synchronizer, std::sync::mpsc::Receiver<SyncProcessedCloud>)> {
        let params = &self.config.sync_params;
        let gravity = params.imu.gravity;
        let madgwick_filter = params.imu.madgwick_setting;
        let imu_calib = params.imu.calibration.clone();
        let undistort = Some(self.build_undistort());
        let init_guess = Some(self.build_init_guess());
        let imu_sync_mode = self.imu_sync_mode(true);
        let mut sync = Synchronizer::create(
            gravity,
            madgwick_filter,
            &imu_calib,
            params,
            params.lidar_vendor,
            imu_sync_mode,
            SynchronizerHooks {
                imu_pose_prediction_undistort: undistort,
                imu_pose_prediction_init_guess: init_guess,
                ready_for_next,
                ..Default::default()
            },
        )?;

        let rx = sync
            .get_chan()
            .ok_or_else(|| anyhow::anyhow!("Synchronizer channel is not available"))?;

        Ok((sync, rx))
    }

    /// Synchronizer that runs on the calling thread. Register every scan it
    /// returns before asking it for the next one.
    pub fn create_stepped_sync(&self) -> anyhow::Result<SteppedSynchronizer> {
        let params = &self.config.sync_params;
        SteppedSynchronizer::create(
            params,
            params.lidar_vendor,
            self.imu_sync_mode(false),
            SteppedSynchronizerHooks {
                imu_pose_prediction_undistort: Some(self.build_undistort()),
                imu_pose_prediction_init_guess: Some(self.build_init_guess()),
                ..Default::default()
            },
        )
    }
}
