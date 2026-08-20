//! Low-latency, output-only IMU propagation.
//!
//! This predictor is deliberately separate from the registration EqF.  LiDAR
//! corrections reseed it, so enabling high-rate output cannot perturb mapping
//! or the corrected trajectory.

use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{Receiver, Sender, SyncSender},
    },
    thread,
};

use nalgebra::{Isometry3, Rotation3, Translation3, UnitQuaternion, Vector3};
use pelorus_core::{ImuMsg, ImuParameters, LioConfig};
use ros_pointcloud2::prelude::TimeMsg;

use crate::{
    ImuEkfPrediction,
    eqf::Eqf,
    imu::{ImuPredictionContext, ImuProcessing},
    registration::{RegistrationState, output_pose_covariance, output_twist_covariance},
};

const COMMAND_CAPACITY: usize = 16_384;
const OUTPUT_CAPACITY: usize = 4_096;
const HISTORY_CAPACITY: usize = 32_768;
const GRAVITY: f64 = 9.81;

#[derive(Clone, Debug)]
pub struct RealtimePrediction {
    pub stamp: TimeMsg,
    pub pose: Isometry3<f64>,
    pub linear_velocity_body: Vector3<f64>,
    pub angular_velocity_body: Vector3<f64>,
    pub pose_covariance: [f64; 36],
    pub twist_covariance: [f64; 36],
}

#[derive(Clone)]
struct CorrectedSnapshot {
    state: crate::eqf::StateIKFom,
    covariance: faer::Mat<f64>,
    /// The covariance is only meaningful together with the origin its error is
    /// measured about. Shipping one without the other silently misinterprets every
    /// entry, by more the further the robot is from the world origin.
    anchor: nalgebra::Vector3<f64>,
    initialized: bool,
    last_time: f64,
    context: ImuPredictionContext,
}

enum Command {
    Imu(ImuMsg),
    PrepareExport(CorrectedSnapshot, Vec<ImuMsg>, SyncSender<()>),
    Corrected(CorrectedSnapshot, SyncSender<()>),
    SetBaseToLidar(Isometry3<f64>, SyncSender<()>),
    Shutdown,
}

pub struct RealtimePredictorHandle {
    command_tx: SyncSender<Command>,
    paused: Arc<AtomicBool>,
    pending_corrections: Arc<AtomicUsize>,
}

impl RealtimePredictorHandle {
    pub fn new(config: &LioConfig) -> (Self, Receiver<RealtimePrediction>) {
        Self::new_with_export(config, None)
    }

    pub fn new_with_export(
        config: &LioConfig,
        export_tx: Option<Sender<RealtimePrediction>>,
    ) -> (Self, Receiver<RealtimePrediction>) {
        let (command_tx, command_rx) = std::sync::mpsc::sync_channel(COMMAND_CAPACITY);
        let (output_tx, output_rx) = std::sync::mpsc::sync_channel(OUTPUT_CAPACITY);
        let paused = Arc::new(AtomicBool::new(true));
        let pending_corrections = Arc::new(AtomicUsize::new(0));
        let worker_paused = Arc::clone(&paused);
        let worker_pending_corrections = Arc::clone(&pending_corrections);
        let worker_config = config.clone();
        thread::spawn(move || {
            worker(
                command_rx,
                output_tx,
                export_tx,
                worker_paused,
                worker_pending_corrections,
                worker_config,
            )
        });
        (
            Self {
                command_tx,
                paused,
                pending_corrections,
            },
            output_rx,
        )
    }

    pub fn imu_callback(&self) -> prelio::RealtimeImuCallback {
        let tx = self.command_tx.clone();
        Arc::new(move |imu| {
            // This stream is output-only: it must never backpressure sensor ingress.
            let _ = tx.try_send(Command::Imu(imu));
        })
    }

    pub fn lidar_barrier_callback(&self) -> prelio::RealtimeLidarBarrierCallback {
        let paused = Arc::clone(&self.paused);
        let pending_corrections = Arc::clone(&self.pending_corrections);
        Arc::new(move || {
            pending_corrections.fetch_add(1, Ordering::AcqRel);
            paused.store(true, Ordering::Release);
        })
    }

    fn snapshot(registration: &RegistrationState) -> CorrectedSnapshot {
        CorrectedSnapshot {
            state: registration.ekf.state,
            covariance: registration.ekf.cov.clone(),
            anchor: registration.ekf.anchor,
            initialized: registration.ekf.initialized,
            last_time: registration.ekf.last_time,
            context: registration.imu_processing.prediction_context(),
        }
    }

    /// Produce all prediction-only export samples preceding this correction.
    pub fn prepare_export(&self, registration: &RegistrationState, imu_msgs: &[ImuMsg]) {
        let snapshot = Self::snapshot(registration);
        let (completed_tx, completed_rx) = std::sync::mpsc::sync_channel(0);
        if self
            .command_tx
            .send(Command::PrepareExport(
                snapshot,
                imu_msgs.to_vec(),
                completed_tx,
            ))
            .is_err()
        {
            return;
        }
        let _ = completed_rx.recv();
    }

    /// Install a coherent post-correction state. Buffered IMUs are replayed by
    /// the worker before live publication resumes.
    pub fn corrected(&self, registration: &RegistrationState) {
        let snapshot = Self::snapshot(registration);
        let (completed_tx, completed_rx) = std::sync::mpsc::sync_channel(0);
        if self
            .command_tx
            .send(Command::Corrected(snapshot, completed_tx))
            .is_err()
        {
            return;
        }
        // Ensure replay is complete before registration advances again.
        let _ = completed_rx.recv();
    }

    pub fn set_base_to_lidar(&self, base_to_lidar: Isometry3<f64>) {
        let (completed_tx, completed_rx) = std::sync::mpsc::sync_channel(0);
        if self
            .command_tx
            .send(Command::SetBaseToLidar(base_to_lidar, completed_tx))
            .is_err()
        {
            return;
        }
        let _ = completed_rx.recv();
    }
}

impl Drop for RealtimePredictorHandle {
    fn drop(&mut self) {
        let _ = self.command_tx.try_send(Command::Shutdown);
    }
}

struct WorkerState {
    ekf: Eqf,
    imu: ImuProcessing,
    seeded: bool,
    history: VecDeque<ImuMsg>,
    params: ImuParameters,
    base_to_lidar: Isometry3<f64>,
    output_base: bool,
}

fn worker(
    command_rx: Receiver<Command>,
    output_tx: SyncSender<RealtimePrediction>,
    export_tx: Option<Sender<RealtimePrediction>>,
    paused: Arc<AtomicBool>,
    pending_corrections: Arc<AtomicUsize>,
    config: LioConfig,
) {
    let lidar_to_imu = Isometry3::from_parts(
        Translation3::from(config.imu_params.initial_translation_lidar_to_imu),
        UnitQuaternion::from_matrix(&config.imu_params.initial_rotation_lidar_to_imu),
    );
    let mut state = WorkerState {
        ekf: Eqf::new(config.prelio_params.imu.gravity, config.base_to_lidar),
        imu: ImuProcessing::new(&config.imu_params, &lidar_to_imu, &config.base_to_lidar),
        seeded: false,
        history: VecDeque::new(),
        params: config.imu_params,
        base_to_lidar: config.base_to_lidar,
        output_base: config.base_link_frame.is_some(),
    };
    let mut export_state = export_tx.as_ref().map(|_| WorkerState {
        ekf: Eqf::new(config.prelio_params.imu.gravity, config.base_to_lidar),
        imu: ImuProcessing::new(&config.imu_params, &lidar_to_imu, &config.base_to_lidar),
        seeded: false,
        history: VecDeque::new(),
        params: config.imu_params.clone(),
        base_to_lidar: config.base_to_lidar,
        output_base: config.base_link_frame.is_some(),
    });

    while let Ok(command) = command_rx.recv() {
        match command {
            Command::Imu(imu) => {
                push_history(&mut state.history, imu.clone());
                if state.seeded && !paused.load(Ordering::Acquire) {
                    if let Some(prediction) = state.predict(&imu) {
                        let _ = output_tx.try_send(prediction);
                    }
                }
            }
            Command::PrepareExport(snapshot, imu_msgs, completed) => {
                if let (Some(export_state), Some(export_tx)) =
                    (export_state.as_mut(), export_tx.as_ref())
                {
                    if export_state.seeded {
                        let interval_end = snapshot.last_time;
                        let interval: Vec<_> = imu_msgs
                            .iter()
                            .filter(|imu| {
                                let stamp = pelorus_core::timestamp_to_secs(&imu.stamp);
                                stamp > export_state.ekf.last_time && stamp < interval_end
                            })
                            .cloned()
                            .collect();
                        for imu in interval {
                            if let Some(prediction) = export_state.predict(&imu) {
                                let _ = export_tx.send(prediction);
                            }
                        }
                    }
                    export_state.install(snapshot.clone());
                }
                let _ = completed.send(());
            }

            Command::Corrected(snapshot, completed) => {
                state.install(snapshot);
                let previous_pending = pending_corrections
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |pending| {
                        Some(pending.saturating_sub(1))
                    })
                    .unwrap_or(0);
                let remaining_corrections = previous_pending.saturating_sub(1);
                if remaining_corrections == 0 {
                    let replay: Vec<_> = state
                        .history
                        .iter()
                        .filter(|imu| {
                            pelorus_core::timestamp_to_secs(&imu.stamp) > state.ekf.last_time
                        })
                        .cloned()
                        .collect();
                    let mut latest = None;
                    for imu in replay {
                        latest = state.predict(&imu).or(latest);
                    }
                    // Suppress replay bursts. Publish only the newest caught-up state.
                    if let Some(prediction) = latest {
                        let _ = output_tx.try_send(prediction);
                    }
                    paused.store(false, Ordering::Release);
                }
                let _ = completed.send(());
            }
            Command::SetBaseToLidar(base_to_lidar, completed) => {
                state.set_base_to_lidar(base_to_lidar);
                if let Some(export_state) = export_state.as_mut() {
                    export_state.set_base_to_lidar(base_to_lidar);
                }
                let _ = completed.send(());
            }
            Command::Shutdown => break,
        }
    }
}

fn push_history(history: &mut VecDeque<ImuMsg>, imu: ImuMsg) {
    history.push_back(imu);
    while history.len() > HISTORY_CAPACITY {
        history.pop_front();
    }
}

impl WorkerState {
    fn set_base_to_lidar(&mut self, base_to_lidar: Isometry3<f64>) {
        self.base_to_lidar = base_to_lidar;
        self.ekf.base_to_lidar = base_to_lidar;
        self.imu.set_base_to_lidar(&base_to_lidar);
    }

    fn install(&mut self, snapshot: CorrectedSnapshot) {
        self.ekf.state = snapshot.state;
        self.ekf.cov = snapshot.covariance;
        self.ekf.anchor = snapshot.anchor;
        self.ekf.initialized = snapshot.initialized;
        self.ekf.last_time = snapshot.last_time;
        self.imu.restore_prediction_context(&snapshot.context);
        self.seeded = snapshot.context.ekf_inited;
    }

    fn predict(&mut self, tail: &ImuMsg) -> Option<RealtimePrediction> {
        let context = self.imu.prediction_context();
        let head = &context.last_imu;
        let head_stamp = pelorus_core::timestamp_to_secs(&head.stamp);
        let tail_stamp = pelorus_core::timestamp_to_secs(&tail.stamp);
        let dt = tail_stamp - self.ekf.last_time.max(head_stamp);
        if !dt.is_finite() || dt <= 0.0 {
            return None;
        }
        let mean_acc = self.imu.mean_acc_norm();
        if mean_acc <= f64::EPSILON {
            return None;
        }
        let acc = [
            (head.linear_acceleration[0] + tail.linear_acceleration[0]) * 0.5 * GRAVITY / mean_acc,
            (head.linear_acceleration[1] + tail.linear_acceleration[1]) * 0.5 * GRAVITY / mean_acc,
            (head.linear_acceleration[2] + tail.linear_acceleration[2]) * 0.5 * GRAVITY / mean_acc,
        ];
        let gyro = [
            (head.angular_velocity[0] + tail.angular_velocity[0]) * 0.5,
            (head.angular_velocity[1] + tail.angular_velocity[1]) * 0.5,
            (head.angular_velocity[2] + tail.angular_velocity[2]) * 0.5,
        ];
        self.imu.predict(&mut self.ekf, acc, gyro, dt).ok()?;
        self.ekf.correct_kinematics(
            self.params.kinematic_model,
            Vector3::from(tail.angular_velocity),
            self.params.kinematic_constraint_std,
        );
        self.imu.update_realtime_tail(tail, self.ekf.state);
        Some(self.output(tail.stamp.clone()))
    }

    fn output(&self, stamp: TimeMsg) -> RealtimePrediction {
        let state = ImuEkfPrediction {
            pose: Isometry3::from_parts(
                Translation3::from(self.ekf.state.pos),
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                    &self.ekf.state.rot.matrix(),
                )),
            ),
            linear_velocity: self.ekf.state.vel,
            angular_velocity: self.imu.angvel_last,
            lidar_imu_extrinsic: Isometry3::from_parts(
                Translation3::from(self.ekf.state.offset_t_l_i),
                UnitQuaternion::from_matrix(&self.ekf.state.offset_r_l_i.matrix()),
            ),
            accel_bias: self.ekf.state.ba,
            gyro_bias: self.ekf.state.bg,
        };
        let lidar_pose = state.pose * state.lidar_imu_extrinsic;
        let pose = if self.output_base {
            lidar_pose * self.base_to_lidar.inverse()
        } else {
            lidar_pose
        };
        let lidar_to_output = if self.output_base {
            self.base_to_lidar.inverse()
        } else {
            Isometry3::identity()
        };
        let linear_velocity_body = pose.rotation.inverse() * state.linear_velocity;
        let r_b_l = if self.output_base {
            self.base_to_lidar.rotation.inverse()
        } else {
            UnitQuaternion::identity()
        };
        let angular_velocity_body =
            (r_b_l * state.lidar_imu_extrinsic.rotation.inverse()) * state.angular_velocity;
        RealtimePrediction {
            stamp,
            pose,
            linear_velocity_body,
            angular_velocity_body,
            pose_covariance: output_pose_covariance(&self.ekf, &lidar_to_output),
            twist_covariance: output_twist_covariance(&self.ekf, &pose),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::time::Duration;

    use nalgebra::{Isometry3, Translation3, UnitQuaternion};
    use pelorus_core::{ImuMsg, LioConfig, time_msg_from_secs};

    use super::{RealtimePredictorHandle, WorkerState};
    use crate::{Registration, eqf::Eqf, imu::ImuProcessing, pipeline::Pipeline};

    fn imu(timestamp: f64) -> ImuMsg {
        ImuMsg {
            stamp: time_msg_from_secs(timestamp),
            linear_acceleration: [0.0, 0.0, 1.0],
            angular_velocity: [0.0, 0.0, 0.01],
            ..Default::default()
        }
    }

    #[test]
    fn correction_barrier_collapses_replay_without_mutating_registration() {
        let config = LioConfig::default();
        let pipeline = Pipeline::new(&config);
        let map = pipeline.map();
        {
            let mut registration = map.lock().unwrap();
            registration.init_imu_ekf(&[imu(0.0), imu(0.01)]);
            registration.ekf.last_time = 0.01;
        }

        let (predictor, output) = RealtimePredictorHandle::new(&config);
        let original_state;
        let original_covariance;
        {
            let registration = map.lock().unwrap();
            original_state = registration.ekf.state;
            original_covariance = registration.ekf.cov.clone();
            predictor.corrected(&registration);
        }

        predictor.imu_callback()(imu(0.02));
        let live = output.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(pelorus_core::timestamp_to_secs(&live.stamp), 0.02);

        predictor.lidar_barrier_callback()();
        predictor.imu_callback()(imu(0.03));
        predictor.imu_callback()(imu(0.04));
        assert!(output.recv_timeout(Duration::from_millis(25)).is_err());

        {
            let registration = map.lock().unwrap();
            predictor.prepare_export(&registration, &[]);
            predictor.corrected(&registration);
        }
        let caught_up = output.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(pelorus_core::timestamp_to_secs(&caught_up.stamp), 0.04);
        assert!(output.recv_timeout(Duration::from_millis(25)).is_err());

        let registration = map.lock().unwrap();
        assert_eq!(registration.ekf.state.pos, original_state.pos);
        assert_eq!(
            registration.ekf.state.rot.matrix(),
            original_state.rot.matrix()
        );
        for row in 0..24 {
            for col in 0..24 {
                assert_eq!(
                    registration.ekf.cov[(row, col)],
                    original_covariance[(row, col)]
                );
            }
        }
    }

    #[test]
    fn export_replays_every_imu_between_corrected_timestamps() {
        let config = LioConfig::default();
        let pipeline = Pipeline::new(&config);
        let map = pipeline.map();
        {
            let mut registration = map.lock().unwrap();
            registration.init_imu_ekf(&[imu(0.0), imu(0.01)]);
            registration.ekf.last_time = 0.01;
        }

        let (export_tx, export_rx) = std::sync::mpsc::channel();
        let (predictor, _live_output) =
            RealtimePredictorHandle::new_with_export(&config, Some(export_tx));
        {
            let registration = map.lock().unwrap();
            predictor.prepare_export(&registration, &[]);
            predictor.corrected(&registration);
        }

        predictor.imu_callback()(imu(0.02));
        predictor.imu_callback()(imu(0.03));
        predictor.imu_callback()(imu(0.04));
        predictor.lidar_barrier_callback()();
        {
            let mut registration = map.lock().unwrap();
            registration.ekf.last_time = 0.04;
            predictor.prepare_export(&registration, &[imu(0.02), imu(0.03), imu(0.04)]);
            predictor.corrected(&registration);
        }

        let stamps: Vec<_> = export_rx
            .try_iter()
            .map(|prediction| pelorus_core::timestamp_to_secs(&prediction.stamp))
            .collect();
        assert_eq!(stamps, vec![0.02, 0.03]);
    }

    #[test]
    fn empty_base_frame_predictions_remain_in_lidar_frame() {
        let config = LioConfig::default();
        let lidar_to_imu = Isometry3::from_parts(
            Translation3::from(config.imu_params.initial_translation_lidar_to_imu),
            UnitQuaternion::from_matrix(&config.imu_params.initial_rotation_lidar_to_imu),
        );
        let mut state = WorkerState {
            ekf: Eqf::new(config.prelio_params.imu.gravity, Isometry3::identity()),
            imu: ImuProcessing::new(&config.imu_params, &lidar_to_imu, &Isometry3::identity()),
            seeded: false,
            history: VecDeque::new(),
            params: config.imu_params,
            base_to_lidar: Isometry3::identity(),
            output_base: false,
        };
        state.set_base_to_lidar(Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.28),
            UnitQuaternion::identity(),
        ));

        let prediction = state.output(time_msg_from_secs(1.0));
        assert!(prediction.pose.translation.z.abs() < 1e-12);
    }

    #[test]
    fn prediction_position_covariance_grows_from_corrected_seed() {
        let config = LioConfig::default();
        let pipeline = Pipeline::new(&config);
        let map = pipeline.map();
        {
            let mut registration = map.lock().unwrap();
            registration.init_imu_ekf(&[imu(0.0), imu(0.01)]);
            registration.ekf.last_time = 0.01;
        }

        let corrected_trace = {
            let registration = map.lock().unwrap();
            let covariance = registration.get_output_pose_covariance(&Isometry3::identity());
            covariance[0] + covariance[7] + covariance[14]
        };
        let (predictor, output) = RealtimePredictorHandle::new(&config);
        {
            let registration = map.lock().unwrap();
            predictor.corrected(&registration);
        }

        let mut traces = Vec::new();
        for stamp in [0.02, 0.03, 0.04, 0.05, 0.06] {
            predictor.imu_callback()(imu(stamp));
            let prediction = output.recv_timeout(Duration::from_secs(1)).unwrap();
            traces.push(
                prediction.pose_covariance[0]
                    + prediction.pose_covariance[7]
                    + prediction.pose_covariance[14],
            );
        }

        assert!(traces[0] > corrected_trace);
        assert!(traces.windows(2).all(|pair| pair[1] > pair[0]));
    }
}
