pub mod tf;

use log::{error, warn};
pub use prelio;

use anyhow::anyhow;
use prelio::config::LifisParameters;
use prelio::imu::ImuMsg;
use prelio::timestamp_to_secs;
use ros2_client::ros2;
use ros2_interfaces_jazzy::geometry_msgs::msg::{
    Point, Pose, PoseWithCovariance, Quaternion, Twist, TwistWithCovariance,
};
use ros2_interfaces_jazzy::{
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::{Imu, PointCloud2},
};
use ros_pointcloud2::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Default, PointConvertible)]
pub struct SkipPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

const N_NTH_SCAN: usize = 1; // TODO to config

impl prelio::PointXYZITimestamp<3> for SkipPoint {
    fn intensity(&self) -> f32 {
        -1.0
    }

    fn timestamp(&self) -> f64 {
        -1.0
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

impl SkipPoint {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        SkipPoint { x, y, z }
    }
}

#[derive(Debug, Clone, Default, PointConvertible)]
pub struct RobosensePoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: u8,
    pub ring: u16,
    pub timestamp: f64,
}

impl prelio::PointXYZITimestamp<6> for RobosensePoint {
    fn intensity(&self) -> f32 {
        self.intensity as f32 / 255.0 // TODO right?
    }

    fn timestamp(&self) -> f64 {
        self.timestamp * 1.0
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible)]
pub struct VelodynePoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub time: f32,
}

impl prelio::PointXYZITimestamp<5> for VelodynePoint {
    fn intensity(&self) -> f32 {
        self.intensity
    }

    fn timestamp(&self) -> f64 {
        self.time as f64 * 1e-6
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Default, PointConvertible)]
pub struct OusterPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    //pub intensity: f32,
    pub t: u32,
}

impl prelio::PointXYZITimestamp<4> for OusterPoint {
    fn intensity(&self) -> f32 {
        //self.intensity
        0.0
    }

    fn timestamp(&self) -> f64 {
        self.t as f64 * 1e-9
    }

    fn xyz(&self) -> nalgebra::Point3<f32> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

pub struct State {
    pub global_polygon: Option<Vec<nalgebra::Point2<f32>>>,
    pub sensor_tf: Option<nalgebra::geometry::Isometry3<f32>>,
    pub imu_tf: Option<nalgebra::geometry::Isometry3<f64>>,

    pub lifis: Arc<Mutex<prelio::Lifis>>,
    pub lifis_params: LifisParameters,
    take_nth_scan: usize,
    pub lidar_vendor: prelio::LidarVendor,
}

impl State {
    pub fn new(
        config: &LifisParameters,
        imu_pose_prediction_undistort: Option<Box<prelio::ImuPredictorUndistort>>,
        imu_pose_prediction_init_guess: Option<Box<prelio::ImuPredictor>>,
    ) -> anyhow::Result<Self> {
        Ok(State {
            global_polygon: None,
            sensor_tf: None,
            imu_tf: None,
            lifis: match config.lidar_vendor {
                prelio::LidarVendor::Velodyne => {
                    Arc::new(Mutex::new(prelio::Lifis::create::<5, VelodynePoint>(
                        config.imu.gravity,
                        config.imu.madgwick_setting,
                        &config.imu.calibration,
                        &config, // TODO more specific
                        imu_pose_prediction_undistort,
                        imu_pose_prediction_init_guess,
                        Some(config.lidar_vendor),
                    )?))
                }
                prelio::LidarVendor::Robosense => {
                    Arc::new(Mutex::new(prelio::Lifis::create::<6, RobosensePoint>(
                        config.imu.gravity,
                        config.imu.madgwick_setting,
                        &config.imu.calibration,
                        &config, // TODO more specific
                        imu_pose_prediction_undistort,
                        imu_pose_prediction_init_guess,
                        Some(config.lidar_vendor),
                    )?))
                }
                prelio::LidarVendor::Ouster => {
                    Arc::new(Mutex::new(prelio::Lifis::create::<4, OusterPoint>(
                        config.imu.gravity,
                        config.imu.madgwick_setting,
                        &config.imu.calibration,
                        &config, // TODO more specific
                        imu_pose_prediction_undistort,
                        imu_pose_prediction_init_guess,
                        Some(config.lidar_vendor),
                    )?))
                }
                prelio::LidarVendor::Skip => {
                    Arc::new(Mutex::new(prelio::Lifis::create::<3, SkipPoint>(
                        config.imu.gravity,
                        config.imu.madgwick_setting,
                        &config.imu.calibration,
                        &config, // TODO more specific
                        imu_pose_prediction_undistort,
                        imu_pose_prediction_init_guess,
                        Some(config.lidar_vendor),
                    )?))
                }
            },
            lifis_params: config.clone(),
            take_nth_scan: N_NTH_SCAN,
            lidar_vendor: config.lidar_vendor,
        })
    }
}

fn cloud_callback(
    state: Arc<Mutex<State>>,
    tf_listener: Arc<tf::TfListener>,
    msg: PointCloud2,
    first_imu_tf: Option<String>,
) -> anyhow::Result<()> {
    let orig_header = msg.header.clone();
    let in_msg: PointCloud2Msg = msg.into();

    let in_stamp = orig_header.stamp;
    let points_n = (in_msg.dimensions.width * in_msg.dimensions.height) as usize;
    if points_n == 0 {
        return Ok(());
    }

    {
        let (imu_tf, sensor_tf) = {
            let (imu_tf_override, global_zero_frame, _, sensor_tf) = {
                let mut state = state.lock().unwrap();

                // throttle the number of scans
                if state.take_nth_scan > 1 {
                    state.take_nth_scan -= 1;
                    return Ok(());
                } else {
                    state.take_nth_scan = N_NTH_SCAN;
                }

                (
                    state.lifis_params.imu_lidar_tf_override,
                    state.lifis_params.global_zero_frame.clone(),
                    state.lifis_params.map_frame.clone(),
                    state.sensor_tf.clone(),
                )
            };

            {
                let sensor_tf = match sensor_tf {
                    Some(tf) => tf,
                    None => {
                        if global_zero_frame.is_empty() {
                            nalgebra::geometry::Isometry3::identity()
                        } else {
                            let tf = tf_listener.lookup_transform(
                                orig_header.frame_id.clone().as_str(),
                                global_zero_frame.clone().as_str(),
                                in_stamp.clone(),
                            )?;
                            nalgebra::geometry::Isometry3::from_parts(
                                nalgebra::geometry::Translation3::new(
                                    tf.transform.translation.x as f32,
                                    tf.transform.translation.y as f32,
                                    tf.transform.translation.z as f32,
                                ),
                                nalgebra::geometry::UnitQuaternion::from_quaternion(
                                    nalgebra::geometry::Quaternion::new(
                                        tf.transform.rotation.w as f32,
                                        tf.transform.rotation.x as f32,
                                        tf.transform.rotation.y as f32,
                                        tf.transform.rotation.z as f32,
                                    ),
                                ),
                            )
                            .inverse()
                        }
                    }
                };

                let imu_tf = if imu_tf_override {
                    match first_imu_tf {
                        None => {
                            warn!("Waiting for first IMU Message to get frame.");
                            return Ok(());
                        }
                        Some(first_imu_tf_frame) => {
                            let tf = tf_listener.lookup_transform(
                                &first_imu_tf_frame,
                                &orig_header.frame_id,
                                in_stamp.clone(),
                            )?;
                            nalgebra::geometry::Isometry3::from_parts(
                                nalgebra::geometry::Translation3::new(
                                    tf.transform.translation.x,
                                    tf.transform.translation.y,
                                    tf.transform.translation.z,
                                ),
                                nalgebra::geometry::UnitQuaternion::from_quaternion(
                                    nalgebra::geometry::Quaternion::new(
                                        tf.transform.rotation.w,
                                        tf.transform.rotation.x,
                                        tf.transform.rotation.y,
                                        tf.transform.rotation.z,
                                    ),
                                ),
                            )
                            .inverse()
                        }
                    }
                } else {
                    state
                        .lock()
                        .unwrap()
                        .imu_tf
                        .unwrap_or(nalgebra::geometry::Isometry3::identity())
                };

                (imu_tf, sensor_tf)
            }
        };

        {
            let mut state = state.lock().unwrap();
            state.sensor_tf = Some(sensor_tf);
            state.imu_tf = Some(imu_tf);
        }

        let params = {
            let (global_polygon, invert_polygon, global_zero_frame, map_frame, mut params) = {
                let state = state.lock().unwrap();

                (
                    state.global_polygon.clone(),
                    state.lifis_params.passthrough_filter.invert_polygon,
                    state.lifis_params.global_zero_frame.clone(),
                    state.lifis_params.map_frame.clone(),
                    state.lifis_params.clone(), // TODO overhead clone on every cloud but now params can not change anymore while runtime
                )
            };

            if let Some(poly) = global_polygon.as_ref() {
                let tf = if global_zero_frame.is_empty() {
                    nalgebra::geometry::Isometry3::identity()
                } else {
                    let tf = tf_listener.lookup_transform(
                        map_frame.as_str(),
                        global_zero_frame.as_str(),
                        in_stamp,
                    )?;
                    if map_frame.is_empty() {
                        return Err(anyhow!("global_zero_frame is given but map_frame is not. Can not update polygon."));
                    }
                    nalgebra::geometry::Isometry3::from_parts(
                        nalgebra::geometry::Translation3::new(
                            tf.transform.translation.x as f32,
                            tf.transform.translation.y as f32,
                            tf.transform.translation.z as f32,
                        ),
                        nalgebra::geometry::UnitQuaternion::from_quaternion(
                            nalgebra::geometry::Quaternion::new(
                                tf.transform.rotation.w as f32,
                                tf.transform.rotation.x as f32,
                                tf.transform.rotation.y as f32,
                                tf.transform.rotation.z as f32,
                            ),
                        ),
                    )
                    .inverse()
                };

                let mut cupcl_poly = Vec::new();
                for p in poly {
                    let tp = tf * nalgebra::Point3::new(p.x, p.y, 0.0);
                    cupcl_poly.push(nalgebra::Point2::new(tp.x, tp.y));
                }
                params.passthrough_filter.polygon = Some(cupcl_poly);
                params.passthrough_filter.invert_polygon = invert_polygon;
            };

            params
        };

        {
            let mut state = state.lock().unwrap();
            state.lifis_params = params;
        }
    };

    {
        let state = state.lock().unwrap();
        let mut lifis = state.lifis.lock().unwrap();
        lifis.add_pointcloud(in_msg);
    }

    Ok(())
}

pub struct LifisNode {
    pub state: Arc<std::sync::Mutex<State>>,
    pub tf_listener: Arc<tf::TfListener>,
    pub logger: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifisNodeParameters {
    pub cloud_src_topic: String,
    pub imu_src_topic: String,
    pub logger_prefix: String,
    pub imu_filtered_topic: String,
    pub imu_predictor_init_time_sec: f64,
    pub cloud_qos: Qos,
    pub imu_qos: Qos,
    pub imu_filtered_qos: Qos,
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug, Default)]
pub enum Qos {
    Sensor,
    #[default]
    SystemDefault,
}

/*

Qos::SystemDefault => Ok(ros2::QosPolicyBuilder::new() QosPolicyBuilder
                  .durability(ros2::policy::Durability::Volatile) QosPolicyBuilder
                  .deadline(ros2::policy::Deadline(ros2::Duration::INFINITE)) QosPolicyBuilder
                  .ownership(ros2::policy::Ownership::Shared) QosPolicyBuilder
                  .reliability(ros2::policy::Reliability::Reliable {
                      max_blocking_time: ros2::Duration::from_millis(100),
                  }) QosPolicyBuilder
                  .history(ros2::policy::History::KeepLast { depth: 1 }) QosPolicyBuilder
                  .lifespan(ros2::policy::Lifespan {
                      duration: ros2::Duration::INFINITE,
                  }) QosPolicyBuilder
                  .build()),
            }*/

impl From<crate::Qos> for ros2::QosPolicies {
    fn from(value: crate::Qos) -> Self {
        match value {
            Qos::Sensor => ros2::QosPolicyBuilder::new()
                .durability(ros2::policy::Durability::Volatile)
                .deadline(ros2::policy::Deadline(ros2::Duration::INFINITE))
                .ownership(ros2::policy::Ownership::Shared)
                .reliability(ros2::policy::Reliability::BestEffort)
                .history(ros2::policy::History::KeepLast { depth: 5 })
                .lifespan(ros2::policy::Lifespan {
                    duration: ros2::Duration::INFINITE,
                })
                .build(),
            _ => ros2::QosPolicyBuilder::new()
                .durability(ros2::policy::Durability::Volatile)
                .deadline(ros2::policy::Deadline(ros2::Duration::INFINITE))
                .ownership(ros2::policy::Ownership::Shared)
                .reliability(ros2::policy::Reliability::Reliable {
                    max_blocking_time: ros2::Duration::from_millis(100),
                })
                .history(ros2::policy::History::KeepLast { depth: 1 })
                .lifespan(ros2::policy::Lifespan {
                    duration: ros2::Duration::INFINITE,
                })
                .build(),
        }
    }
}

impl Default for LifisNodeParameters {
    fn default() -> Self {
        LifisNodeParameters {
            cloud_src_topic: "cloud".to_string(),
            imu_src_topic: "imu".to_string(),
            logger_prefix: "lifis".to_string(),
            imu_filtered_topic: "imu_filtered".to_string(),
            imu_predictor_init_time_sec: 0.1,
            cloud_qos: Qos::Sensor,
            imu_qos: Qos::Sensor,
            imu_filtered_qos: Qos::Sensor,
        }
    }
}

#[derive(Debug)]
pub struct LifisTick {
    pub deskewed_cloud: Vec<PointXYZ>,
    pub cloud_header: HeaderMsg,
    pub imu_odom: Odometry,
    //pub runtime_eval: prelio::timing::RuntimeEvaluator,
}

impl LifisNode {
    pub fn new(
        node: &mut ros2_client::Node,
        params: &LifisParameters,
        node_params: &LifisNodeParameters,
        imu_pose_predictor_init: Option<Arc<Mutex<dyn FnMut(&[ImuMsg]) + Send + Sync>>>,
        imu_pose_prediction_undistort: Option<Box<prelio::ImuPredictorUndistort>>,
        imu_pose_prediction_init_guess: Option<Box<prelio::ImuPredictor>>,
    ) -> anyhow::Result<(Self, std::sync::mpsc::Receiver<LifisTick>)> {
        let tf_listener = Arc::new(tf::TfListener::new(node));

        let state = State::new(
            &params,
            imu_pose_prediction_undistort,
            imu_pose_prediction_init_guess,
        )?;

        let (chan, imu_frame) = {
            let mut lifis = state.lifis.lock().unwrap();
            let chan = lifis.get_chan().unwrap();

            let imu_frame = state.lifis_params.imu.imu_frame.clone();
            (chan, imu_frame)
        };

        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            Self::transform(tx, chan, &imu_frame);
        });

        // first option None = override not set.
        let imu_tf_frame: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        let cloud_subscriber = node
            .create_subscription::<PointCloud2>(
                &node
                    .create_topic(
                        &ros2_client::Name::new("/", &node_params.cloud_src_topic).unwrap(),
                        ros2_client::MessageTypeName::new("sensor_msgs", "PointCloud2"),
                        &node_params.cloud_qos.into(),
                    )
                    .unwrap(),
                None,
            )
            .unwrap();

        // Share with cloud callback, TODO maybe the only one who needs it, so remove mutex and arc?
        let state = Arc::new(std::sync::Mutex::new(state));

        let tf_listener_clone = tf_listener.clone();
        let state_clone = state.clone();
        let imu_tf_frame_lidar_inner = Arc::clone(&imu_tf_frame);
        std::thread::spawn(move || loop {
            match cloud_subscriber.take() {
                Err(e) => {
                    error!("Could not read cloud: {e}");
                }
                Ok(Some((msg, _msg_info))) => {
                    // TODO if config says we are system time, we can also the source timestamp of dds
                    /*if let Some(source_timestamp) = msg_info.source_timestamp() {
                        msg.header.stamp = source_timestamp
                    }*/
                    let imu_tf_frame_lidar_inner =
                        imu_tf_frame_lidar_inner.clone().lock().unwrap().clone();

                    match cloud_callback(
                        state_clone.clone(),
                        tf_listener_clone.clone(),
                        msg,
                        imu_tf_frame_lidar_inner,
                    ) {
                        Ok(_) => {}
                        Err(e) => {
                            error!("Error in cloud callback: {e}");
                        }
                    }
                }
                Ok(None) => {}
            }
        });

        let state_clone = state.clone();
        let imu_subscription = node
            .create_subscription::<Imu>(
                &node
                    .create_topic(
                        &ros2_client::Name::new("/", &node_params.imu_src_topic).unwrap(),
                        ros2_client::MessageTypeName::new("sensor_msgs", "Imu"),
                        &node_params.imu_qos.into(),
                    )
                    .unwrap(),
                None,
            )
            .unwrap();

        // do not publish filtered imu if no topic is provided
        let imu_filtered_publisher = if node_params.imu_filtered_topic == "" {
            None
        } else {
            Some(Arc::new(Mutex::new(
                node.create_publisher(
                    &node
                        .create_topic(
                            &ros2_client::Name::new("/", &node_params.imu_filtered_topic).unwrap(),
                            ros2_client::MessageTypeName::new("sensor_msgs", "Imu"),
                            &node_params.imu_filtered_qos.into(),
                        )
                        .unwrap(),
                    None,
                )
                .unwrap(),
            )))
        };

        let imu_init_time_counter = Arc::new(Mutex::new(None));
        let imu_init_buffer = Arc::new(Mutex::new(Some(Vec::new())));
        let imu_tf_frame_lidar_inner = Arc::clone(&imu_tf_frame);
        let init_imu_time = node_params.imu_predictor_init_time_sec;
        std::thread::spawn(move || {
            let imu_pose_predictor_init = imu_pose_predictor_init.clone();

            match imu_subscription.take() {
                Err(e) => {
                    error!("Could not read imu: {e}");
                }
                Ok(Some((msg, _msg_info))) => {
                    imu_tf_frame_lidar_inner
                        .clone()
                        .lock()
                        .unwrap()
                        .get_or_insert(msg.header.frame_id.clone());
                    let imu_init_time_counter = imu_init_time_counter.clone();
                    let imu_init_buffer = imu_init_buffer.clone();

                    let imu_lifis = imu_to_lifis(msg.clone());
                    let curr_stamp = timestamp_to_secs(&imu_lifis.stamp);

                    if let Some(imu_pose_predictor_init) = imu_pose_predictor_init.as_ref() {
                        let mut imu_pose_predictor_init = imu_pose_predictor_init.lock().unwrap();
                        let mut imu_init_time_counter = imu_init_time_counter.lock().unwrap();
                        if imu_init_time_counter.is_none() {
                            *imu_init_time_counter = Some(curr_stamp);
                        }
                        if let Some(init_time) = imu_init_time_counter.as_ref() {
                            let mut imu_init_buffer = imu_init_buffer.lock().unwrap();
                            let delta = curr_stamp - init_time;
                            if delta < init_imu_time {
                                if let Some(imu_init_buffer) = imu_init_buffer.as_mut() {
                                    imu_init_buffer.push(imu_lifis);
                                    return; // skip adding to lifis if we are still in the init phase
                                }
                                // we are only here, if there was an initial buffer, but it was already consumed so skip init and continue with normal processing
                            } else {
                                // apply the init buffer
                                if let Some(imu_init_buffer) = imu_init_buffer.take() {
                                    imu_pose_predictor_init(&imu_init_buffer);
                                }
                            }
                        }
                    }

                    let state = state_clone.lock().unwrap();
                    let filtered = state.lifis.lock().unwrap().add_imu(imu_to_lifis(msg));
                    match (filtered, imu_filtered_publisher.as_ref()) {
                        (Some(filtered), Some(imu_filtered_publisher)) => {
                            imu_filtered_publisher
                                .clone()
                                .lock()
                                .unwrap()
                                .publish(&lifis_to_imu(filtered))
                                .expect("could not publish filtered imu");
                        }
                        _ => {}
                    }
                }
                Ok(None) => {}
            }
        });

        Ok((
            Self {
                state,
                tf_listener,
                logger: "".to_string(), // TODO rm logger prefix
            },
            rx,
        ))
    }

    pub fn transform(
        sender: std::sync::mpsc::Sender<LifisTick>,
        receiver: std::sync::mpsc::Receiver<prelio::LifisProcessedCloud>,
        imu_frame: &String,
    ) {
        while let Ok(synced_data) = receiver.recv() {
            let pose = PoseWithCovariance {
                pose: Pose {
                    position: Point {
                        x: synced_data.end_of_frame_guess.translation.vector.x,
                        y: synced_data.end_of_frame_guess.translation.vector.y,
                        z: synced_data.end_of_frame_guess.translation.vector.z,
                    },
                    orientation: Quaternion {
                        x: synced_data.end_of_frame_guess.rotation.quaternion().i,
                        y: synced_data.end_of_frame_guess.rotation.quaternion().j,
                        z: synced_data.end_of_frame_guess.rotation.quaternion().k,
                        w: synced_data.end_of_frame_guess.rotation.quaternion().w,
                    },
                },
                covariance: synced_data.pose_covariance.into(),
            };

            let twist = TwistWithCovariance {
                twist: Twist {
                    linear: ros2_interfaces_jazzy::geometry_msgs::msg::Vector3 {
                        x: synced_data.end_of_frame_twist.linear.x,
                        y: synced_data.end_of_frame_twist.linear.y,
                        z: synced_data.end_of_frame_twist.linear.z,
                    },
                    angular: ros2_interfaces_jazzy::geometry_msgs::msg::Vector3 {
                        x: synced_data.end_of_frame_twist.angular.x,
                        y: synced_data.end_of_frame_twist.angular.y,
                        z: synced_data.end_of_frame_twist.angular.z,
                    },
                },
                covariance: synced_data.twist_covariance.into(),
            };

            let odom = ros2_interfaces_jazzy::nav_msgs::msg::Odometry {
                header: ros2_interfaces_jazzy::std_msgs::msg::Header {
                    stamp: ros2_interfaces_jazzy::builtin_interfaces::msg::Time {
                        sec: synced_data.end_of_frame_time.sec,
                        nanosec: synced_data.end_of_frame_time.nanosec,
                    },
                    frame_id: synced_data.header.frame_id.clone(),
                },
                child_frame_id: imu_frame.clone(),
                pose,
                twist,
            };

            let res = LifisTick {
                deskewed_cloud: synced_data.cloud,
                cloud_header: synced_data.header,
                imu_odom: odom,
                //runtime_eval: synced_data.runtime_eval,
            };

            sender.send(res).expect("could not send lifis tick");
        }
    }
}

fn imu_to_lifis(msg: Imu) -> prelio::imu::ImuMsg {
    prelio::imu::ImuMsg {
        stamp: ros_pointcloud2::ros::TimeMsg {
            sec: msg.header.stamp.sec,
            nanosec: msg.header.stamp.nanosec,
        },
        frame_id: msg.header.frame_id,
        orientation: [
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
        ],
        orientation_covariance: [
            msg.orientation_covariance[0],
            msg.orientation_covariance[1],
            msg.orientation_covariance[2],
            msg.orientation_covariance[3],
            msg.orientation_covariance[4],
            msg.orientation_covariance[5],
            msg.orientation_covariance[6],
            msg.orientation_covariance[7],
            msg.orientation_covariance[8],
        ],
        angular_velocity: [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ],
        angular_velocity_covariance: [
            msg.angular_velocity_covariance[0],
            msg.angular_velocity_covariance[1],
            msg.angular_velocity_covariance[2],
            msg.angular_velocity_covariance[3],
            msg.angular_velocity_covariance[4],
            msg.angular_velocity_covariance[5],
            msg.angular_velocity_covariance[6],
            msg.angular_velocity_covariance[7],
            msg.angular_velocity_covariance[8],
        ],
        linear_acceleration: [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ],
        linear_acceleration_covariance: [
            msg.linear_acceleration_covariance[0],
            msg.linear_acceleration_covariance[1],
            msg.linear_acceleration_covariance[2],
            msg.linear_acceleration_covariance[3],
            msg.linear_acceleration_covariance[4],
            msg.linear_acceleration_covariance[5],
            msg.linear_acceleration_covariance[6],
            msg.linear_acceleration_covariance[7],
            msg.linear_acceleration_covariance[8],
        ],
    }
}

fn lifis_to_imu(msg: prelio::imu::ImuMsg) -> Imu {
    Imu {
        header: ros2_interfaces_jazzy::std_msgs::msg::Header {
            stamp: ros2_interfaces_jazzy::builtin_interfaces::msg::Time {
                sec: msg.stamp.sec,
                nanosec: msg.stamp.nanosec,
            },
            frame_id: msg.frame_id,
        },
        orientation: ros2_interfaces_jazzy::geometry_msgs::msg::Quaternion {
            x: msg.orientation[1],
            y: msg.orientation[2],
            z: msg.orientation[3],
            w: msg.orientation[0],
        },
        orientation_covariance: [
            msg.orientation_covariance[0],
            msg.orientation_covariance[1],
            msg.orientation_covariance[2],
            msg.orientation_covariance[3],
            msg.orientation_covariance[4],
            msg.orientation_covariance[5],
            msg.orientation_covariance[6],
            msg.orientation_covariance[7],
            msg.orientation_covariance[8],
        ],
        angular_velocity: ros2_interfaces_jazzy::geometry_msgs::msg::Vector3 {
            x: msg.angular_velocity[0],
            y: msg.angular_velocity[1],
            z: msg.angular_velocity[2],
        },
        angular_velocity_covariance: [
            msg.angular_velocity_covariance[0],
            msg.angular_velocity_covariance[1],
            msg.angular_velocity_covariance[2],
            msg.angular_velocity_covariance[3],
            msg.angular_velocity_covariance[4],
            msg.angular_velocity_covariance[5],
            msg.angular_velocity_covariance[6],
            msg.angular_velocity_covariance[7],
            msg.angular_velocity_covariance[8],
        ],
        linear_acceleration: ros2_interfaces_jazzy::geometry_msgs::msg::Vector3 {
            x: msg.linear_acceleration[0],
            y: msg.linear_acceleration[1],
            z: msg.linear_acceleration[2],
        },
        linear_acceleration_covariance: [
            msg.linear_acceleration_covariance[0],
            msg.linear_acceleration_covariance[1],
            msg.linear_acceleration_covariance[2],
            msg.linear_acceleration_covariance[3],
            msg.linear_acceleration_covariance[4],
            msg.linear_acceleration_covariance[5],
            msg.linear_acceleration_covariance[6],
            msg.linear_acceleration_covariance[7],
            msg.linear_acceleration_covariance[8],
        ],
    }
}
