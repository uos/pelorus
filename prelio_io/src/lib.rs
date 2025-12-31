#[cfg(feature = "ros2-client")]
pub mod tf;
#[cfg(feature = "ratpub")]
pub mod tf_lh;
#[cfg(feature = "r2r")]
pub mod tf_r2r;

#[allow(unused_imports)]
use futures_lite::{StreamExt, future};
#[allow(unused_imports)]
use log::{debug, error, info, warn};
use pelorus_core::{
    ImuMsg, ImuPredictor, ImuPredictorUndistort, LidarVendor, LifisNodeParameters, LifisParameters,
    LivoxMid360Point, LivoxPoint, OdomMsg, OusterPoint, PrelioProcessedCloud, RobosensePoint,
    VelodynePoint, timing::RuntimeEvaluator,
};
pub use prelio;

use anyhow::anyhow;
use prelio::timestamp_to_secs;

#[cfg(feature = "r2r")]
use r2r::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{
        Point, Pose, PoseWithCovariance, Quaternion, Twist, TwistWithCovariance, Vector3,
    },
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::{Imu, PointCloud2},
    std_msgs::msg::Header,
};
#[cfg(feature = "ratpub")]
use ratpub::Node;
use ros_pointcloud2::prelude::*;
#[cfg(feature = "ros2-client")]
use ros2_client::ros2;
#[cfg(feature = "ratpub")]
use ros2_interfaces_jazzy_rkyv::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{
        Point, Pose, PoseWithCovariance, Quaternion, Twist, TwistWithCovariance, Vector3,
    },
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::{Imu, PointCloud2},
    std_msgs::msg::Header,
};
#[cfg(feature = "ros2-client")]
use ros2_interfaces_jazzy_serde::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{
        Point, Pose, PoseWithCovariance, Quaternion, Twist, TwistWithCovariance, Vector3,
    },
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::{Imu, PointCloud2},
    std_msgs::msg::Header,
};
use std::sync::{Arc, Mutex};

#[cfg(feature = "ros2-client")]
impl_pointcloud2_for_ros2_interfaces_jazzy_serde!();
#[cfg(feature = "ratpub")]
impl_pointcloud2_for_ros2_interfaces_jazzy_rkyv!();
#[cfg(feature = "r2r")]
impl_pointcloud2_for_r2r!();

const N_NTH_SCAN: usize = 1; // TODO to config

pub struct State {
    pub global_polygon: Option<Vec<nalgebra::Point2<f32>>>,
    pub sensor_tf: Option<nalgebra::geometry::Isometry3<f32>>,
    pub imu_tf: Option<nalgebra::geometry::Isometry3<f64>>,

    pub lifis: Arc<Mutex<prelio::Prelio>>,
    pub lifis_params: LifisParameters,
    take_nth_scan: usize,
    pub lidar_vendor: LidarVendor,
    #[cfg(feature = "rerun")]
    pub rec: Arc<rerun::RecordingStream>,
}

impl State {
    pub fn new(
        config: &LifisParameters,
        imu_pose_prediction_undistort: Option<Box<ImuPredictorUndistort>>,
        imu_pose_prediction_init_guess: Option<Box<ImuPredictor>>,
        #[cfg(feature = "rerun")] rec: Arc<rerun::RecordingStream>,
        ready_for_next: Option<std::sync::mpsc::Receiver<()>>,
    ) -> anyhow::Result<Self> {
        Ok(State {
            #[cfg(feature = "rerun")]
            rec,
            global_polygon: None,
            sensor_tf: None,
            imu_tf: None,
            lifis: match config.lidar_vendor {
                LidarVendor::Velodyne => {
                    Arc::new(Mutex::new(prelio::Prelio::create::<6, VelodynePoint>(
                        config.imu.gravity,
                        config.imu.madgwick_setting,
                        &config.imu.calibration,
                        config, // TODO more specific
                        imu_pose_prediction_undistort,
                        imu_pose_prediction_init_guess,
                        Some(config.lidar_vendor),
                        ready_for_next,
                    )?))
                }
                LidarVendor::Livox => {
                    Arc::new(Mutex::new(prelio::Prelio::create::<5, LivoxPoint>(
                        config.imu.gravity,
                        config.imu.madgwick_setting,
                        &config.imu.calibration,
                        config, // TODO more specific
                        imu_pose_prediction_undistort,
                        imu_pose_prediction_init_guess,
                        Some(config.lidar_vendor),
                        ready_for_next,
                    )?))
                }
                LidarVendor::LivoxMid360 => {
                    Arc::new(Mutex::new(prelio::Prelio::create::<5, LivoxMid360Point>(
                        config.imu.gravity,
                        config.imu.madgwick_setting,
                        &config.imu.calibration,
                        config, // TODO more specific
                        imu_pose_prediction_undistort,
                        imu_pose_prediction_init_guess,
                        Some(config.lidar_vendor),
                        ready_for_next,
                    )?))
                }
                LidarVendor::Robosense => {
                    Arc::new(Mutex::new(prelio::Prelio::create::<6, RobosensePoint>(
                        config.imu.gravity,
                        config.imu.madgwick_setting,
                        &config.imu.calibration,
                        config, // TODO more specific
                        imu_pose_prediction_undistort,
                        imu_pose_prediction_init_guess,
                        Some(config.lidar_vendor),
                        ready_for_next,
                    )?))
                }
                LidarVendor::Ouster => {
                    Arc::new(Mutex::new(prelio::Prelio::create::<4, OusterPoint>(
                        config.imu.gravity,
                        config.imu.madgwick_setting,
                        &config.imu.calibration,
                        config, // TODO more specific
                        imu_pose_prediction_undistort,
                        imu_pose_prediction_init_guess,
                        Some(config.lidar_vendor),
                        ready_for_next,
                    )?))
                } // prelio::LidarVendor::Skip => {
                  //     Arc::new(Mutex::new(prelio::Prelio::create::<3, SkipPoint>(
                  //         config.imu.gravity,
                  //         config.imu.madgwick_setting,
                  //         &config.imu.calibration,
                  //         &config, // TODO more specific
                  //         imu_pose_prediction_undistort,
                  //         imu_pose_prediction_init_guess,
                  //         Some(config.lidar_vendor),
                  //         ready_for_next,
                  //     )?))
                  // }
            },
            lifis_params: config.clone(),
            take_nth_scan: N_NTH_SCAN,
            lidar_vendor: config.lidar_vendor,
        })
    }
}

fn cloud_callback(
    state: Arc<Mutex<State>>,
    #[cfg(feature = "ros2-client")] tf_listener: Arc<tf::TfListener>,
    #[cfg(feature = "r2r")] tf_listener: Arc<tf_r2r::TfListener>,
    #[cfg(feature = "ratpub")] tf_listener: Arc<tf_lh::TfListener>,
    #[cfg(feature = "ros2-client")] msg: PointCloud2,
    #[cfg(feature = "ratpub")] msg: PointCloud2,
    #[cfg(feature = "r2r")] msg: r2r::sensor_msgs::msg::PointCloud2,
    first_imu_tf: Option<String>,
) -> anyhow::Result<()> {
    let orig_header = msg.header.clone();
    #[cfg(feature = "ros2-client")]
    let in_msg = impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(msg);
    #[cfg(feature = "ratpub")]
    let in_msg = impl_ros2_interfaces_jazzy_rkyv::to_pointcloud2_msg(msg);
    #[cfg(feature = "r2r")]
    let in_msg = impl_r2r::to_pointcloud2_msg(msg);

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
                    state.sensor_tf,
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

                // dbg!(imu_tf);
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
                        return Err(anyhow!(
                            "global_zero_frame is given but map_frame is not. Can not update polygon."
                        ));
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

pub struct PrelioNode {
    pub state: Arc<std::sync::Mutex<State>>,
    #[cfg(feature = "ros2-client")]
    pub tf_listener: Arc<tf::TfListener>,
    #[cfg(feature = "ratpub")]
    pub tf_listener: Arc<tf_lh::TfListener>,
    #[cfg(feature = "r2r")]
    pub tf_listener: Arc<tf_r2r::TfListener>,
    pub logger: String,
}

#[cfg(feature = "ros2-client")]
impl From<crate::Qos> for ros2::QosPolicies {
    fn from(value: crate::Qos) -> Self {
        match value {
            Qos::Sensor(rel) => ros2::QosPolicyBuilder::new()
                .durability(ros2::policy::Durability::Volatile)
                .deadline(ros2::policy::Deadline(ros2::Duration::INFINITE))
                .ownership(ros2::policy::Ownership::Shared)
                .reliability(match rel {
                    Reliability::Reliable => ros2::policy::Reliability::Reliable {
                        max_blocking_time: ros2::Duration::INFINITE,
                    },
                    Reliability::BestEffort => ros2::policy::Reliability::BestEffort,
                })
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

#[derive(Debug)]
pub struct Tick {
    pub deskewed_cloud: Vec<PointXYZI>,
    pub map_only: bool,
    pub cloud_header: HeaderMsg,
    pub odom_time: TimeMsg,
    pub imu_odom: Odometry,
    pub runtime_eval: RuntimeEvaluator,
}

#[cfg(feature = "ros2-client")]
fn split_namespace_topic(input: &str) -> (String, String) {
    let parts: Vec<&str> = input.split('/').collect();

    if parts.is_empty() {
        // Handle empty string case
        (String::from(""), String::from(""))
    } else {
        let last_part = parts.last().unwrap(); // Safe because we checked is_empty()
        let mut namespace = String::new();

        for (i, &part) in parts.iter().enumerate() {
            if i < parts.len() - 1 {
                if !namespace.is_empty() {
                    namespace.push('/');
                }
                namespace.push_str(part);
            }
        }

        (namespace, "/".to_string() + last_part) // Add the leading slash to the topic
    }
}

impl PrelioNode {
    pub fn new(
        #[cfg(feature = "ros2-client")] node: &mut ros2_client::Node,
        #[cfg(feature = "r2r")] node: &mut r2r::Node,
        #[cfg(feature = "ratpub")] node: &Node,
        #[cfg(feature = "ratpub")] rt: Arc<tokio::runtime::Runtime>,
        params: &LifisParameters,
        node_params: &LifisNodeParameters,
        imu_pose_predictor_init: Option<Arc<Mutex<dyn FnMut(&[ImuMsg]) + Send + Sync>>>,
        imu_pose_prediction_undistort: Option<Box<ImuPredictorUndistort>>,
        imu_pose_prediction_init_guess: Option<Box<ImuPredictor>>,
        #[cfg(feature = "rerun")] rec: Arc<rerun::RecordingStream>,
        ready_for_next: Option<std::sync::mpsc::Receiver<()>>,
    ) -> anyhow::Result<(Self, std::sync::mpsc::Receiver<Tick>)> {
        #[cfg(feature = "ros2-client")]
        let tf_listener = Arc::new(tf::TfListener::new(node));

        #[cfg(feature = "r2r")]
        let tf_listener = Arc::new(tf_r2r::TfListener::new(node));

        #[cfg(feature = "ratpub")]
        let tf_listener = Arc::new(tf_lh::TfListener::new(node, rt.clone()));

        let state = State::new(
            params,
            imu_pose_prediction_undistort,
            imu_pose_prediction_init_guess,
            #[cfg(feature = "rerun")]
            Arc::clone(&rec),
            ready_for_next,
        )?;

        let chan = {
            let mut lifis = state.lifis.lock().unwrap();
            lifis.get_chan().unwrap()
        };

        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            Self::transform_types(tx, chan);
        });

        // first option None = override not set.
        let imu_tf_frame: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        #[cfg(feature = "ros2-client")]
        let cloud_subscriber = node
            .create_subscription::<PointCloud2>(
                &node
                    .create_topic(
                        &ros2_client::Name::new(
                            &split_namespace_topic(&node_params.cloud_src_topic).0,
                            &split_namespace_topic(&node_params.cloud_src_topic)
                                .1
                                .trim_start_matches("/"),
                        )
                        .unwrap(),
                        ros2_client::MessageTypeName::new("sensor_msgs", "PointCloud2"),
                        &node_params.cloud_qos.into(),
                    )
                    .unwrap(),
                Some(node_params.cloud_qos.into()),
            )
            .unwrap();

        #[cfg(feature = "r2r")]
        let cloud_subscriber = node
            .subscribe::<PointCloud2>(&node_params.cloud_src_topic, node_params.cloud_qos.into())?;

        #[cfg(feature = "ratpub")]
        let mut cloud_subscriber = rt.block_on(async {
            node.create_subscriber::<PointCloud2>(node_params.cloud_src_topic.clone(), 10)
                .await
                .unwrap()
        });

        #[cfg(feature = "r2r")]
        let wheel_subscriber =
            node.subscribe::<Odometry>(&node_params.wheel_src_topic, node_params.wheel_qos.into())?;

        #[cfg(feature = "ratpub")]
        let mut wheel_subscriber = rt.block_on(async {
            node.create_subscriber::<Odometry>(node_params.wheel_src_topic.clone(), 10)
                .await
                .unwrap()
        });

        // Share with cloud callback, TODO maybe the only one who needs it, so remove mutex and arc?
        let state = Arc::new(std::sync::Mutex::new(state));

        #[cfg(not(feature = "no-imu-sync"))]
        struct ImuInitState {
            start_time: Option<f64>,
            buffer: Option<Vec<ImuMsg>>,
        }
        #[cfg(not(feature = "no-imu-sync"))]
        let imu_init_state = Arc::new(Mutex::new(ImuInitState {
            start_time: None,
            buffer: Some(Vec::new()),
        }));

        #[cfg(any(feature = "ros2-client", feature = "ratpub", feature = "r2r"))]
        let tf_listener_clone = tf_listener.clone();
        #[cfg(any(feature = "ros2-client", feature = "ratpub", feature = "r2r"))]
        let state_clone = state.clone();
        #[cfg(any(feature = "ros2-client", feature = "ratpub", feature = "r2r"))]
        let imu_tf_frame_lidar_inner = Arc::clone(&imu_tf_frame);
        // #[cfg(all(not(feature = "no-imu-sync"), feature = "ratpub"))]
        // let lidar_imu_init_buffer = imu_init_buffer.clone();
        #[cfg(feature = "ros2-client")]
        std::thread::spawn(move || {
            loop {
                match cloud_subscriber.take() {
                    Err(e) => {
                        error!("Could not read cloud: {e}");
                    }
                    Ok(Some((msg, _msg_info))) => {
                        // error!("received cloud message");
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
            }
        });
        #[cfg(feature = "r2r")]
        std::thread::spawn(move || {
            future::block_on(async {
                cloud_subscriber
                    .for_each(|msg| {
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
                                error!("{:?}", e);
                            }
                        }
                    })
                    .await;
            })
        });

        #[cfg(feature = "ratpub")]
        {
            let wheel_sub_rt = Arc::clone(&rt);
            let state_wheel = state.clone();
            let ext = nalgebra::Isometry3::from_parts(
                nalgebra::Translation3::new(0.0, 0.0, 0.0),
                nalgebra::UnitQuaternion::from_matrix(&params.wheel_to_imu_rotation),
            );
            std::thread::spawn(move || {
                wheel_sub_rt.block_on(async {
                    while let Some(msg) = wheel_subscriber.next().await {
                        // Transform odom to IMU frame and send to prelio for sync'd processing

                        use pelorus_core::OdomMsg;
                        let v = nalgebra::Vector3::new(
                            msg.twist.twist.linear.x,
                            msg.twist.twist.linear.y,
                            msg.twist.twist.linear.z,
                        );
                        let t = ext.transform_vector(&v);

                        let odom_msg = OdomMsg {
                            twist_linear: t,
                            twist_angular: nalgebra::Vector3::new(
                                msg.twist.twist.angular.x,
                                msg.twist.twist.angular.y,
                                msg.twist.twist.angular.z,
                            ),
                            twist_cov: msg.twist.covariance,
                            stamp: prelio::TimeMsg {
                                sec: msg.header.stamp.sec,
                                nanosec: msg.header.stamp.nanosec,
                            },
                        };

                        // Send to prelio's sync worker for synchronized lidar-imu-odom processing
                        let state_lock = state_wheel.lock().unwrap();
                        state_lock.lifis.lock().unwrap().add_odom(odom_msg);
                    }
                })
            });
        }
        #[cfg(feature = "r2r")]
        {
            let state_wheel = state.clone();
            let ext = nalgebra::Isometry3::from_parts(
                nalgebra::Translation3::new(0.0, 0.0, 0.0),
                nalgebra::UnitQuaternion::from_matrix(&params.wheel_to_imu_rotation),
            );
            std::thread::spawn(move || {
                future::block_on(async {
                    wheel_subscriber
                        .for_each(|msg| {
                            let mut odom_msg = odom_to_lifis(msg);
                            odom_msg.twist_linear = ext.transform_vector(&odom_msg.twist_linear);
                            let state_lock = state_wheel.lock().unwrap();
                            state_lock.lifis.lock().unwrap().add_odom(odom_msg);
                        })
                        .await
                })
            });
        }
        // let state_clone = state.clone();

        #[cfg(feature = "ratpub")]
        {
            let cloud_sub_rt = Arc::clone(&rt);
            std::thread::spawn(move || {
                cloud_sub_rt.block_on(async {
                    while let Some(msg) = cloud_subscriber.next().await {
                        // #[cfg(not(feature = "no-imu-sync"))]
                        // {
                        //     let b = lidar_imu_init_buffer.lock().unwrap();
                        //     if b.is_some() {
                        //         continue;
                        //     }
                        // }
                        // debug!("received cloud");
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
                                error!("{:?}", e);
                            }
                        }
                    }
                })
            });
        }
        let state_clone = state.clone();
        #[cfg(feature = "ros2-client")]
        let imu_subscription = node
            .create_subscription::<Imu>(
                &node
                    .create_topic(
                        &ros2_client::Name::new(
                            &split_namespace_topic(&node_params.imu_src_topic).0,
                            &split_namespace_topic(&node_params.imu_src_topic)
                                .1
                                .trim_start_matches("/"),
                        )
                        .unwrap(),
                        ros2_client::MessageTypeName::new("sensor_msgs", "Imu"),
                        &node_params.imu_qos.into(),
                    )
                    .unwrap(),
                None,
            )
            .unwrap();

        #[cfg(all(feature = "r2r", not(feature = "no-imu-sync")))]
        let imu_subscription =
            node.subscribe::<Imu>(&node_params.imu_src_topic, node_params.imu_qos.into())?;

        // do not publish filtered imu if no topic is provided
        #[cfg(feature = "ros2-client")]
        let imu_filtered_publisher = if node_params.imu_filtered_topic == "" {
            None
        } else {
            Some(Arc::new(Mutex::new(
                node.create_publisher(
                    &node
                        .create_topic(
                            &ros2_client::Name::new(
                                &split_namespace_topic(&node_params.imu_filtered_topic).0,
                                &split_namespace_topic(&node_params.imu_filtered_topic)
                                    .1
                                    .trim_start_matches("/"),
                            )
                            .unwrap(),
                            ros2_client::MessageTypeName::new("sensor_msgs", "Imu"),
                            &node_params.imu_filtered_qos.into(),
                        )
                        .unwrap(),
                    None,
                )
                .unwrap(),
            )))
        };
        #[cfg(feature = "r2r")]
        let imu_filtered_publisher = if node_params.imu_filtered_topic == "" {
            None
        } else {
            Some(Arc::new(Mutex::new(node.create_publisher::<Imu>(
                &node_params.imu_filtered_topic,
                node_params.imu_filtered_qos.into(),
            )?)))
        };

        let imu_tf_frame_lidar_inner = Arc::clone(&imu_tf_frame);
        let init_imu_time = node_params.imu_predictor_init_time_sec;
        #[cfg(feature = "ros2-client")]
        std::thread::spawn(move || {
            // TODO use while let
            let imu_pose_predictor_init = imu_pose_predictor_init.clone();
            loop {
                match imu_subscription.take() {
                    Err(e) => {
                        error!("Could not read imu: {e}");
                    }
                    Ok(Some((msg, _msg_info))) => {
                        // debug!("received imu message");
                        imu_tf_frame_lidar_inner
                            .clone()
                            .lock()
                            .unwrap()
                            .get_or_insert(msg.header.frame_id.clone());
                        let imu_init_state = imu_init_state.clone();

                        let imu_lifis = imu_to_lifis(msg.clone());
                        let curr_stamp = timestamp_to_secs(&imu_lifis.stamp);

                        if let Some(imu_pose_predictor_init) = imu_pose_predictor_init.as_ref() {
                            let mut imu_pose_predictor_init =
                                imu_pose_predictor_init.lock().unwrap();
                            let mut init_state = imu_init_state.lock().unwrap();

                            if init_state.start_time.is_none() {
                                init_state.start_time = Some(curr_stamp);
                            }

                            if let Some(init_time) = init_state.start_time {
                                let delta = curr_stamp - init_time;
                                if delta < init_imu_time {
                                    if let Some(ref mut buffer) = init_state.buffer {
                                        buffer.push(imu_lifis);
                                        continue; // skip adding to lifis if we are still in the init phase
                                    }
                                    // we are only here, if there was an initial buffer, but it was already consumed so skip init and continue with normal processing
                                } else {
                                    // apply the init buffer
                                    if let Some(buffer) = init_state.buffer.take() {
                                        imu_pose_predictor_init(&buffer);
                                        info!("EKF: Imu initialized");
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
            }
        });

        #[cfg(feature = "ratpub")]
        {
            let imu_sub_rt = Arc::clone(&rt);
            let mut imu_sub = imu_sub_rt.block_on(async {
                node.create_subscriber::<Imu>(node_params.imu_src_topic.clone(), 100)
                    .await
                    .unwrap()
            });

            let imu_imu_init_state = imu_init_state.clone();
            std::thread::spawn(move || {
                let imu_pose_predictor_init = imu_pose_predictor_init.clone();
                imu_sub_rt.block_on(async {
                    while let Some(msg) = imu_sub.next().await {
                        // debug!("received imu");
                        let state = state_clone.lock().unwrap();
                        {
                            imu_tf_frame_lidar_inner
                                .clone()
                                .lock()
                                .unwrap()
                                .get_or_insert(msg.header.frame_id.clone());
                        }
                        let imu_init_state = imu_imu_init_state.clone();

                        let imu_lifis = imu_to_lifis(msg.clone());
                        let curr_stamp = timestamp_to_secs(&imu_lifis.stamp);

                        if let Some(imu_pose_predictor_init) = imu_pose_predictor_init.as_ref() {
                            let mut imu_pose_predictor_init =
                                imu_pose_predictor_init.lock().unwrap();
                            let mut init_state = imu_init_state.lock().unwrap();

                            if init_state.start_time.is_none() {
                                init_state.start_time = Some(curr_stamp);
                            }

                            if let Some(init_time) = init_state.start_time {
                                let delta = curr_stamp - init_time;
                                if delta < init_imu_time {
                                    if let Some(ref mut buffer) = init_state.buffer {
                                        buffer.push(imu_lifis);
                                        continue; // skip adding to lifis if we are still in the init phase
                                    }
                                    // we are only here, if there was an initial buffer, but it was already consumed so skip init and continue with normal processing
                                } else {
                                    // apply the init buffer
                                    if let Some(buffer) = init_state.buffer.take() {
                                        imu_pose_predictor_init(&buffer);
                                        info!("EKF: Imu initialized");
                                    }
                                }
                            }
                        }

                        // outputs filtered imu msg to pub again but we can not right now
                        let _ = state.lifis.lock().unwrap().add_imu(imu_to_lifis(msg));

                        // let filtered = {
                        //     let state = state_clone.lock().unwrap();
                        //     let mut lifis = state.lifis.lock().unwrap();
                        //     lifis.add_imu(imu_to_lifis(msg))
                        // };

                        // TODO can not pub filtered because imu_init_fn escapes its lifetime
                        // match (filtered, wants_to_pub_imu_filtered) {
                        //     (Some(filtered), true) => {
                        //         // node.publish(&imu_filtered_topic, &lifis_to_imu(filtered))
                        //         //     .await
                        //         //     .expect("could not publish filtered imu");
                        //     }
                        //     _ => {}
                        // }
                    }
                });
            });
        }

        // TODO inline with upper because logic is same but conditionals around block on and err handling
        #[cfg(all(feature = "r2r", not(feature = "no-imu-sync")))]
        std::thread::spawn(move || {
            let imu_pose_predictor_init = imu_pose_predictor_init.clone();
            future::block_on(async {
                imu_subscription
                    .for_each(|msg| {
                        let state = state_clone.lock().unwrap();
                        imu_tf_frame_lidar_inner
                            .clone()
                            .lock()
                            .unwrap()
                            .get_or_insert(msg.header.frame_id.clone());
                        let imu_init_state = imu_init_state.clone();

                        let imu_lifis = imu_to_lifis(msg.clone());
                        let curr_stamp = timestamp_to_secs(&imu_lifis.stamp);

                        if let Some(imu_pose_predictor_init) = imu_pose_predictor_init.as_ref() {
                            let mut imu_pose_predictor_init =
                                imu_pose_predictor_init.lock().unwrap();
                            let mut init_state = imu_init_state.lock().unwrap();

                            if init_state.start_time.is_none() {
                                init_state.start_time = Some(curr_stamp);
                            }

                            if let Some(init_time) = init_state.start_time {
                                let delta = curr_stamp - init_time;
                                if delta < init_imu_time {
                                    if let Some(ref mut buffer) = init_state.buffer {
                                        buffer.push(imu_lifis);
                                        return; // skip adding to lifis if we are still in the init phase
                                    }
                                    // we are only here, if there was an initial buffer, but it was already consumed so skip init and continue with normal processing
                                } else {
                                    // apply the init buffer
                                    if let Some(buffer) = init_state.buffer.take() {
                                        imu_pose_predictor_init(&buffer);
                                        info!("EKF: Imu initialized");
                                    }
                                }
                            }
                        }

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
                    })
                    .await;
            })
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

    /// Translates ROS types to Prelio Types
    pub fn transform_types(
        sender: std::sync::mpsc::Sender<Tick>,
        receiver: std::sync::mpsc::Receiver<PrelioProcessedCloud>,
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
                #[cfg(not(feature = "r2r"))]
                covariance: synced_data.pose_covariance,
                #[cfg(feature = "r2r")]
                covariance: synced_data.pose_covariance.to_vec(),
            };

            let twist = TwistWithCovariance {
                twist: Twist {
                    linear: Vector3 {
                        x: synced_data.end_of_frame_twist.linear.x,
                        y: synced_data.end_of_frame_twist.linear.y,
                        z: synced_data.end_of_frame_twist.linear.z,
                    },
                    angular: Vector3 {
                        x: synced_data.end_of_frame_twist.angular.x,
                        y: synced_data.end_of_frame_twist.angular.y,
                        z: synced_data.end_of_frame_twist.angular.z,
                    },
                },
                #[cfg(not(feature = "r2r"))]
                covariance: synced_data.twist_covariance,
                #[cfg(feature = "r2r")]
                covariance: synced_data.twist_covariance.to_vec(),
            };

            let odom = Odometry {
                header: Header {
                    stamp: Time {
                        sec: synced_data.end_of_frame_time.sec,
                        nanosec: synced_data.end_of_frame_time.nanosec,
                    },
                    frame_id: synced_data.header.frame_id.clone(),
                },
                child_frame_id: synced_data.imu_frame,
                pose,
                twist,
            };

            let res = Tick {
                map_only: synced_data.should_map_only,
                deskewed_cloud: synced_data.cloud,
                cloud_header: synced_data.header,
                imu_odom: odom,
                runtime_eval: synced_data.runtime_eval,
                odom_time: synced_data.lidar_end_time_abs,
            };

            sender.send(res).expect("could not send lifis tick");
        }
    }
}

// only used with r2r
#[allow(dead_code)]
fn odom_to_lifis(msg: Odometry) -> OdomMsg {
    let odom_msg = OdomMsg {
        twist_linear: nalgebra::Vector3::new(
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ),
        twist_angular: nalgebra::Vector3::new(
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ),
        twist_cov: [
            msg.twist.covariance[0],
            msg.twist.covariance[1],
            msg.twist.covariance[2],
            msg.twist.covariance[3],
            msg.twist.covariance[4],
            msg.twist.covariance[5],
            msg.twist.covariance[6],
            msg.twist.covariance[7],
            msg.twist.covariance[8],
            msg.twist.covariance[9],
            msg.twist.covariance[10],
            msg.twist.covariance[11],
            msg.twist.covariance[12],
            msg.twist.covariance[13],
            msg.twist.covariance[14],
            msg.twist.covariance[15],
            msg.twist.covariance[16],
            msg.twist.covariance[17],
            msg.twist.covariance[18],
            msg.twist.covariance[19],
            msg.twist.covariance[20],
            msg.twist.covariance[21],
            msg.twist.covariance[22],
            msg.twist.covariance[23],
            msg.twist.covariance[24],
            msg.twist.covariance[25],
            msg.twist.covariance[26],
            msg.twist.covariance[27],
            msg.twist.covariance[28],
            msg.twist.covariance[29],
            msg.twist.covariance[30],
            msg.twist.covariance[31],
            msg.twist.covariance[32],
            msg.twist.covariance[33],
            msg.twist.covariance[34],
            msg.twist.covariance[35],
        ],
        stamp: ros_pointcloud2::ros::TimeMsg {
            sec: msg.header.stamp.sec,
            nanosec: msg.header.stamp.nanosec,
        },
    };
    odom_msg
}

fn imu_to_lifis(msg: Imu) -> ImuMsg {
    ImuMsg {
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

#[allow(dead_code)] // only used with ROS
fn lifis_to_imu(msg: ImuMsg) -> Imu {
    Imu {
        header: Header {
            stamp: Time {
                sec: msg.stamp.sec,
                nanosec: msg.stamp.nanosec,
            },
            frame_id: msg.frame_id,
        },
        orientation: Quaternion {
            x: msg.orientation[1],
            y: msg.orientation[2],
            z: msg.orientation[3],
            w: msg.orientation[0],
        },
        #[cfg(feature = "ros2-client")]
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
        #[cfg(feature = "ratpub")]
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
        #[cfg(feature = "r2r")]
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
        ]
        .to_vec(),
        angular_velocity: Vector3 {
            x: msg.angular_velocity[0],
            y: msg.angular_velocity[1],
            z: msg.angular_velocity[2],
        },
        #[cfg(feature = "ros2-client")]
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
        #[cfg(feature = "ratpub")]
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
        #[cfg(feature = "r2r")]
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
        ]
        .to_vec(),
        linear_acceleration: Vector3 {
            x: msg.linear_acceleration[0],
            y: msg.linear_acceleration[1],
            z: msg.linear_acceleration[2],
        },
        #[cfg(feature = "ros2-client")]
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
        #[cfg(feature = "ratpub")]
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
        #[cfg(feature = "r2r")]
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
        ]
        .to_vec(),
    }
}
