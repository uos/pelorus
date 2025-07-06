#[allow(unused_imports)]
use anyhow::Error;
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[global_allocator]
#[cfg(feature = "mimalloc")]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{command, Parser, Subcommand};
use futures_lite::future;
use log::{error, info};
use nalgebra::{Isometry3, Translation3};
#[cfg(feature = "r2r")]
use r2r::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{Point, PoseStamped, Quaternion, Transform, TransformStamped, Vector3},
    nav_msgs::msg::{Odometry, Path},
    sensor_msgs::msg::PointCloud2,
    std_msgs::msg::Header,
    visualization_msgs::msg::Marker,
    QosProfile,
};
#[cfg(feature = "ratpub")]
use ratpub::Node;
#[cfg(feature = "rerun")]
use rerun::external::glam;
#[cfg(feature = "ros2-client")]
use ros2_client::{NodeName, NodeOptions};
#[cfg(feature = "ros2-client")]
use ros2_interfaces_jazzy::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{Point, PoseStamped, Quaternion, Transform, TransformStamped, Vector3},
    nav_msgs::msg::Path,
    sensor_msgs::msg::PointCloud2,
    std_msgs::msg::Header,
    visualization_msgs::msg::Marker,
};
#[cfg(feature = "ratpub")]
#[allow(unused_imports)]
use ros2_interfaces_jazzy_rkyv::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{Point, PoseStamped, Quaternion, Transform, TransformStamped, Vector3},
    nav_msgs::msg::{Odometry, Path},
    sensor_msgs::msg::PointCloud2,
    std_msgs::msg::Header,
    visualization_msgs::msg::Marker,
};

use serde::{Deserialize, Serialize};
use std::io::Read;

use either::*;
#[cfg(feature = "ros2-client")]
use prelio_ros2::tf::TfBroadcaster;
#[cfg(feature = "ratpub")]
use prelio_ros2::tf_lh::TfBroadcaster;
#[cfg(feature = "r2r")]
use prelio_ros2::tf_r2r::TfBroadcaster;
use prelio_ros2::{prelio::config::LifisParameters, LifisNodeParameters, Qos, Reliability};

use pelorus_reg::{sfasterlio::SFasterLio, Registration};
use prelio_ros2::{prelio::imu::ImuMsg, PrelioNode};
use ros_pointcloud2::prelude::*;
//use small_gicp_sys::*;
use std::sync::{Arc, Mutex};

pub fn timestamp_to_secs(t: &TimeMsg) -> f64 {
    t.sec as f64 + t.nanosec as f64 / 1_000_000_000.0
}

pub fn rpcl2_time_to_r2r_time(t: &ros_pointcloud2::ros::TimeMsg) -> TimeMsg {
    TimeMsg {
        sec: t.sec,
        nanosec: t.nanosec,
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum LioMethod {
    SmallGICPCPU,

    #[default]
    SFasterLioCPU,

    JlioGPU,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LioConfig {
    pub feedback_fps: u32,
    pub registered_cloud_qos: prelio_ros2::Qos,
    pub odom_qos: prelio_ros2::Qos,
    pub imu_odom_qos: prelio_ros2::Qos,
    pub imu_path_qos: prelio_ros2::Qos,
    pub path_qos: prelio_ros2::Qos,
    pub map_resolution: f64,
    pub map_capacity: usize,
    pub imu_params: pelorus_reg::imu::ImuParameters,
    pub imu_init_length: u32,
    pub lidar_odom_cov: f64,
    pub enable_live_extrinsic_calibration: bool,
    pub prelio_params: LifisParameters,
    pub prelio_node_params: LifisNodeParameters,
    pub k_neighbors: usize,
    pub neighbor_max_range: f64,
    pub ekf_max_iterations: i8,
    pub dist_update_sensitivity: f32,
}

impl Default for LioConfig {
    fn default() -> Self {
        Self {
            map_resolution: 0.5,
            lidar_odom_cov: 0.001,
            imu_init_length: 10,
            feedback_fps: 1,
            enable_live_extrinsic_calibration: false,
            registered_cloud_qos: Qos::Sensor(Reliability::Reliable),
            odom_qos: Qos::default(),
            imu_odom_qos: Qos::default(),
            imu_path_qos: Qos::default(),
            path_qos: Qos::default(),
            imu_params: pelorus_reg::imu::ImuParameters::default(),
            prelio_params: LifisParameters::default(),
            prelio_node_params: LifisNodeParameters::default(),
            k_neighbors: 5,
            neighbor_max_range: 5.0,
            ekf_max_iterations: 4,
            dist_update_sensitivity: 60.0,
            map_capacity: 30_000,
        }
    }
}

#[allow(dead_code)]
fn lerp_color(c1: (f32, f32, f32), c2: (f32, f32, f32), t: f32) -> (f32, f32, f32) {
    (
        c1.0 * (1.0 - t) + c2.0 * t,
        c1.1 * (1.0 - t) + c2.1 * t,
        c1.2 * (1.0 - t) + c2.2 * t,
    )
}

fn imu_msg_to_pelorus_reg(m: &ImuMsg) -> pelorus_reg::ImuMsg {
    pelorus_reg::ImuMsg {
        timestamp_sec: prelio_ros2::prelio::timestamp_to_secs(&m.stamp),
        orientation: nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            m.orientation[0],
            m.orientation[1],
            m.orientation[2],
            m.orientation[3],
        )),
        angular_velocity: nalgebra::Vector3::new(
            m.angular_velocity[0],
            m.angular_velocity[1],
            m.angular_velocity[2],
        ),
        linear_acceleration: nalgebra::Vector3::new(
            m.linear_acceleration[0],
            m.linear_acceleration[1],
            m.linear_acceleration[2],
        ),
        orientation_covariance: m.orientation_covariance,
        angular_velocity_covariance: m.angular_velocity_covariance,
        linear_acceleration_covariance: m.linear_acceleration_covariance,
    }
}

fn read_config<T>(file_path: &str) -> Result<T, Box<dyn std::error::Error>>
where
    T: for<'a> Deserialize<'a>,
{
    let mut file = std::fs::File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let config: T = toml::from_str(&contents)?;

    Ok(config)
}

#[derive(Debug, Clone, Parser)]
pub struct ConfigArgs {
    #[arg(short, long, default_value = "lio.toml")]
    pub config_path: String,
}

#[derive(Debug, Clone, Parser)]
pub enum ConfigCommand {
    Init(ConfigArgs),
    Test(ConfigArgs),
}

#[derive(Debug, Clone, Subcommand)]
#[command()]
pub enum Command {
    Run(ConfigArgs),

    //#[command(visible_alias = "w")]
    #[command(subcommand)]
    Config(ConfigCommand),
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    pub command: Command,
}

fn lio(config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // TODO add logging file
    // log4rs::init_file("examples/async_listener/log4rs.yaml", Default::default()).unwrap();
    #[cfg(any(feature = "ratpub", feature = "rat"))]
    let rt = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap(),
    );

    #[cfg(feature = "rat")]
    rat::init("pelorus", None, Some(Arc::clone(&rt))).unwrap();

    #[cfg(not(feature = "ratpub"))]
    #[allow(unused_variables)]
    let namespace = "/lio";
    #[cfg(feature = "ros2-client")]
    let ctx = ros2_client::Context::new()?;
    #[cfg(feature = "ros2-client")]
    let mut node = ctx.new_node(
        NodeName::new(namespace, "lio_node")?,
        NodeOptions::new().enable_rosout(true),
    )?;
    #[cfg(feature = "ratpub")]
    let node =
        Arc::new(rt.block_on(async { Node::create("pelorus_node".to_owned()).await.unwrap() }));
    #[cfg(feature = "r2r")]
    let ctx = r2r::Context::create()?;
    #[cfg(feature = "r2r")]
    let mut node = r2r::Node::create(ctx, "lio", "lio")?;

    // TODO add config path via CLI (clap)
    // also make option to export default config to file
    let config = read_config(config_path);
    let config = match config {
        Err(e) => {
            if let Some(io_err) = e.downcast_ref::<std::io::Error>() {
                if io_err.kind() == std::io::ErrorKind::NotFound {
                    LioConfig::default()
                } else {
                    return Err(e);
                }
            } else {
                return Err(e);
            }
        }
        Ok(cfg) => cfg,
    };

    #[cfg(not(feature = "ratpub"))]
    let tf_broadcaster = TfBroadcaster::new(&mut node);
    #[cfg(feature = "ratpub")]
    let tf_broadcaster = Arc::new(TfBroadcaster::new(&node, rt.clone()));

    #[cfg(feature = "gicp")]
    let map = Arc::new(Mutex::new(small_gicp_sys::Map::new(config.map_resolution)));
    #[cfg(not(feature = "gicp"))]
    let map = Arc::new(Mutex::new(SFasterLio::new(
        config.map_resolution,
        config.map_capacity,
        config.prelio_params.imu.gravity,
        &config.imu_params,
        config.lidar_odom_cov,
        config.ekf_max_iterations,
        config.k_neighbors,
        config.neighbor_max_range,
        config.enable_live_extrinsic_calibration,
        Isometry3::identity(), // TODO params
        pelorus_reg::IVoxNearbyType::Nearby26,
        config.dist_update_sensitivity,
    )));

    #[cfg(not(feature = "gicp"))]
    let mean_acc = Arc::new(Mutex::new(None));

    #[cfg(feature = "rerun")]
    let rec = Arc::new(rerun::RecordingStreamBuilder::new("pelorus").spawn()?);

    let lidar_vendor = config.prelio_params.lidar_vendor.clone();

    let (ready_for_next_tx, ready_for_next_rx) = std::sync::mpsc::channel();
    let (_lifis_node, synced_rx) = PrelioNode::new(
        #[cfg(not(feature = "ratpub"))]
        &mut node,
        #[cfg(feature = "ratpub")]
        &node,
        #[cfg(feature = "ratpub")]
        rt.clone(),
        &config.prelio_params,
        &config.prelio_node_params,
        #[cfg(not(feature = "gicp"))]
        Some({
            let map = map.clone();
            let mean_acc = mean_acc.clone();
            Arc::new(Mutex::new(move |msg: &[ImuMsg]| {
                let msg: Vec<pelorus_reg::ImuMsg> =
                    msg.iter().map(|m| imu_msg_to_pelorus_reg(m)).collect();
                let mut map = map.lock().unwrap();
                let init_mean_acc = map.init_imu_ekf(&msg);
                let mut mean_acc = mean_acc.lock().expect("only user");
                *mean_acc = Some(init_mean_acc);
            }))
        }),
        #[cfg(feature = "gicp")]
        None,
        #[cfg(not(feature = "gicp"))]
        {
            let map = map.clone();

            Some(Box::new(
                move |imu_msgs: &[prelio_ros2::prelio::imu::ImuMsg],
                      last_cloud_end_time,
                      cloud_begin_time,
                      cloud_end_time,
                      cloud| {
                    debug_assert!(!imu_msgs.is_empty());
                    debug_assert!(last_cloud_end_time > 0.0);
                    debug_assert!(cloud_begin_time > 0.0);
                    debug_assert!(cloud_end_time > 0.0);

                    // TODO why even use different ImuMsg types?
                    let imus = imu_msgs.iter().map(|m| imu_msg_to_pelorus_reg(m));
                    let imus_vec = imus.collect::<Vec<_>>();

                    let mut map = map.lock().unwrap();
                    let cloud_undistorted = map.undistort(
                        &imus_vec,
                        last_cloud_end_time,
                        cloud_begin_time,
                        cloud_end_time,
                        cloud,
                        Some(lidar_vendor),
                    );

                    let pose = map.get_state().pose.clone();

                    Right((pose, cloud_undistorted))
                },
            ))
        },
        #[cfg(feature = "gicp")]
        None,
        #[cfg(not(feature = "gicp"))]
        {
            let map = map.clone();
            Some(Box::new(
                move |_imu_msgs: &[prelio_ros2::prelio::imu::ImuMsg],
                      _last_cloud_end_time,
                      _cloud_begin_time,
                      _cloud_end_time| {
                    let ekfstate = map.lock().unwrap().get_state();
                    prelio_ros2::prelio::ImuPosePrediction {
                        pose: ekfstate.pose,
                        linear_velocity: ekfstate.linear_velocity,
                        angular_velocity: nalgebra::Vector3::default(),
                        lidar_imu_extrinsic_rotation: nalgebra::Matrix3::identity(),
                        lidar_imu_extrinsic_translation: nalgebra::Vector3::default(),
                    }
                },
            ))
        },
        #[cfg(feature = "gicp")]
        None,
        #[cfg(feature = "rerun")]
        Arc::clone(&rec),
        #[cfg(feature = "gicp")]
        None,
        #[cfg(not(feature = "gicp"))]
        Some(ready_for_next_rx),
    )?;

    #[cfg(all(not(feature = "no-viz"), feature = "ros2-client"))]
    let map_publisher = node
        .create_publisher::<PointCloud2>(
            &node
                .create_topic(
                    &ros2_client::Name::new(&namespace, "map").unwrap(),
                    ros2_client::MessageTypeName::new("sensor_msgs", "PointCloud2"),
                    &Qos::default().into(),
                )
                .unwrap(),
            None,
        )
        .unwrap();
    #[cfg(all(not(feature = "no-viz"), feature = "ros2-client"))]
    let cloud_publisher = node
        .create_publisher(
            &node
                .create_topic(
                    &ros2_client::Name::new(&namespace, "registered").unwrap(),
                    ros2_client::MessageTypeName::new("sensor_msgs", "PointCloud2"),
                    &config.registered_cloud_qos.into(),
                )
                .unwrap(),
            None,
        )
        .unwrap();
    #[cfg(feature = "ros2-client")]
    let odom_publisher = node
        .create_publisher(
            &node
                .create_topic(
                    &ros2_client::Name::new(&namespace, "odom").unwrap(),
                    ros2_client::MessageTypeName::new("nav_msgs", "Odometry"),
                    &config.odom_qos.into(),
                )
                .unwrap(),
            None,
        )
        .unwrap();
    #[cfg(all(not(feature = "no-viz"), feature = "ros2-client"))]
    let lidar_path_publisher = node
        .create_publisher(
            &node
                .create_topic(
                    &ros2_client::Name::new(&namespace, "path").unwrap(),
                    ros2_client::MessageTypeName::new("nav_msgs", "Path"),
                    &config.path_qos.into(),
                )
                .unwrap(),
            None,
        )
        .unwrap();
    #[cfg(all(not(feature = "no-viz"), feature = "ros2-client"))]
    let imu_odom_publisher = node
        .create_publisher(
            &node
                .create_topic(
                    &ros2_client::Name::new(&namespace, "imu_odom").unwrap(),
                    ros2_client::MessageTypeName::new("nav_msgs", "Odometry"),
                    &config.imu_odom_qos.into(),
                )
                .unwrap(),
            None,
        )
        .unwrap();

    #[cfg(all(not(feature = "no-viz"), feature = "ros2-client"))]
    let imu_path_publisher = node
        .create_publisher(
            &node
                .create_topic(
                    &ros2_client::Name::new(&namespace, "imu_path").unwrap(),
                    ros2_client::MessageTypeName::new("nav_msgs", "Path"),
                    &config.imu_path_qos.into(),
                )
                .unwrap(),
            None,
        )
        .unwrap();
    #[cfg(all(
        not(feature = "no-viz"),
        feature = "ros2-client",
        not(feature = "rerun")
    ))]
    let registered_visual_publisher = node
        .create_publisher::<Marker>(
            &node
                .create_topic(
                    &ros2_client::Name::new(&format!("{namespace}/viz"), "registered").unwrap(),
                    ros2_client::MessageTypeName::new("visualization_msgs", "Marker"),
                    &ros2_client::DEFAULT_PUBLISHER_QOS,
                )
                .unwrap(),
            None,
        )
        .unwrap();

    #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    let map_publisher = node.create_publisher::<PointCloud2>("map", QosProfile::default())?;
    // #[cfg(feature = "ratpub")]
    // let map_publisher = rt.block_on(async {
    //     node.create_publisher::<PointCloud2>("map".to_owned())
    //         .await
    //         .unwrap()
    // });
    #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    let cloud_publisher =
        node.create_publisher::<PointCloud2>("registered", config.registered_cloud_qos.into())?;
    #[cfg(feature = "ratpub")]
    let cloud_publisher = rt.block_on(async {
        node.create_publisher::<PointCloud2>("registered".to_owned())
            .await
            .unwrap()
    });
    #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    let odom_publisher = node.create_publisher::<Odometry>("odom", config.odom_qos.into())?;
    #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
    let odom_publisher = rt.block_on(async {
        node.create_publisher::<Odometry>("odom".to_owned())
            .await
            .unwrap()
    });
    #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    let lidar_path_publisher = node.create_publisher::<Path>("path", config.path_qos.into())?;
    #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
    let lidar_path_publisher = rt.block_on(async {
        node.create_publisher::<Path>("path".to_owned())
            .await
            .unwrap()
    });
    #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    let imu_odom_publisher =
        node.create_publisher::<Odometry>("imu_odom", config.imu_odom_qos.into())?;
    #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
    let imu_odom_publisher = rt.block_on(async {
        node.create_publisher::<Odometry>("imu_odom".to_owned())
            .await
            .unwrap()
    });
    #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    let imu_path_publisher =
        node.create_publisher::<Path>("imu_path", config.imu_path_qos.into())?;
    #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
    let imu_path_publisher = rt.block_on(async {
        node.create_publisher::<Path>("imu_path".to_owned())
            .await
            .unwrap()
    });
    #[cfg(all(not(feature = "no-viz"), feature = "r2r", not(feature = "rerun")))]
    let registered_visual_publisher =
        node.create_publisher::<Marker>("registered_viz", r2r::QosProfile::default())?;

    let mut t_world_lidar = nalgebra::Isometry3::identity();
    let mut t_last_current = nalgebra::Isometry3::identity();

    let mut lidar_path = Path {
        header: Default::default(),
        poses: vec![],
    };

    lidar_path.header.frame_id = config.prelio_params.map_frame.clone();

    let mut imu_path = Path {
        header: Default::default(),
        poses: vec![],
    };

    imu_path.header.frame_id = config.prelio_params.map_frame.clone();

    let mut last_scan_time = None;
    // let mut total_viz_marker = 0;

    //let mut avg_iterations = 0.0;
    //let mut iteration_updates = 0;

    let path_update_translate_threshold = 0.0; // default 0.15 meters - TODO config

    let mut eval = prelio_ros2::prelio::timing::RuntimeEvaluator::new();

    let render_interval = if config.feedback_fps == 0 {
        None
    } else {
        Some(std::time::Duration::from_secs_f64(
            1.0 / config.feedback_fps as f64,
        ))
    };
    let mut last_render = std::time::Instant::now();

    std::thread::spawn(move || {
        #[cfg(feature = "gicp")]
        let mut first_scan = true;

        // Initialize Prelio prediction
        ready_for_next_tx.send(()).unwrap();

        while let Ok(lifis_tick) = synced_rx.recv() {
            eval.merge(&lifis_tick.runtime_eval);

            let cloud = lifis_tick.deskewed_cloud;
            let orig_frame = lifis_tick.cloud_header.frame_id.clone();
            let orig_time = lifis_tick.cloud_header.stamp.clone();
            let current_cloud_time = timestamp_to_secs(&orig_time);
            if let Some(existing_last_scan_time) = last_scan_time {
                if current_cloud_time < existing_last_scan_time {
                    let mut map = map.lock().unwrap();
                    #[cfg(not(feature = "gicp"))]
                    map.clear_localmap();
                    #[cfg(feature = "gicp")]
                    map.reset(config.map_resolution);

                    info!("Resetting map");
                    t_world_lidar = nalgebra::Isometry3::identity();
                    t_last_current = nalgebra::Isometry3::identity();
                }
            }

            last_scan_time = Some(timestamp_to_secs(&orig_time));

            // use the last difference if there is no odom from lifis
            let init_guess = if lifis_tick.imu_odom.pose.pose.position.x == 0.0
                && lifis_tick.imu_odom.pose.pose.position.y == 0.0
                && lifis_tick.imu_odom.pose.pose.position.z == 0.0
            {
                // lifis did not return a orientation (for example when not using IMU at all), use the last difference as guess
                if lifis_tick.imu_odom.pose.pose.orientation.w == 1.0
                    && lifis_tick.imu_odom.pose.pose.orientation.x == 0.0
                    && lifis_tick.imu_odom.pose.pose.orientation.y == 0.0
                    && lifis_tick.imu_odom.pose.pose.orientation.z == 0.0
                {
                    t_last_current
                } else {
                    // there is an orientation calculated by lifis for undeskewing the cloud, so we can use that
                    let initial_guess_translation = t_last_current.translation.clone();
                    let initial_guess_rotation =
                        nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                            lifis_tick.imu_odom.pose.pose.orientation.w,
                            lifis_tick.imu_odom.pose.pose.orientation.x,
                            lifis_tick.imu_odom.pose.pose.orientation.y,
                            lifis_tick.imu_odom.pose.pose.orientation.z,
                        ));
                    nalgebra::Isometry3::from_parts(
                        initial_guess_translation,
                        initial_guess_rotation,
                    )
                }
            } else {
                let initial_guess_translation = Translation3::new(
                    lifis_tick.imu_odom.pose.pose.position.x,
                    lifis_tick.imu_odom.pose.pose.position.y,
                    lifis_tick.imu_odom.pose.pose.position.z,
                );

                let initial_guess_rotation =
                    nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                        lifis_tick.imu_odom.pose.pose.orientation.w,
                        lifis_tick.imu_odom.pose.pose.orientation.x,
                        lifis_tick.imu_odom.pose.pose.orientation.y,
                        lifis_tick.imu_odom.pose.pose.orientation.z,
                    ));

                nalgebra::Isometry3::from_parts(initial_guess_translation, initial_guess_rotation)
            };

            let init_guess_world = t_world_lidar * init_guess;

            let register_pose = {
                let mut map = map.lock().unwrap();
                #[cfg(not(feature = "gicp"))]
                match map.register_and_map(&cloud, Some(&ready_for_next_tx), &mut eval) {
                    Ok(pose) => pose,
                    Err(err) => {
                        error!("{err}");
                        continue;
                    }
                }
                #[cfg(feature = "gicp")]
                {
                    if first_scan {
                        eval.start("map update").unwrap();
                        map.add(&cloud, None);
                        eval.stop("map update").unwrap();
                        first_scan = false;

                        CloudRegisterResult {
                            pose: Isometry3::identity(),
                            avg_iterations: None,
                            registered_cloud: None,
                        }
                    } else {
                        eval.start("reg").unwrap();
                        let res = map.register(&cloud, &init_guess_world);
                        eval.stop("reg").unwrap();
                        eval.start("map update").unwrap();
                        map.add(&cloud, Some(res.transform_target_to_source));
                        eval.stop("map update").unwrap();

                        CloudRegisterResult {
                            pose: res.transform_target_to_source,
                            avg_iterations: Some(res.num_iterations as u32),
                            registered_cloud: None,
                        }
                    }
                }
            };

            log::debug!("got a new registration");

            // Transformation from the last pose to the current pose in the local frame.
            t_last_current = t_world_lidar.inverse() * register_pose.pose;
            t_world_lidar = register_pose.pose;

            let mut lidar_odom = lifis_tick.imu_odom.clone();
            lidar_odom.header.stamp = Time {
                sec: orig_time.sec,
                nanosec: orig_time.nanosec,
            };
            lidar_odom.header.frame_id = config.prelio_params.map_frame.clone();
            lidar_odom.child_frame_id = orig_frame.clone();
            lidar_odom.pose.pose.position.x = t_world_lidar.translation.vector.x;
            lidar_odom.pose.pose.position.y = t_world_lidar.translation.vector.y;
            lidar_odom.pose.pose.position.z = t_world_lidar.translation.vector.z;
            lidar_odom.pose.pose.orientation.x = t_world_lidar.rotation.i;
            lidar_odom.pose.pose.orientation.y = t_world_lidar.rotation.j;
            lidar_odom.pose.pose.orientation.z = t_world_lidar.rotation.k;
            lidar_odom.pose.pose.orientation.w = t_world_lidar.rotation.w;

            let lp_pose = &lidar_path.poses;
            let changed_lidar_path = match lp_pose.last() {
                Some(last_pose) => {
                    let last_pose = last_pose.pose.clone();
                    let last_pose_position = last_pose.position;
                    let current_pose_position = lidar_odom.pose.pose.position.clone();

                    let distance = nalgebra::distance(
                        &nalgebra::Point3::new(
                            last_pose_position.x,
                            last_pose_position.y,
                            last_pose_position.z,
                        ),
                        &nalgebra::Point3::new(
                            current_pose_position.x,
                            current_pose_position.y,
                            current_pose_position.z,
                        ),
                    );

                    if distance > path_update_translate_threshold {
                        lidar_path.poses.push(PoseStamped {
                            pose: lidar_odom.pose.pose.clone(),
                            header: lidar_odom.header.clone(),
                        });
                        true
                    } else {
                        false
                    }
                }
                None => {
                    lidar_path.poses.push(PoseStamped {
                        pose: lidar_odom.pose.pose.clone(),
                        header: lidar_odom.header.clone(),
                    });
                    true
                }
            };

            #[cfg(debug_assertions)]
            #[cfg(all(not(feature = "no-viz"), not(feature = "gicp"), not(feature = "rerun")))]
            {
                if let Some(registered_normals) = register_pose.registered_cloud {
                    let random_color = {
                        let mut rng = rand::rng();
                        let r: f32 = rng.gen();
                        let g: f32 = rng.gen();
                        let b: f32 = rng.gen();
                        let v = nalgebra::Vector3::new(r, g, b).normalize();
                        let a: f32 = 0.7;
                        (v.x, v.y, v.z, a)
                    };

                    {
                        let mut marr = Marker::default();
                        marr.header = Header {
                            frame_id: config.prelio_params.map_frame.clone(),
                            stamp: Time {
                                sec: orig_time.sec,
                                nanosec: orig_time.nanosec,
                            },
                        };
                        marr.id = total_viz_marker;
                        total_viz_marker += 1;

                        // Live for ever, RIP Rviz
                        marr.lifetime.sec = 0;
                        marr.lifetime.nanosec = 0;

                        marr.type_ = 5; // Arrow
                        marr.action = 0; // add

                        marr.scale.x = 0.02; // measured distance to the plane
                        marr.scale.y = 0.02; // shaft
                        marr.scale.z = 0.02; // head

                        marr.color.r = random_color.0;
                        marr.color.g = random_color.1;
                        marr.color.b = random_color.2;
                        marr.color.a = random_color.3;

                        marr.header.frame_id = orig_frame.clone();
                        marr.header.stamp.nanosec = orig_time.nanosec; // TODO rpcl2 from timemsg
                        marr.header.stamp.sec = orig_time.sec;

                        //let draw_length = 0.25;

                        marr.points = registered_normals
                            .iter() // downsample_avg(normal_iter, 0.5).iter() || normal_iter
                            .map(|normal| {
                                let normal_viz = nalgebra::Vector3::new(
                                    normal.normal_x as f64,
                                    normal.normal_y as f64,
                                    normal.normal_z as f64,
                                );

                                let position_viz = normal.xyz_f64();
                                let end_position =
                                    position_viz + normal_viz.normalize() * normal.intensity as f64;

                                let mut point_start = Point::default();
                                point_start.x = position_viz.x;
                                point_start.y = position_viz.y;
                                point_start.z = position_viz.z;

                                let mut point_end = Point::default();
                                point_end.x = end_position.x;
                                point_end.y = end_position.y;
                                point_end.z = end_position.z;

                                [point_start, point_end]
                            })
                            .flatten()
                            .collect();

                        registered_visual_publisher
                            .publish(&marr)
                            .expect("Failed to publish registered visualization");
                    }
                }
            }

            // transform cloud inplace for publish
            /*let mut cloud: PointCloud2 = {
                let mut cloud = cloud;
                cloud.iter_mut().for_each(|p| {
                    let p_t = t_world_lidar.transform_point(&p.xyz_f64());
                    p.x = p_t.x as f32;
                    p.y = p_t.y as f32;
                    p.z = p_t.z as f32;
                });
                ros_pointcloud2::PointCloud2Msg::try_from_vec(cloud)
                    .unwrap()
                    .into()
            };*/

            // color cloud from green to red based on index in the cloud
            /*let start_color = (0.0, 1.0, 0.0);
            let end_color = (1.0, 0.0, 0.0);
            let cloud = cloud
                .iter()
                .enumerate()
                .map(|(index, p)| {
                    let color =
                        lerp_color(start_color, end_color, index as f32 / cloud.len() as f32);
                    PointXYZRGB::new(
                        p.x,
                        p.y,
                        p.z,
                        (color.0 * 255.0) as u8,
                        (color.1 * 255.0) as u8,
                        (color.2 * 255.0) as u8,
                    )
                })
                .collect();*/
            //dbg!("REG CLOUD cloud", cloud.len());
            let mut cloud: PointCloud2 = ros_pointcloud2::PointCloud2Msg::try_from_vec(cloud)
                .unwrap()
                .into();
            let map_to_lidar_transform = t_world_lidar.clone();
            cloud.header.frame_id = orig_frame.clone();

            #[cfg(all(not(feature = "gicp")))]
            {
                let total_map = map.lock().unwrap().get_entire_map();
                #[cfg(feature = "rerun")]
                {
                    let points = total_map.iter().map(|v| glam::Vec3::new(v.x, v.y, v.z));

                    if let Err(err) = rec.log("world/map", &rerun::Points3D::new(points)) {
                        error!("Error sending to rerun {err}");
                    }
                }
                #[cfg(all(not(feature = "no-viz"), not(feature = "rerun")))]
                {
                    let mut total_map = PointCloud2Msg::try_from_vec(total_map).unwrap();
                    total_map.header.frame_id = config.prelio_params.map_frame.clone();
                    let msg: PointCloud2 = total_map.into();
                    map_publisher.publish(&msg).expect("Failed to publish map");
                }
            }

            // -- TF --
            #[cfg(feature = "rerun")]
            {
                let pose = rerun::Transform3D::from_translation(glam::Vec3::new(
                    map_to_lidar_transform.translation.x as f32,
                    map_to_lidar_transform.translation.y as f32,
                    map_to_lidar_transform.translation.z as f32,
                ))
                .with_quaternion(glam::Quat::from_xyzw(
                    map_to_lidar_transform.rotation.quaternion().i as f32,
                    map_to_lidar_transform.rotation.quaternion().j as f32,
                    map_to_lidar_transform.rotation.quaternion().k as f32,
                    map_to_lidar_transform.rotation.quaternion().w as f32,
                ));

                if let Err(err) = rec.log("world/lidar", &pose) {
                    error!("Error sending to rerun {err}");
                }

                {
                    let ps: ros_pointcloud2::PointCloud2Msg = cloud.clone().into();
                    let points = ps
                        .try_into_iter()
                        .unwrap()
                        .map(|v: PointXYZ| glam::Vec3::new(v.x, v.y, v.z));

                    if let Err(err) =
                        rec.log("world/lidar/undistorted", &rerun::Points3D::new(points))
                    {
                        error!("Error sending to rerun {err}");
                    }
                }
            }

            let tf = TransformStamped {
                header: Header {
                    frame_id: config.prelio_params.map_frame.clone(),
                    stamp: Time {
                        sec: orig_time.sec,
                        nanosec: orig_time.nanosec,
                    },
                },
                child_frame_id: orig_frame,
                transform: Transform {
                    translation: Vector3 {
                        x: map_to_lidar_transform.translation.vector.x,
                        y: map_to_lidar_transform.translation.vector.y,
                        z: map_to_lidar_transform.translation.vector.z,
                    },
                    rotation: Quaternion {
                        x: map_to_lidar_transform.rotation.quaternion().i,
                        y: map_to_lidar_transform.rotation.quaternion().j,
                        z: map_to_lidar_transform.rotation.quaternion().k,
                        w: map_to_lidar_transform.rotation.quaternion().w,
                    },
                },
            };
            #[cfg(feature = "ratpub")]
            {
                let inner_tf_broad = Arc::clone(&tf_broadcaster);
                std::thread::spawn(move || {
                    future::block_on(async move {
                        inner_tf_broad
                            .send_transform(tf)
                            .await
                            .expect("Failed to send transform");
                    });
                });
            }
            #[cfg(not(feature = "ratpub"))]
            tf_broadcaster
                .send_transform(tf)
                .expect("Failed to send transform");
            // -- --

            let mut imu_odom = lidar_odom.clone();
            imu_odom.pose.pose.position.x = init_guess_world.translation.vector.x;
            imu_odom.pose.pose.position.y = init_guess_world.translation.vector.y;
            imu_odom.pose.pose.position.z = init_guess_world.translation.vector.z;
            imu_odom.pose.pose.orientation.x = init_guess_world.rotation.quaternion().i;
            imu_odom.pose.pose.orientation.y = init_guess_world.rotation.quaternion().j;
            imu_odom.pose.pose.orientation.z = init_guess_world.rotation.quaternion().k;
            imu_odom.pose.pose.orientation.w = init_guess_world.rotation.quaternion().w;

            imu_path.poses.push(PoseStamped {
                pose: imu_odom.pose.pose.clone(),
                header: imu_odom.header.clone(),
            });

            #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
            rt.block_on(async {
                cloud_publisher
                    .publish(&cloud)
                    .await
                    .expect("Failed to publish cloud");

                odom_publisher
                    .publish(&lidar_odom)
                    .await
                    .expect("Failed to publish odom");
            });
            #[cfg(not(feature = "ratpub"))]
            {
                cloud_publisher
                    .publish(&cloud)
                    .expect("Failed to publish cloud");

                odom_publisher
                    .publish(&lidar_odom)
                    .expect("Failed to publish odom");
            }

            if changed_lidar_path {
                #[cfg(all(not(feature = "no-viz"), feature = "rerun"))]
                {
                    let lidar_path =
                        rerun::LineStrip3D::from_iter(lidar_path.poses.iter().map(|p| {
                            glam::Vec3::new(
                                p.pose.position.x as f32,
                                p.pose.position.y as f32,
                                p.pose.position.z as f32,
                            )
                        }));
                    let lidar_path_strips = rerun::LineStrips3D::new(std::iter::once(lidar_path));
                    if let Err(err) = rec.log("world", &lidar_path_strips) {
                        error!("Error sending to rerun {err}");
                    }
                }
                #[cfg(all(not(feature = "no-viz"), not(feature = "ratpub")))]
                lidar_path_publisher
                    .publish(&lidar_path)
                    .expect("Failed to publish path");
            }

            #[cfg(all(not(feature = "no-viz"), not(feature = "ratpub")))]
            imu_odom_publisher
                .publish(&imu_odom)
                .expect("Failed to publish imu odom");

            #[cfg(all(not(feature = "no-viz"), not(feature = "ratpub")))]
            imu_path_publisher
                .publish(&imu_path)
                .expect("Failed to publish imu path"); // TODO this will spam a lot, maybe also throttle like lidar_path

            #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
            let pub_lidar_path = lidar_path.clone();
            #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
            let pub_imu_path = imu_path.clone();
            #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
            rt.block_on(async {
                lidar_path_publisher
                    .publish(&pub_lidar_path)
                    .await
                    .expect("Failed to publish path");

                imu_odom_publisher
                    .publish(&imu_odom)
                    .await
                    .expect("Failed to publish imu odom");

                imu_path_publisher
                    .publish(&pub_imu_path)
                    .await
                    .expect("Failed to publish imu path"); // TODO this will spam a lot, maybe also throttle like lidar_path
            });
            #[cfg(feature = "rerun")]
            {
                let imu_path = rerun::LineStrip3D::from_iter(imu_path.poses.iter().map(|p| {
                    glam::Vec3::new(
                        p.pose.position.x as f32,
                        p.pose.position.y as f32,
                        p.pose.position.z as f32,
                    )
                }));
                let imu_path_strips = rerun::LineStrips3D::new(std::iter::once(imu_path));
                if let Err(err) = rec.log("imu_path", &imu_path_strips) {
                    error!("Error sending to rerun {err}");
                }
            }

            #[cfg(feature = "gicp")]
            let name = "Lidar Odometry (GICP)";
            #[cfg(not(feature = "gicp"))]
            let name = "Lidar Inertial Odometry";

            if let Some(render_interval) = render_interval {
                if last_render.elapsed() > render_interval {
                    last_render = std::time::Instant::now();
                    println!(
                        "{}[2J{}[1;L{}\n{}",
                        27 as char,
                        27 as char,
                        &name,
                        eval.render()
                    );
                }
            }

            // mark to send new data
            #[cfg(feature = "rat")]
            rat::bacon(
                "tick_end",
                &mut "".to_string(),
                rat::VariableType::StaticOnly,
            )
            .unwrap();
        }
        info!("--- END lifis tick ---");
    });

    #[cfg(feature = "ratpub")]
    {
        std::thread::park();
        Ok(())
    }

    #[cfg(feature = "ros2-client")]
    future::block_on(async {
        match node.spinner().unwrap().spin().await {
            Err(e) => {
                error!("Node did not exit succesfully {e}");
            }
            Ok(_) => {}
        }
    });
    #[cfg(feature = "r2r")]
    {
        let spin_timeout = std::time::Duration::from_millis(10);
        loop {
            node.spin_once(spin_timeout.clone());
        }
    }

    #[cfg(feature = "ros2-client")]
    Ok(())
}

/*#[cfg(debug_assertions)]
fn main() -> color_eyre::eyre::Result<(), Box<dyn std::error::Error>> {
    color_eyre::install()?;
    env_logger::init();
    log::warn!("Running in dev mode.");
    lio()
}*/

//#[cfg(not(debug_assertions))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    match args.command {
        Command::Run(args) => lio(&args.config_path),
        Command::Config(args) => match args {
            ConfigCommand::Init(args) => {
                let config = LioConfig::default();
                let toml = toml::to_string(&config)?;
                std::fs::write(&args.config_path, toml)?;
                Ok(())
            }
            ConfigCommand::Test(args) => {
                let _config = read_config::<LioConfig>(&args.config_path)?;
                Ok(())
            }
        },
    }
}
