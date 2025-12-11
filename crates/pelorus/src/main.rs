#[allow(unused_imports)]
use anyhow::Error;
use anyhow::anyhow;
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;
#[cfg(feature = "rat")]
use std::fs;
#[cfg(feature = "rat")]
use std::process::{Command as ProcessCommand, Stdio};
use std::time::UNIX_EPOCH;

#[global_allocator]
#[cfg(feature = "mimalloc")]
static GLOBAL: MiMalloc = MiMalloc;

mod config;
#[cfg(feature = "evaluate")]
mod evaluate;
mod mapping;
#[cfg(feature = "rat")]
mod output_tui;
use clap::{Parser, Subcommand};
#[allow(unused_imports)]
use futures_lite::future;
use log::{error, info};
use nalgebra::{Isometry3, Translation3, UnitQuaternion};
#[cfg(feature = "r2r")]
#[allow(unused_imports)]
use r2r::{
    QosProfile,
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{Point, PoseStamped, Quaternion, Transform, TransformStamped, Vector3},
    nav_msgs::msg::{Odometry, Path},
    sensor_msgs::msg::PointCloud2,
    std_msgs::msg::Header,
    visualization_msgs::msg::Marker,
};
#[cfg(feature = "ratpub")]
use ratpub::Node;
#[cfg(feature = "rerun")]
use rerun::external::glam;
#[cfg(feature = "ros2-client")]
use ros2_client::{NodeName, NodeOptions};
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
#[cfg(feature = "ros2-client")]
#[allow(unused_imports)]
use ros2_interfaces_jazzy_serde::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{Point, PoseStamped, Quaternion, Transform, TransformStamped, Vector3},
    nav_msgs::msg::Path,
    sensor_msgs::msg::PointCloud2,
    std_msgs::msg::Header,
    visualization_msgs::msg::Marker,
};

use serde::{Deserialize, Serialize};

use either::*;
#[cfg(feature = "ros2-client")]
use prelio_ros2::tf::TfBroadcaster;
#[cfg(feature = "ratpub")]
use prelio_ros2::tf_lh::TfBroadcaster;
#[cfg(feature = "r2r")]
use prelio_ros2::tf_r2r::TfBroadcaster;
use prelio_ros2::{
    LifisNodeParameters,
    prelio::{LocatedBufferView, config::LifisParameters},
};

use pelorus_reg::{Registration, registration::RegistrationState};
use prelio_ros2::{PrelioNode, prelio::imu::ImuMsg};
use ros_pointcloud2::prelude::*;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

pub fn timestamp_to_secs(t: &TimeMsg) -> f64 {
    t.sec as f64 + t.nanosec as f64 / 1_000_000_000.0
}

pub fn rpcl2_time_to_r2r_time(t: &ros_pointcloud2::ros::TimeMsg) -> TimeMsg {
    TimeMsg {
        sec: t.sec,
        nanosec: t.nanosec,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
    pub lidar_odom_cov: f64,
    pub enable_live_extrinsic_calibration: bool, // TODO rm
    pub prelio_params: LifisParameters,
    pub prelio_node_params: LifisNodeParameters,
    pub k_neighbors: usize,
    pub neighbor_max_range: f64,
    pub ekf_max_iterations: i8, // TODO rm
    pub dist_update_sensitivity: f32,
    pub lidar_as_map_tf: bool, // Transforms all points and paths back to the lidar TF before publishing. All calculations are done in the IMU frame.
    pub mapping_resolution: f32,
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

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Clone, Subcommand)]
#[command()]
pub enum Command {
    /// Run the Lidar-Inertial Odometry process
    #[command(subcommand)]
    Run(RunArgs),

    /// Manage configuration files
    #[command(subcommand)]
    Config(ConfigCommand),

    #[cfg(feature = "evaluate")]
    Evaluate(EvalArgs),
}

#[derive(Debug, Clone, Parser)]
#[command()]
pub enum RunArgs {
    /// Spawn the server process and wait for input
    Online(ConfigArgs),
    /// Use a Bagfile directly and run synchronously
    #[cfg(feature = "rat")]
    Offline(OfflineArgs),
}

#[derive(Debug, Clone, Parser)]
pub struct OfflineArgs {
    /// Optional path to a custom '.rl' configuration file.
    /// If not provided, only the default configuration will be used
    #[arg(short, long)]
    pub config_path: Option<PathBuf>,

    /// Path to the Bagfile (directory) as input. The Bagfile must be use the MCAP storage backend
    pub bagfile_path: PathBuf,

    /// Path to the Minot binary
    #[arg(short, long, default_value = "minot")]
    pub minot_path: PathBuf,

    /// Export a TUM trajectory file.
    #[arg(short, long)]
    pub export_trajectory: bool,
}

#[derive(Debug, Clone, Parser)]
pub struct ConfigArgs {
    /// Optional path to a custom '.rl' configuration file to overwrite defaults.
    /// If not provided, only the default configuration will be used
    #[arg(short, long)]
    pub config_path: Option<PathBuf>,

    /// Export a TUM trajectory file.
    #[arg(short, long)]
    pub export_trajectory: bool,

    /// Publish tick_end checkpoint using ratlib from Minot.
    #[arg(short, long)]
    pub rat: bool,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ConfigCommand {
    /// Prints the default configuration to standard output.
    /// You can redirect this to a file, e.g., `pelorus config init > my_config.rl`
    Init,
    /// Compiles and validates a given configuration file against the defaults
    Test(TestArgs),
}

#[derive(Debug, Clone, Parser)]
pub struct EvalArgs {
    pub estimated_file: PathBuf,
    pub ground_truth_file: PathBuf,
}

#[derive(Debug, Clone, Parser)]
pub struct TestArgs {
    /// The path of the configuration file to test
    #[arg(default_value = "pelorus.rl")]
    pub config_path: PathBuf,
}

#[derive(Debug)]
#[allow(dead_code)]
struct RegisteredCloud {
    cloud: ros_pointcloud2::PointCloud2Msg,
    registration: nalgebra::Isometry3<f32>,
    timestamp: f64,
}

fn lio(
    config: LioConfig,
    export_trajectory: bool,
    synchronous_mode: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    lio_with_ready_signal(config, None, export_trajectory, synchronous_mode)
}

fn lio_with_ready_signal(
    config: LioConfig,
    ready_signal: Option<std::sync::mpsc::Sender<()>>,
    export_trajectory: bool,
    #[allow(unused_variables)] synchronous_mode: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(any(feature = "ratpub", feature = "rat"))]
    let rt = if cfg!(feature = "ratpub") || (synchronous_mode && cfg!(feature = "rat")) {
        Some(Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap(),
        ))
    } else {
        None
    };

    #[cfg(feature = "rat")]
    if synchronous_mode {
        if let Some(rt) = rt.as_ref() {
            rat::init("pelorus", None, Some(Arc::clone(&rt))).unwrap();
        }
    }

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
    let node = Arc::new(
        rt.as_ref()
            .expect("rt always exists with ratpub feat")
            .block_on(async { Node::create("pelorus_node".to_owned()).await.unwrap() }),
    );

    #[cfg(feature = "r2r")]
    let ctx = r2r::Context::create()?;
    #[cfg(feature = "r2r")]
    let mut node = r2r::Node::create(ctx, "lio", "lio")?;

    #[cfg(not(feature = "ratpub"))]
    let tf_broadcaster = TfBroadcaster::new(&mut node);
    #[cfg(feature = "ratpub")]
    let tf_broadcaster = Arc::new(TfBroadcaster::new(
        &node,
        rt.as_ref()
            .expect("rt always exists with ratpub feat")
            .clone(),
    ));

    let ext = Isometry3::from_parts(
        nalgebra::Translation3::new(
            config.imu_params.initial_translation_lidar_to_imu.x,
            config.imu_params.initial_translation_lidar_to_imu.y,
            config.imu_params.initial_translation_lidar_to_imu.z,
        ),
        UnitQuaternion::from_matrix(&config.imu_params.initial_rotation_lidar_to_imu),
    );

    let map = Arc::new(Mutex::new(RegistrationState::new(
        config.map_resolution,
        config.map_capacity,
        config.prelio_params.imu.gravity,
        &config.imu_params,
        config.lidar_odom_cov,
        // config.k_neighbors, // TODO remove from config
        config.neighbor_max_range,
        ext,
        pelorus_reg::IVoxNearbyType::Nearby26,
        config.dist_update_sensitivity,
    )));

    let mean_acc = Arc::new(Mutex::new(None));

    #[cfg(feature = "rerun")]
    let rec = {
        let rerun_server_addr =
            std::env::var("RERUN_SERVER_ADDR").unwrap_or("127.0.0.1:9876".to_string());
        let rec = Arc::new(
            rerun::RecordingStreamBuilder::new("pelorus")
                .connect_grpc_opts(format!("rerun+http://{}/proxy", rerun_server_addr))?,
        );
        rec
    };
    // let rec = Arc::new(rerun::RecordingStreamBuilder::new("pelorus").spawn()?);

    let lidar_vendor = config.prelio_params.lidar_vendor.clone();

    let (ready_for_next_tx, ready_for_next_rx) = std::sync::mpsc::channel();
    let (_lifis_node, synced_rx) = PrelioNode::new(
        #[cfg(not(feature = "ratpub"))]
        &mut node,
        #[cfg(feature = "ratpub")]
        &node,
        #[cfg(feature = "ratpub")]
        rt.as_ref()
            .expect("rt always exists with ratpub feat")
            .clone(),
        &config.prelio_params,
        &config.prelio_node_params,
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
                        angular_velocity: nalgebra::Vector3::default(), // TODO add ang vel
                                                                        // lidar_imu_extrinsic_rotation: nalgebra::Matrix3::identity(),
                                                                        // lidar_imu_extrinsic_translation: nalgebra::Vector3::default(),
                    }
                },
            ))
        },
        #[cfg(feature = "rerun")]
        Arc::clone(&rec),
        Some(ready_for_next_rx),
    )?;

    // #[cfg(feature = "ros2-client")]
    // {
    //     #[cfg(all(feature = "map", not(feature = "no-viz")))]
    //     {
    //         let map_publisher = node
    //             .create_publisher::<PointCloud2>(
    //                 &node
    //                     .create_topic(
    //                         &ros2_client::Name::new(&namespace, "map").unwrap(),
    //                         ros2_client::MessageTypeName::new("sensor_msgs", "PointCloud2"),
    //                         &Qos::default().into(),
    //                     )
    //                     .unwrap(),
    //                 None,
    //             )
    //             .unwrap();
    //     }
    // }
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
    #[cfg(any(feature = "ros2-client"))]
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
    // #[cfg(all(
    //     not(feature = "no-viz"),
    //     feature = "ros2-client",
    //     not(feature = "rerun")
    // ))]
    // let registered_visual_publisher = node
    //     .create_publisher::<Marker>(
    //         &node
    //             .create_topic(
    //                 &ros2_client::Name::new(&format!("{namespace}/viz"), "registered").unwrap(),
    //                 ros2_client::MessageTypeName::new("visualization_msgs", "Marker"),
    //                 &ros2_client::DEFAULT_PUBLISHER_QOS,
    //             )
    //             .unwrap(),
    //         None,
    //     )
    //     .unwrap();
    // #[cfg(all(not(feature = "no-viz"), feature = "map", feature = "r2r"))]
    // let map_publisher = node.create_publisher::<PointCloud2>("map", QosProfile::default())?;
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
    let cloud_publisher = rt
        .as_ref()
        .expect("rt always exists with ratpub feat")
        .block_on(async {
            node.create_publisher::<PointCloud2>("registered".to_owned())
                .await
                .unwrap()
        });
    #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    let odom_publisher = node.create_publisher::<Odometry>("odom", config.odom_qos.into())?;
    #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
    let odom_publisher = rt
        .as_ref()
        .expect("rt always exists with ratpub feat")
        .block_on(async {
            node.create_publisher::<Odometry>("odom".to_owned())
                .await
                .unwrap()
        });
    #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    let lidar_path_publisher = node.create_publisher::<Path>("path", config.path_qos.into())?;
    #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
    let lidar_path_publisher = rt
        .as_ref()
        .expect("rt always exists with ratpub feat")
        .block_on(async {
            node.create_publisher::<Path>("path".to_owned())
                .await
                .unwrap()
        });
    #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    let imu_odom_publisher =
        node.create_publisher::<Odometry>("imu_odom", config.imu_odom_qos.into())?;
    #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
    let imu_odom_publisher = rt
        .as_ref()
        .expect("rt always exists with ratpub feat")
        .block_on(async {
            node.create_publisher::<Odometry>("imu_odom".to_owned())
                .await
                .unwrap()
        });
    #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    let imu_path_publisher =
        node.create_publisher::<Path>("imu_path", config.imu_path_qos.into())?;
    #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
    let imu_path_publisher = rt
        .as_ref()
        .expect("rt always exists with ratpub feat")
        .block_on(async {
            node.create_publisher::<Path>("imu_path".to_owned())
                .await
                .unwrap()
        });
    // #[cfg(all(
    //     not(feature = "no-viz"),
    //     feature = "map",
    //     feature = "r2r",
    //     not(feature = "rerun")
    // ))]
    // let registered_visual_publisher =
    // node.create_publisher::<Marker>("registered_viz", r2r::QosProfile::default())?;

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

    let path_update_translate_threshold = 0.15; // default 0.15 meters - TODO config

    let mut eval = prelio_ros2::prelio::timing::RuntimeEvaluator::new();

    let render_interval = if config.feedback_fps == 0 {
        None
    } else {
        Some(std::time::Duration::from_secs_f64(
            1.0 / config.feedback_fps as f64,
        ))
    };
    let mut last_render = std::time::Instant::now();

    #[cfg(feature = "mapping")]
    #[allow(unused_variables)]
    let (out_chan_sender, out_chan_receiver) = std::sync::mpsc::channel::<RegisteredCloud>();

    // TODO handle just mapping without rerun with fileio
    // handle mapping of the registered results
    #[cfg(all(feature = "mapping", feature = "rerun"))]
    {
        let mapping_rec = rec.clone();
        let mapping_resolution = config.mapping_resolution;
        std::thread::spawn(move || {
            let mut total_map = mapping::VoxelGrid::new(mapping_resolution);
            while let Ok(reg) = out_chan_receiver.recv() {
                reg.cloud.try_into_iter().unwrap().for_each(|p: PointXYZ| {
                    let p = nalgebra::Point3::new(p.x, p.y, p.z);
                    let p = reg.registration.transform_point(&p);
                    total_map.insert(p);
                });

                mapping_rec.set_timestamp_secs_since_epoch("sensor", reg.timestamp);
                if let Err(err) = mapping_rec.log(
                    "world/mapping",
                    &rerun::Points3D::update_fields()
                        .with_positions(total_map.iter().map(|p| glam::Vec3::new(p.x, p.y, p.z))),
                ) {
                    error!("Error sending to rerun {err}");
                }

                #[cfg(feature = "rat")]
                if synchronous_mode {
                    rat::bacon(
                        "tick_end",
                        &mut "".to_string(),
                        rat::VariableType::StaticOnly,
                    )
                    .unwrap();
                }
            }
        });
    }

    std::thread::spawn(move || {
        // Initialize Prelio prediction
        ready_for_next_tx.send(()).unwrap();

        #[cfg(feature = "cuda")]
        let cuda_stream = cuda::Stream::new();

        // TODO avoid casting f32, maybe res needs f64?
        #[cfg(feature = "cuda")]
        cuda::init_map(
            &cuda_stream,
            config.map_capacity,
            config.map_resolution as f32,
            config.dist_update_sensitivity,
            config.neighbor_max_range as f32,
            cuda::NeighborType::Neighbor26,
        );

        let mut trajectory_file = {
            if export_trajectory {
                let name = petname::petname(2, "_").unwrap();
                let filename = format!(
                    "pelorus_trajectory_{}_{}.txt",
                    std::time::SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("time goes forward")
                        .as_secs(),
                    name
                );
                println!("Saving trajectory to {}", &filename);
                Some(std::fs::File::create(&filename).unwrap())
            } else {
                None
            }
        };

        if let Some(signal) = ready_signal {
            let _ = signal.send(());
        }

        while let Ok(lifis_tick) = synced_rx.recv() {
            eval.merge(&lifis_tick.runtime_eval);

            #[cfg(feature = "cuda")]
            let cloud =
                LocatedBufferView::copy_from_slice_cuda(&cuda_stream, &lifis_tick.deskewed_cloud);
            #[cfg(not(feature = "cuda"))]
            let cloud = LocatedBufferView::Ram(lifis_tick.deskewed_cloud);

            let orig_frame = lifis_tick.cloud_header.frame_id.clone();
            let orig_time = lifis_tick.cloud_header.stamp.clone();
            let current_cloud_time = timestamp_to_secs(&lifis_tick.odom_time);
            if let Some(existing_last_scan_time) = last_scan_time {
                if current_cloud_time < existing_last_scan_time {
                    let mut map = map.lock().unwrap();
                    map.clear_localmap();

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

            let (register_pose, ekf_state) = {
                let mut map = map.lock().unwrap();
                match map.register_and_map(
                    lifis_tick.map_only,
                    ext,
                    &cloud,
                    Some(&ready_for_next_tx),
                    &mut eval,
                    #[cfg(feature = "cuda")]
                    &cuda_stream,
                ) {
                    Ok(Some(pose)) => (pose, map.get_state()),
                    Ok(None) => continue, // skip outputting trivial solutions on initialization
                    Err(err) => {
                        error!("{err}");
                        continue;
                    }
                }
            };

            log::debug!("Got a new registration");

            #[cfg(feature = "rerun")]
            rec.set_timestamp_secs_since_epoch("sensor", current_cloud_time);

            let rev_imu_tf = if config.lidar_as_map_tf {
                ekf_state.lidar_imu_extrinsic.inverse().cast::<f32>()
            } else {
                Isometry3::identity()
            };

            // Transformation from the last pose to the current pose in the local frame.
            t_last_current = t_world_lidar.inverse() * register_pose.pose;
            t_world_lidar = rev_imu_tf.cast() * register_pose.pose;

            let mut lidar_odom = lifis_tick.imu_odom.clone();
            lidar_odom.header.stamp = Time {
                sec: lifis_tick.odom_time.sec,
                nanosec: lifis_tick.odom_time.nanosec,
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
            // TODO publish with covariances to own topic separately

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

            let mut cloud: PointCloud2 = {
                match cloud {
                    LocatedBufferView::Ram(items) => {
                        ros_pointcloud2::PointCloud2Msg::try_from_vec(&items)
                            .unwrap()
                            .into()
                    }
                    #[cfg(feature = "cuda")]
                    LocatedBufferView::Cuda { buffer: cloud } => {
                        let buff = cloud.into_cloud(&cuda_stream);
                        ros_pointcloud2::PointCloud2Msg::try_from_vec(&buff)
                            .unwrap()
                            .into()
                    }
                }
            };
            let map_to_lidar_transform = t_world_lidar.clone().cast::<f32>();
            cloud.header.frame_id = orig_frame.clone();
            if let Some(mut trajectory_file) = trajectory_file.as_mut() {
                let time = timestamp_to_secs(&lifis_tick.cloud_header.stamp);
                let t = &map_to_lidar_transform.translation.vector;
                let q = map_to_lidar_transform.rotation.quaternion();

                // <time in seconds> <translation_x> <translation_y> <translation_z> <quaternion_x> <quaternion_y> <quaternion_z> <quaternion_w>
                // Format the string to write to the file
                let content_to_write = format!(
                    "{} {} {} {} {} {} {} {}\n",
                    time, t.x, t.y, t.z, q.i, q.j, q.k, q.w
                );

                std::io::Write::write_all(&mut trajectory_file, content_to_write.as_bytes())
                    .unwrap();
            }

            #[cfg(all(feature = "map", all(feature = "rerun", feature = "ratpub")))]
            {
                eval.start("pub_localmap").unwrap();
                #[cfg(not(feature = "cuda"))]
                let total_map = map.lock().unwrap().get_entire_map();
                #[cfg(feature = "cuda")]
                let total_map = cuda::get_map();
                #[cfg(feature = "rerun")]
                {
                    let points = total_map.iter().map(|v| {
                        let v = rev_imu_tf.transform_point(&v);
                        glam::Vec3::new(v.x, v.y, v.z)
                    });

                    if let Err(err) = rec.log(
                        "world/map",
                        &rerun::Points3D::update_fields().with_positions(points),
                    ) {
                        error!("Error sending to rerun {err}");
                    }
                }
                #[cfg(all(
                    not(feature = "no-viz"),
                    not(feature = "rerun"),
                    not(feature = "ratpub") // currently no use of the total map in ratpub
                ))]
                {
                    let mut total_map = PointCloud2Msg::try_from_vec(
                        total_map
                            .iter()
                            .map(|p| {
                                let p = rev_imu_tf.transform_point(&p);
                                PointXYZ::new(p.x, p.y, p.z)
                            })
                            .collect(),
                    )
                    .unwrap();
                    total_map.header.frame_id = config.prelio_params.map_frame.clone();
                    let msg: PointCloud2 = total_map.into();
                    map_publisher.publish(&msg).expect("Failed to publish map");
                }
                eval.stop("pub_localmap").unwrap();
            }

            // -- TF --
            let tf = TransformStamped {
                header: Header {
                    frame_id: config.prelio_params.map_frame.clone(),
                    stamp: Time {
                        sec: lifis_tick.odom_time.sec,
                        nanosec: lifis_tick.odom_time.nanosec,
                    },
                },
                child_frame_id: orig_frame,
                transform: Transform {
                    translation: Vector3 {
                        x: map_to_lidar_transform.translation.vector.x as f64,
                        y: map_to_lidar_transform.translation.vector.y as f64,
                        z: map_to_lidar_transform.translation.vector.z as f64,
                    },
                    rotation: Quaternion {
                        x: map_to_lidar_transform.rotation.quaternion().i as f64,
                        y: map_to_lidar_transform.rotation.quaternion().j as f64,
                        z: map_to_lidar_transform.rotation.quaternion().k as f64,
                        w: map_to_lidar_transform.rotation.quaternion().w as f64,
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

            #[cfg(feature = "rerun")]
            {
                let pose = rerun::Transform3D::from_translation(glam::Vec3::new(
                    map_to_lidar_transform.translation.x,
                    map_to_lidar_transform.translation.y,
                    map_to_lidar_transform.translation.z,
                ))
                .with_quaternion(glam::Quat::from_xyzw(
                    map_to_lidar_transform.rotation.quaternion().i,
                    map_to_lidar_transform.rotation.quaternion().j,
                    map_to_lidar_transform.rotation.quaternion().k,
                    map_to_lidar_transform.rotation.quaternion().w,
                ));

                if let Err(err) = rec.log("world/lidar", &pose) {
                    error!("Error sending to rerun {err}");
                }
            }

            #[cfg(feature = "mapping")]
            {
                let ps: ros_pointcloud2::PointCloud2Msg = cloud.clone().into();
                let reg_out = RegisteredCloud {
                    cloud: ps,
                    registration: map_to_lidar_transform,
                    timestamp: current_cloud_time,
                };

                let _ = out_chan_sender.send(reg_out); // currently ignore res
            }

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

            #[cfg(feature = "rerun")]
            {
                const DECAY: bool = false;

                let ps: ros_pointcloud2::PointCloud2Msg = cloud.clone().into();
                let gradient = colorous::TURBO;
                let (positions, colors): (Vec<_>, Vec<_>) = ps
                    .try_into_iter()
                    .unwrap()
                    .map(|v: PointXYZI| {
                        let color = gradient.eval_continuous(v.intensity as f64);
                        let v = if DECAY {
                            map_to_lidar_transform
                                .transform_point(&nalgebra::Point3::new(v.x, v.y, v.z).cast())
                        } else {
                            nalgebra::Point3::new(v.x, v.y, v.z)
                        };
                        let pos = glam::Vec3::new(v.x as f32, v.y as f32, v.z as f32);
                        let color_rgb = [color.r, color.g, color.b];

                        (pos, color_rgb)
                    })
                    .unzip();

                let topic = if DECAY {
                    format!("undistorted_{:?}", current_cloud_time)
                } else {
                    "world/lidar/undistorted".to_string()
                };

                if let Err(err) = rec.log(
                    topic,
                    &rerun::Points3D::update_fields()
                        .with_positions(positions)
                        .with_colors(colors)
                        .with_radii([rerun::Radius::new_ui_points(1.25)]), // for video scene_points: 0.025
                ) {
                    error!("Error sending to rerun {err}");
                }
            }

            #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
            rt.as_ref()
                .expect("rt always exists with ratpub feat")
                .block_on(async {
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
            rt.as_ref()
                .expect("rt always exists with ratpub feat")
                .block_on(async {
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
            #[cfg(all(feature = "rat", not(feature = "mapping")))]
            if synchronous_mode {
                rat::bacon(
                    "tick_end",
                    &mut "".to_string(),
                    rat::VariableType::StaticOnly,
                )
                .unwrap();
            }
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    const DEFAULT_CONFIG_STR: &str = include_str!("../../../default.rl");

    match args.command {
        Command::Run(args) => {
            match args {
                RunArgs::Online(config_args) => {
                    env_logger::init();

                    let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
                        .map_err(|e| anyhow!("Failed to compile embedded default config: {}", e))?;

                    let user_ast = if let Some(config_path) = config_args.config_path {
                        let path = std::path::Path::new(&config_path);
                        if !path.exists() {
                            return Err(anyhow!(
                                "Error: Configuration file not found at '{}'",
                                config_path.to_string_lossy()
                            )
                            .into());
                        }
                        ratslang::compile_file(&path.to_path_buf(), None, None).map_err(|e| {
                            anyhow!(
                                "Failed to compile user config '{}': {}",
                                config_path.to_string_lossy(),
                                e
                            )
                        })?
                    } else {
                        ratslang::compile_code("")?
                    };

                    let config = config::ast_to_config(default_ast, user_ast)?;
                    lio(config, config_args.export_trajectory, config_args.rat)
                }
                #[cfg(feature = "rat")]
                RunArgs::Offline(offline_args) => {
                    // TODO check minot version and if features are correct, else install using script to use one more feature so we can talk to it. Use our stdin/stdout for the script and start TUI when minot is ok.
                    let output_capture = std::sync::Arc::new(output_tui::OutputCapture::new());
                    let capture_for_minot_stdout = output_capture.clone();
                    let capture_for_minot_stderr = output_capture.clone();
                    let capture_for_tui = output_capture.clone();

                    let logger = output_tui::CaptureLogger::new(output_capture.clone());
                    log::set_boxed_logger(Box::new(logger))
                        .map_err(|e| anyhow!("Failed to set logger: {}", e))?;
                    log::set_max_level(log::LevelFilter::Info);

                    let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
                        .map_err(|e| anyhow!("Failed to compile embedded default config: {}", e))?;

                    let user_ast = if let Some(config_path) = offline_args.config_path {
                        let path = std::path::Path::new(&config_path);
                        if !path.exists() {
                            return Err(anyhow!(
                                "Error: Configuration file not found at '{}'",
                                config_path.to_string_lossy()
                            )
                            .into());
                        }
                        ratslang::compile_file(&path.to_path_buf(), None, None).map_err(|e| {
                            anyhow!(
                                "Failed to compile user config '{}': {}",
                                config_path.to_string_lossy(),
                                e
                            )
                        })?
                    } else {
                        ratslang::compile_code("")?
                    };

                    let config = {
                        let mut config = config::ast_to_config(default_ast, user_ast)?;
                        config.feedback_fps = 0; // disable feedback for offline
                        config
                    };

                    let lidar_topic = &config.prelio_node_params.cloud_src_topic;
                    let imu_topic = &config.prelio_node_params.imu_src_topic;
                    let imu_init_time_ms =
                        (config.prelio_node_params.imu_predictor_init_time_sec * 1000.0) as u64;
                    let scan_init_samples = config.prelio_params.initial_scan_accumulate;
                    let bagfile_path_canon = offline_args.bagfile_path.canonicalize().unwrap();
                    let bagfile_path = bagfile_path_canon.clone().to_string_lossy().to_string();

                    let mt_content = format!(
                        r#"
_bag.{{
  cloud.{{
    _topic = "{lidar_topic}"
    _short = "l"
    _type = Cloud
  }}
  imu.{{
    _topic = "{imu_topic}"
    _short = "i"
    _type = Imu
  }}
}}
_start_locked = false

pf! [i, l], l 1 tick_end
pf! l 1 tick_end
pf! i 1 tick_end

reset! {bagfile_path}

# imu init
pf! i 1 
pf! [i,l] {imu_init_time_ms}ms
pf! i 1 

# initial aggregate, starting loop
pf! [i, l], l {scan_init_samples}
pf! l 1
pf! i 1

pf! [i, l], l 1
pf! l 1
pf! i 1
"#
                    );

                    let mt_file_path = std::env::temp_dir().join("pelorus_offline.mt");
                    fs::write(&mt_file_path, mt_content)
                        .map_err(|e| anyhow!("Failed to write .mt file: {}", e))?;

                    info!("Generated .mt file at: {:?}", mt_file_path);

                    // Spawn minot in headless mode with --sync
                    let mut minot_child = ProcessCommand::new(&offline_args.minot_path)
                        .arg("headless")
                        .arg(&mt_file_path)
                        .arg("--sync")
                        .stdin(Stdio::piped())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn()
                        .map_err(|e| anyhow!("Failed to spawn minot: {}", e))?;

                    info!("Spawned minot process with PID: {}", minot_child.id());

                    let mut minot_stdin = minot_child
                        .stdin
                        .take()
                        .ok_or_else(|| anyhow!("Failed to open minot stdin"))?;
                    let minot_stdout = minot_child
                        .stdout
                        .take()
                        .ok_or_else(|| anyhow!("Failed to open minot stdout"))?;
                    let minot_stderr = minot_child
                        .stderr
                        .take()
                        .ok_or_else(|| anyhow!("Failed to open minot stderr"))?;

                    info!("Waiting for minot to be ready...");
                    use std::io::{BufRead, BufReader};

                    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
                    let _ready_tx_guard = ready_tx.clone();
                    let (exit_tx, exit_rx) = std::sync::mpsc::channel();

                    // Thread for stdout
                    let exit_tx_clone = exit_tx.clone();
                    let ready_tx_clone = ready_tx.clone();
                    std::thread::spawn(move || {
                        let reader = BufReader::new(minot_stdout);
                        for line in reader.lines() {
                            match line {
                                Ok(line) => {
                                    capture_for_minot_stdout.add_minot_line(line.clone());

                                    // detect end of bagfile
                                    if line.to_lowercase().contains("end of bagfile") {
                                        let _ = exit_tx_clone.send(());
                                    }
                                    if line.contains("--sync")
                                        || line.to_lowercase().contains("waiting for input")
                                        || line.to_lowercase().contains("press enter")
                                        || line.to_lowercase().contains("stdin")
                                    {
                                        let _ = ready_tx_clone.send(true);
                                    }
                                }
                                Err(e) => {
                                    capture_for_minot_stdout
                                        .add_minot_line(format!("Error reading stdout: {}", e));
                                    break;
                                }
                            }
                        }
                    });

                    // Thread for stderr
                    std::thread::spawn(move || {
                        let reader = BufReader::new(minot_stderr);
                        for line in reader.lines() {
                            match line {
                                Ok(line) => {
                                    capture_for_minot_stderr.add_minot_line(line.clone());

                                    // detect readiness on stderr
                                    if line.contains(
                                        "Waiting for input on stdin before execution (--sync mode)",
                                    ) || line.to_lowercase().contains("waiting for input")
                                        || line.to_lowercase().contains("press enter")
                                        || line.to_lowercase().contains("stdin")
                                    {
                                        let _ = ready_tx.send(true);
                                    }
                                }
                                Err(e) => {
                                    capture_for_minot_stderr
                                        .add_minot_line(format!("Error reading stderr: {}", e));
                                    break;
                                }
                            }
                        }
                    });

                    let got_ready = ready_rx
                        .recv_timeout(std::time::Duration::from_secs(30))
                        .is_ok();
                    if !got_ready {
                        info!("Did not receive explicit Minot ready; proceeding as fallback");
                    }

                    info!("Minot is ready, starting pelorus offline processing...");

                    let (pelorus_ready_tx, pelorus_ready_rx) = std::sync::mpsc::channel();

                    // start LIO in thread
                    let lio_handle = std::thread::spawn(move || {
                        if let Err(e) = lio_with_ready_signal(
                            config,
                            Some(pelorus_ready_tx),
                            offline_args.export_trajectory,
                            true,
                        ) {
                            error!("LIO failed: {}", e);
                            return Err(format!("LIO failed: {}", e));
                        }
                        Ok(())
                    });

                    info!("Waiting for pelorus to initialize...");
                    match pelorus_ready_rx.recv_timeout(std::time::Duration::from_secs(5)) {
                        Ok(_) => info!("Pelorus is ready!"),
                        Err(_) => info!(
                            "Pelorus ready not received in 5s; proceeding to trigger minot anyway"
                        ),
                    }
                    info!("Triggering minot to start processing...");

                    use std::io::Write;
                    minot_stdin
                        .write_all(b"\n")
                        .map_err(|e| anyhow!("Failed to write to minot stdin: {}", e))?;
                    minot_stdin
                        .flush()
                        .map_err(|e| anyhow!("Failed to flush minot stdin: {}", e))?;

                    // start TUI in thread
                    let (tui_stop_tx, tui_stop_rx) = std::sync::mpsc::channel();
                    let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel();
                    let tui_thread = std::thread::spawn(move || {
                        let mut tui = output_tui::OutputTui::new(
                            capture_for_tui,
                            bagfile_path_canon
                                .file_name()
                                .unwrap()
                                .to_string_lossy()
                                .to_string(),
                        )
                        .with_stop_receiver(tui_stop_rx)
                        .with_shutdown_sender(shutdown_tx);
                        let _ = tui.run();
                    });

                    // wait for end, TODO use different function to handle generating output after end
                    let lio_finished =
                        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let lio_finished_clone = lio_finished.clone();

                    std::thread::spawn(move || {
                        let _ = lio_handle.join();
                        lio_finished_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                    });

                    // wait loop
                    loop {
                        if exit_rx.try_recv().is_ok() {
                            break;
                        }
                        if lio_finished.load(std::sync::atomic::Ordering::SeqCst) {
                            break;
                        }
                        if shutdown_rx.try_recv().is_ok() {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }

                    // stop TUI thread gracefully to restore terminal state
                    let _ = tui_stop_tx.send(());
                    let _ = tui_thread.join();

                    if exit_rx.try_recv().is_err()
                        && !lio_finished.load(std::sync::atomic::Ordering::SeqCst)
                    {
                        let _ = minot_child.kill();
                        std::process::exit(130); // "Command terminated by user."
                    }

                    Ok(())
                }
            }
        }
        Command::Config(cmd) => match cmd {
            ConfigCommand::Init => {
                print!("{}", DEFAULT_CONFIG_STR);
                Ok(())
            }
            ConfigCommand::Test(args) => {
                println!(
                    "Testing configuration file: {}",
                    args.config_path.to_string_lossy()
                );

                let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
                    .map_err(|e| anyhow!("Failed to compile embedded default config: {}", e))?;

                let path = std::path::Path::new(&args.config_path);
                if !path.exists() {
                    return Err(anyhow!(
                        "Error: Configuration file not found at '{}'",
                        args.config_path.to_string_lossy()
                    )
                    .into());
                }
                let user_ast =
                    ratslang::compile_file(&path.to_path_buf(), None, None).map_err(|e| {
                        anyhow!(
                            "Failed to compile user config '{}': {}",
                            args.config_path.to_string_lossy(),
                            e
                        )
                    })?;

                let _ = config::ast_to_config(default_ast, user_ast)?;
                println!(
                    "Configuration file at '{}' is valid.",
                    args.config_path.to_string_lossy()
                );
                Ok(())
            }
        },
        #[cfg(feature = "evaluate")]
        Command::Evaluate(paths) => {
            let estimated_path = std::path::PathBuf::from(&paths.estimated_file);
            let ground_truth_path = std::path::PathBuf::from(&paths.ground_truth_file);
            trajectory::visualize_results(&estimated_path, &ground_truth_path)?;
            Ok(())
        }
    }
}
