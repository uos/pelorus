#[allow(unused_imports)]
use anyhow::Error;
use anyhow::anyhow;
use clap::{CommandFactory, ValueEnum};
use clap_complete::{Shell, generate};
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;
use pelorus_core::config::{ast_to_config, ast_to_config_with_user_usage};
use pelorus_core::timing::RuntimeEvaluator;
use pelorus_core::{LioConfig, timestamp_to_secs};
use pelorus_reg::pipeline::Pipeline;
#[cfg(feature = "r2r")]
use prelio_io::impl_r2r;
#[cfg(feature = "ros2-client")]
use prelio_io::impl_ros2_interfaces_jazzy_jazzy;
#[cfg(feature = "ratpub")]
use prelio_io::impl_ros2_interfaces_jazzy_rkyv;
#[cfg(feature = "rat")]
use std::fs;
#[cfg(feature = "rat")]
use std::process::{Command as ProcessCommand, Stdio};
const PATH_LIMIT: usize = 10_000;

use ros_pointcloud2::prelude::*;

#[global_allocator]
#[cfg(feature = "mimalloc")]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{Parser, Subcommand};
#[allow(unused_imports)]
use futures_lite::future;
use log::{error, info, warn};
use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use std::io::IsTerminal;
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
use std::collections::VecDeque;

use prelio_io::prelio::LocatedBufferView;
#[cfg(feature = "ros2-client")]
use prelio_io::tf::TfBroadcaster;
#[cfg(feature = "ratpub")]
use prelio_io::tf_lh::TfBroadcaster;
#[cfg(feature = "r2r")]
use prelio_io::tf_r2r::TfBroadcaster;

use pelorus_reg::Registration;
use prelio_io::PrelioNode;
use std::path::PathBuf;

#[cfg(feature = "evaluate")]
use pelorus::evaluate;
use pelorus::stats_tui::{
    SharedStats, StatsTui, init_logger, update_log_level,
};
use std::time::UNIX_EPOCH;

#[allow(dead_code)]
fn lerp_color(c1: (f32, f32, f32), c2: (f32, f32, f32), t: f32) -> (f32, f32, f32) {
    (
        c1.0 * (1.0 - t) + c2.0 * t,
        c1.1 * (1.0 - t) + c2.1 * t,
        c1.2 * (1.0 - t) + c2.2 * t,
    )
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
    /// Run the odometry process
    #[command(subcommand)]
    Run(RunArgs),

    /// Create or validate configuration files
    #[command(subcommand)]
    Config(ConfigCommand),

    /// Evaluate exported trajectories
    #[cfg(feature = "evaluate")]
    Evaluate(EvalArgs),

    /// Generate shell completions
    /// Example usage: `pelorus completions bash > pelorus_completions.sh`
    Completions {
        /// The shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
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
    /// Generate a configuration. Choose 'recommended' for recommended settings or 'full' for the complete default config.
    Generate(GenerateArgs),
    /// Compiles and validates a given configuration file against the defaults
    Verify(TestArgs),
}

#[derive(Debug, Clone, Parser)]
pub struct EvalArgs {
    /// Path in TUM format.
    pub estimated_file: PathBuf,
    /// Path in TUM format.
    pub reference_file: PathBuf,
}

#[derive(Debug, Clone, Parser)]
pub struct TestArgs {
    /// The path of the configuration file to test. Use '-' to read from stdin
    #[arg(default_value = "pelorus.rl")]
    pub config_path: PathBuf,
}

#[derive(Debug, Clone, Parser)]
pub struct GenerateArgs {
    /// Which configuration to generate: 'recommended' or 'full'
    #[arg(value_enum)]
    pub kind: GenerateKind,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum GenerateKind {
    Recommended,
    Full,
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
    lio_with_ready_signal(
        config,
        None,
        export_trajectory,
        synchronous_mode,
        None,
        None,
    )
}

fn lio_with_ready_signal(
    config: LioConfig,
    ready_signal: Option<std::sync::mpsc::Sender<()>>,
    export_trajectory: bool,
    #[allow(unused_variables)] synchronous_mode: bool,
    shared_stats: Option<std::sync::Arc<SharedStats>>,
    // Optional flag that, when set to true, will cause the LIO loop to exit
    shutdown_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(any(feature = "ratpub", feature = "rat"))]
    let rt = if cfg!(feature = "ratpub") || (synchronous_mode && cfg!(feature = "rat")) {
        Some(std::sync::Arc::new(
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
            rat::init("pelorus", None, Some(std::sync::Arc::clone(rt))).unwrap();
        }
    }

    #[cfg(not(feature = "ratpub"))]
    #[allow(unused_variables)]
    let namespace = "/pelorus";
    #[cfg(feature = "ros2-client")]
    let ctx = ros2_client::Context::new()?;
    #[cfg(feature = "ros2-client")]
    let mut node = ctx.new_node(
        NodeName::new(namespace, "pelorus_node")?,
        NodeOptions::new().enable_rosout(true),
    )?;
    #[cfg(feature = "ratpub")]
    let node = std::sync::Arc::new(
        rt.as_ref()
            .expect("rt always exists with ratpub feat")
            .block_on(async { Node::create("pelorus_node".to_owned()).await.unwrap() }),
    );

    #[cfg(feature = "r2r")]
    let ctx = r2r::Context::create()?;
    #[cfg(feature = "r2r")]
    let mut node = r2r::Node::create(ctx, "pelorus", "pelorus")?;

    #[cfg(not(feature = "ratpub"))]
    let tf_broadcaster = TfBroadcaster::new(&mut node);
    #[cfg(feature = "ratpub")]
    let tf_broadcaster = std::sync::Arc::new(TfBroadcaster::new(
        &node,
        rt.as_ref()
            .expect("rt always exists with ratpub feat")
            .clone(),
    ));

    let pipeline = Pipeline::new(&config);
    let map = pipeline.map();
    let ext = pipeline.ext();
    let _odom_ext = Isometry3::from_parts(
        nalgebra::Translation3::new(0.0, 0.0, 0.0),
        UnitQuaternion::from_matrix(&config.prelio_params.wheel_to_imu_rotation),
    );

    // Set lidar vendor in shared stats if present
    if let Some(ref stats) = shared_stats {
        stats.set_lidar_vendor(config.prelio_params.lidar_vendor);
    }

    #[cfg(feature = "rerun")]
    let rec = {
        let rerun_enabled = std::env::var("RERUN_ENABLED")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        if rerun_enabled {
            let rerun_server_addr =
                std::env::var("RERUN_SERVER_ADDR").unwrap_or("127.0.0.1:9876".to_string());

            // First check if the server is reachable via TCP
            let is_reachable = std::net::TcpStream::connect_timeout(
                &rerun_server_addr
                    .parse()
                    .unwrap_or_else(|_| std::net::SocketAddr::from(([127, 0, 0, 1], 9876))),
                std::time::Duration::from_millis(500),
            )
            .is_ok();

            if !is_reachable {
                info!(
                    "Rerun server not reachable at {}. Visualization disabled.",
                    rerun_server_addr
                );
                None
            } else {
                // Server is reachable, try to connect
                match rerun::RecordingStreamBuilder::new("pelorus")
                    .connect_grpc_opts(format!("rerun+http://{}/proxy", rerun_server_addr))
                {
                    Ok(stream) => {
                        info!("Connected to Rerun at {}", rerun_server_addr);
                        Some(std::sync::Arc::new(stream))
                    }
                    Err(e) => {
                        warn!(
                            "Could not connect to Rerun at {}: {}. Visualization disabled.",
                            rerun_server_addr, e
                        );
                        None
                    }
                }
            }
        } else {
            info!("Rerun disabled via RERUN_ENABLED=false");
            None
        }
    };

    #[cfg(not(feature = "rerun"))]
    let _rerun_not_enabled = ();

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
        Some(pipeline.build_init_callback()),
        Some(pipeline.build_undistort()),
        Some(pipeline.build_init_guess()),
        #[cfg(feature = "rerun")]
        rec.clone(),
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
    #[cfg(feature = "ros2-client")]
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

    // #[cfg(all(not(feature = "no-viz"), feature = "ros2-client"))]
    // let imu_path_publisher = node
    //     .create_publisher(
    //         &node
    //             .create_topic(
    //                 &ros2_client::Name::new(&namespace, "imu_path").unwrap(),
    //                 ros2_client::MessageTypeName::new("nav_msgs", "Path"),
    //                 &config.imu_path_qos.into(),
    //             )
    //             .unwrap(),
    //         None,
    //     )
    //     .unwrap();
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
    #[cfg(feature = "r2r")]
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
    #[cfg(feature = "r2r")]
    let odom_publisher = node.create_publisher::<Odometry>("odom", config.odom_qos.into())?;
    #[cfg(feature = "ratpub")]
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
    // #[cfg(all(not(feature = "no-viz"), feature = "r2r"))]
    // let imu_path_publisher =
    //     node.create_publisher::<Path>("imu_path", config.imu_path_qos.into())?;
    // #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
    // let imu_path_publisher = rt
    //     .as_ref()
    //     .expect("rt always exists with ratpub feat")
    //     .block_on(async {
    //         node.create_publisher::<Path>("imu_path".to_owned())
    //             .await
    //             .unwrap()
    //     });
    // #[cfg(all(
    //     not(feature = "no-viz"),
    //     feature = "map",
    //     feature = "r2r",
    //     not(feature = "rerun")
    // ))]
    // let registered_visual_publisher =
    // node.create_publisher::<Marker>("registered_viz", r2r::QosProfile::default())?;

    let mut t_world = nalgebra::Isometry3::identity();
    let mut t_last_current = nalgebra::Isometry3::identity();

    // Use VecDeque for efficient removal from front (O(1) instead of O(n))
    let mut lidar_path_poses: VecDeque<PoseStamped> = VecDeque::new();
    let lidar_path_frame_id = config.prelio_params.map_frame.clone();

    // let mut imu_path_poses: VecDeque<PoseStamped> = VecDeque::new();
    // let imu_path_frame_id = config.prelio_params.map_frame.clone();

    let mut last_scan_time = None;

    #[cfg(not(feature = "no-viz"))]
    let path_update_translate_threshold = 0.05; // default 0.15 meters - TODO config
    #[cfg(not(feature = "no-viz"))]
    let path_update_rotate_threshold = 0.1; // radians (~5.7 degrees) - TODO config

    let mut eval = RuntimeEvaluator::new();

    let debug_render_interval = if config.debug_fps == 0 {
        None
    } else {
        Some(std::time::Duration::from_secs_f64(
            1.0 / config.debug_fps as f64,
        ))
    };
    let mut last_render = std::time::Instant::now();

    // Only create mapping channel if we will actually use it (mapping + rerun both enabled AND rerun is connected)
    #[cfg(all(feature = "mapping", feature = "rerun"))]
    let mapping_channel = if rec.is_some() {
        Some(std::sync::mpsc::channel::<RegisteredCloud>())
    } else {
        None
    };

    #[cfg(all(feature = "mapping", feature = "rerun"))]
    #[allow(unused_variables)]
    let (out_chan_sender, out_chan_receiver) = if let Some((tx, rx)) = mapping_channel {
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    // TODO handle just mapping without rerun with fileio
    // handle mapping of the registered results
    #[cfg(all(feature = "mapping", feature = "rerun"))]
    if let (Some(out_chan_receiver), Some(ref rec)) = (out_chan_receiver, &rec) {
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

    // Clone shutdown_flag for the processing thread; original is used for the spinner
    let processing_shutdown_flag = shutdown_flag.clone();

    let processing_handle = std::thread::spawn(move || {
        // Initialize Prelio prediction
        ready_for_next_tx.send(()).unwrap();

        #[cfg(feature = "cuda")]
        let cuda_stream = pelorus_cuda::Stream::new();

        // TODO avoid casting f32, maybe res needs f64?
        #[cfg(feature = "cuda")]
        pelorus_cuda::init_map(
            &cuda_stream,
            config.map_capacity,
            config.map_resolution as f32,
            config.dist_update_sensitivity,
            config.neighbor_max_range as f32,
            pelorus_cuda::NeighborType::Neighbor26,
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

        // Main processing loop. Use recv_timeout when a shutdown flag is provided so we can
        // check for shutdown requests periodically. If no shutdown flag is provided, block
        // on recv() as before.
        loop {
            let lifis_tick = match processing_shutdown_flag.as_ref() {
                Some(flag) => {
                    // First check if shutdown was already requested to avoid blocking.
                    if flag.load(std::sync::atomic::Ordering::Relaxed) {
                        info!("Shutdown flag detected, exiting processing loop");
                        break;
                    }

                    // Block with a short timeout so we can still respond promptly to shutdown signals.
                    match synced_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                        Ok(t) => t,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            info!("synced_rx disconnected, exiting processing loop");
                            break;
                        }
                    }
                }
                None => match synced_rx.recv() {
                    Ok(t) => t,
                    Err(_) => break,
                },
            };
            // Record scan arrival for measuring actual sensor rate
            if let Some(ref stats) = shared_stats {
                stats.record_scan();
            }

            eval.merge(&lifis_tick.runtime_eval);

            #[cfg(feature = "cuda")]
            let cloud_for_reg = LocatedBufferView::copy_from_slice_cuda(
                &cuda_stream,
                lifis_tick.deskewed_cloud.as_slice(),
            );
            #[cfg(not(feature = "cuda"))]
            let cloud_for_reg = LocatedBufferView::Ram(lifis_tick.deskewed_cloud);

            let cloud_out = lifis_tick.deskewed_cloud_passthrough;

            let orig_frame = lifis_tick.cloud_header.frame_id.clone();
            let orig_time = lifis_tick.cloud_header.stamp;
            let current_cloud_time = timestamp_to_secs(&lifis_tick.odom_time);
            if let Some(existing_last_scan_time) = last_scan_time {
                if current_cloud_time < existing_last_scan_time {
                    let mut map = map.lock().unwrap();
                    map.clear_localmap();

                    info!("Resetting map");
                    t_world = nalgebra::Isometry3::identity();
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
                    let initial_guess_translation = t_last_current.translation;
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

            let init_guess_world = t_world * init_guess;

            let (register_pose, ekf_state) = {
                let mut map = map.lock().unwrap();
                match map.register_and_map(
                    lifis_tick.map_only,
                    ext,
                    &cloud_for_reg,
                    current_cloud_time,
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

            // --- Adapting Coordinate Systems ---
            #[cfg(feature = "rerun")]
            if let Some(ref rec) = rec {
                rec.set_timestamp_secs_since_epoch("sensor", current_cloud_time);
            }

            #[cfg(feature = "rerun")]
            let ros_to_rerun = Isometry3::<f64>::identity(); // placeholder for other visualizations if needed

            let old_t_world = t_world;

            // TODO maybe not inverse when not transpose the same result
            t_world = if config.lidar_as_map_tf {
                ekf_state.lidar_imu_extrinsic.inverse() * register_pose.pose
            } else {
                register_pose.pose
            };

            // optional transform from global_lidar to global_imu
            // t_world = corr_t_l_i.cast() * register_pose.pose;

            // transformation from the last pose to the current pose in the local frame
            t_last_current = old_t_world.inverse() * t_world;

            let mut pelorus_odom = Odometry::default();
            pelorus_odom.header.stamp = Time {
                sec: lifis_tick.odom_time.sec,
                nanosec: lifis_tick.odom_time.nanosec,
            };
            pelorus_odom.header.frame_id = config.prelio_params.map_frame.clone();
            pelorus_odom.child_frame_id = orig_frame.clone();
            pelorus_odom.pose.pose.position.x = t_world.translation.vector.x;
            pelorus_odom.pose.pose.position.y = t_world.translation.vector.y;
            pelorus_odom.pose.pose.position.z = t_world.translation.vector.z;
            pelorus_odom.pose.pose.orientation.x = t_world.rotation.i;
            pelorus_odom.pose.pose.orientation.y = t_world.rotation.j;
            pelorus_odom.pose.pose.orientation.z = t_world.rotation.k;
            pelorus_odom.pose.pose.orientation.w = t_world.rotation.w;
            {
                // TODO covariances probably not in the right frame right now and ignores extrinsic rotation
                let (pose_cov_iekf, twist_cov_iekf) = {
                    let map = pipeline.map();
                    let guard = map.lock().unwrap();
                    let (pose_cov, _lv_cov_unused) = guard.get_odometry_covariances();
                    let tw_cov = guard.get_twist_covariance_6x6();
                    (pose_cov, tw_cov)
                };

                #[cfg(not(feature = "r2r"))]
                {
                    pelorus_odom.pose.covariance = pose_cov_iekf;
                    pelorus_odom.twist.covariance = twist_cov_iekf;
                }

                #[cfg(feature = "r2r")]
                {
                    pelorus_odom.pose.covariance = pose_cov_iekf.to_vec();
                    pelorus_odom.twist.covariance = twist_cov_iekf.to_vec();
                }
            }

            // publish paths but only add new poses if the robot moved enough
            #[cfg(not(feature = "no-viz"))]
            let changed_lidar_path = {
                match lidar_path_poses.back() {
                    Some(last_pose) => {
                        let last_pose = last_pose.pose.clone();
                        let last_pose_position = last_pose.position;
                        let current_pose_position = pelorus_odom.pose.pose.position.clone();

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

                        // Check rotational difference
                        let last_quat = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                            last_pose.orientation.w,
                            last_pose.orientation.x,
                            last_pose.orientation.y,
                            last_pose.orientation.z,
                        ));
                        let current_quat =
                            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                                pelorus_odom.pose.pose.orientation.w,
                                pelorus_odom.pose.pose.orientation.x,
                                pelorus_odom.pose.pose.orientation.y,
                                pelorus_odom.pose.pose.orientation.z,
                            ));
                        let rotation_angle = last_quat.angle_to(&current_quat);

                        if distance > path_update_translate_threshold
                            || rotation_angle > path_update_rotate_threshold
                        {
                            lidar_path_poses.push_back(PoseStamped {
                                pose: pelorus_odom.pose.pose.clone(),
                                header: pelorus_odom.header.clone(),
                            });
                            if lidar_path_poses.len() > PATH_LIMIT {
                                lidar_path_poses.pop_front();
                            }
                            true
                        } else {
                            false
                        }
                    }
                    None => {
                        lidar_path_poses.push_back(PoseStamped {
                            pose: pelorus_odom.pose.pose.clone(),
                            header: pelorus_odom.header.clone(),
                        });
                        if lidar_path_poses.len() > PATH_LIMIT {
                            lidar_path_poses.pop_front();
                        }
                        true
                    }
                }
            };

            let mut cloud: PointCloud2 = {
                match (cloud_out, cloud_for_reg) {
                    (None, LocatedBufferView::Ram(items)) => {
                        let msg = PointCloud2Msg::try_from_vec(items).expect("msg");
                        #[cfg(feature = "ros2-client")]
                        let out = impl_ros2_interfaces_jazzy_serde::from_pointcloud2_msg(msg);
                        #[cfg(feature = "ratpub")]
                        let out = impl_ros2_interfaces_jazzy_rkyv::from_pointcloud2_msg(msg);
                        #[cfg(feature = "r2r")]
                        let out = impl_r2r::from_pointcloud2_msg(msg);

                        out
                    }
                    // ignore downsampled one if we have the raw
                    (Some(items), _) => {
                        let msg = PointCloud2Msg::try_from_vec(items).expect("msg");
                        #[cfg(feature = "ros2-client")]
                        let out = impl_ros2_interfaces_jazzy_serde::from_pointcloud2_msg(msg);
                        #[cfg(feature = "ratpub")]
                        let out = impl_ros2_interfaces_jazzy_rkyv::from_pointcloud2_msg(msg);
                        #[cfg(feature = "r2r")]
                        let out = impl_r2r::from_pointcloud2_msg(msg);

                        out
                    }
                    #[cfg(feature = "cuda")]
                    (
                        None,
                        LocatedBufferView::Cuda {
                            buffer: cloud_for_reg,
                        },
                    ) => {
                        let buff = cloud_for_reg.into_cloud(&cuda_stream);
                        let msg = ros_pointcloud2::PointCloud2Msg::try_from_vec(buff).unwrap();
                        #[cfg(feature = "ros2-client")]
                        let out = impl_ros2_interfaces_jazzy_serde::from_pointcloud2_msg(msg);
                        #[cfg(feature = "ratpub")]
                        let out = impl_ros2_interfaces_jazzy_rkyv::from_pointcloud2_msg(msg);
                        #[cfg(feature = "r2r")]
                        let out = impl_r2r::from_pointcloud2_msg(msg);

                        out
                    }
                }
            };

            cloud.header.frame_id = orig_frame.clone();
            cloud.header.stamp = Time {
                sec: lifis_tick.odom_time.sec,
                nanosec: lifis_tick.odom_time.nanosec,
            };
            if let Some(mut trajectory_file) = trajectory_file.as_mut() {
                let t = &t_world.translation.vector;
                let q = t_world.rotation.quaternion();

                // <time in seconds> <translation_x> <translation_y> <translation_z> <quaternion_x> <quaternion_y> <quaternion_z> <quaternion_w>
                // Format the string to write to the file
                let content_to_write = format!(
                    "{} {} {} {} {} {} {} {}\n",
                    current_cloud_time, t.x, t.y, t.z, q.i, q.j, q.k, q.w
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
                let total_map = pelorus_cuda::get_map();
                #[cfg(feature = "rerun")]
                if let Some(ref rec) = rec {
                    let points = total_map.iter().map(|v| {
                        // TODO maybe not inverse(), see above in lidar_as_map_tf
                        let v = ros_to_rerun.cast()
                            * ekf_state
                                .lidar_imu_extrinsic
                                .inverse()
                                .cast()
                                .transform_point(v);
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
                                let p = corr_t_l_i.transform_point(&p);
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
                        x: t_world.translation.vector.x as f64,
                        y: t_world.translation.vector.y as f64,
                        z: t_world.translation.vector.z as f64,
                    },
                    rotation: Quaternion {
                        x: t_world.rotation.quaternion().i as f64,
                        y: t_world.rotation.quaternion().j as f64,
                        z: t_world.rotation.quaternion().k as f64,
                        w: t_world.rotation.quaternion().w as f64,
                    },
                },
            };
            #[cfg(feature = "ratpub")]
            {
                let inner_tf_broad = std::sync::Arc::clone(&tf_broadcaster);
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
            if let Some(ref rec) = rec {
                let t_world_rerun = ros_to_rerun.cast() * t_world;
                let pose = rerun::Transform3D::from_translation(glam::Vec3::new(
                    t_world_rerun.translation.x as f32,
                    t_world_rerun.translation.y as f32,
                    t_world_rerun.translation.z as f32,
                ))
                .with_quaternion(glam::Quat::from_xyzw(
                    t_world_rerun.rotation.quaternion().i as f32,
                    t_world_rerun.rotation.quaternion().j as f32,
                    t_world_rerun.rotation.quaternion().k as f32,
                    t_world_rerun.rotation.quaternion().w as f32,
                ));

                if let Err(err) = rec.log("world/lidar", &pose) {
                    error!("Error sending to rerun {err}");
                }
            }

            #[cfg(all(feature = "mapping", feature = "rerun"))]
            if let Some(ref sender) = out_chan_sender {
                let ps: ros_pointcloud2::PointCloud2Msg = cloud.clone().into();
                let reg_out = RegisteredCloud {
                    cloud: ps,
                    registration: map_to_lidar_transform,
                    timestamp: current_cloud_time,
                };

                let _ = sender.send(reg_out); // currently ignore res
            }

            let mut imu_odom = pelorus_odom.clone();
            imu_odom.pose.pose.position.x = init_guess_world.translation.vector.x;
            imu_odom.pose.pose.position.y = init_guess_world.translation.vector.y;
            imu_odom.pose.pose.position.z = init_guess_world.translation.vector.z;
            imu_odom.pose.pose.orientation.x = init_guess_world.rotation.quaternion().i;
            imu_odom.pose.pose.orientation.y = init_guess_world.rotation.quaternion().j;
            imu_odom.pose.pose.orientation.z = init_guess_world.rotation.quaternion().k;
            imu_odom.pose.pose.orientation.w = init_guess_world.rotation.quaternion().w;
            imu_odom.pose.covariance = pelorus_odom.pose.covariance.clone();
            imu_odom.twist.covariance = pelorus_odom.twist.covariance.clone();

            // #[cfg(not(feature = "no-viz"))]
            // imu_path_poses.push_back(PoseStamped {
            //     pose: imu_odom.pose.pose.clone(),
            //     header: imu_odom.header.clone(),
            // });
            // #[cfg(not(feature = "no-viz"))]
            // if imu_path_poses.len() > PATH_LIMIT {
            //     imu_path_poses.pop_front();
            // }

            #[cfg(feature = "rerun")]
            if let Some(ref rec) = rec {
                #[cfg(feature = "ros2-client")]
                let ps = impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(cloud.clone());
                #[cfg(feature = "ratpub")]
                let ps = impl_ros2_interfaces_jazzy_rkyv::to_pointcloud2_msg(cloud.clone());
                #[cfg(feature = "r2r")]
                let ps = impl_r2r::to_pointcloud2_msg(cloud.clone());

                let gradient = colorous::TURBO;
                let (positions, colors): (Vec<_>, Vec<_>) = ps
                    .try_into_iter()
                    .unwrap()
                    .map(|v: PointXYZI| {
                        let color = gradient.eval_continuous(v.intensity as f64 / 255.0);
                        let v = if config.rerun_undistorted_static_world {
                            t_world.cast() * nalgebra::Point3::new(v.x, v.y, v.z)
                        } else {
                            nalgebra::Point3::new(v.x, v.y, v.z)
                        };
                        let pos = glam::Vec3::new(v.x as f32, v.y as f32, v.z as f32);
                        let color_rgb = [color.r, color.g, color.b];

                        (pos, color_rgb)
                    })
                    .unzip();

                let topic = if config.rerun_undistorted_static_world {
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

            #[cfg(feature = "ratpub")]
            rt.as_ref()
                .expect("rt always exists with ratpub feat")
                .block_on(async {
                    cloud_publisher
                        .publish(&cloud)
                        .await
                        .expect("Failed to publish cloud");

                    odom_publisher
                        .publish(&pelorus_odom)
                        .await
                        .expect("Failed to publish odom");
                });
            #[cfg(not(feature = "ratpub"))]
            {
                cloud_publisher
                    .publish(&cloud)
                    .expect("Failed to publish cloud");

                odom_publisher
                    .publish(&pelorus_odom)
                    .expect("Failed to publish odom");
            }

            #[cfg(not(feature = "no-viz"))]
            if changed_lidar_path {
                #[cfg(all(not(feature = "no-viz"), feature = "rerun"))]
                if let Some(ref rec) = rec {
                    // Visualize the lidar path as a line strip
                    let lidar_path_strip =
                        rerun::LineStrip3D::from_iter(lidar_path_poses.iter().map(|p| {
                            glam::Vec3::new(
                                p.pose.position.x as f32,
                                p.pose.position.y as f32,
                                p.pose.position.z as f32,
                            )
                        }));
                    let lidar_path_strips =
                        rerun::LineStrips3D::new(std::iter::once(lidar_path_strip))
                            .with_colors([rerun::Color::from_rgb(255, 128, 0)]); // Orange color for lidar path

                    if let Err(err) = rec.log("world/lidar_path", &lidar_path_strips) {
                        error!("Error sending lidar path to rerun {err}");
                    }
                }
                #[cfg(all(not(feature = "no-viz"), not(feature = "ratpub")))]
                {
                    let lidar_path = Path {
                        header: Header {
                            frame_id: lidar_path_frame_id.clone(),
                            stamp: pelorus_odom.header.stamp.clone(),
                        },
                        poses: lidar_path_poses.iter().cloned().collect(),
                    };
                    lidar_path_publisher
                        .publish(&lidar_path)
                        .expect("Failed to publish path");
                }
            }

            #[cfg(all(not(feature = "no-viz"), not(feature = "ratpub")))]
            imu_odom_publisher
                .publish(&imu_odom)
                .expect("Failed to publish imu odom");

            // #[cfg(all(not(feature = "no-viz"), not(feature = "ratpub")))]
            // imu_path_publisher
            //     .publish(&imu_path)
            //     .expect("Failed to publish imu path"); // TODO this will spam a lot, maybe also throttle like lidar_path

            #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
            let pub_lidar_path = Path {
                header: Header {
                    frame_id: lidar_path_frame_id.clone(),
                    stamp: pelorus_odom.header.stamp.clone(),
                },
                poses: lidar_path_poses.iter().cloned().collect(),
            };
            // #[cfg(all(not(feature = "no-viz"), feature = "ratpub"))]
            // let pub_imu_path = Path {
            //     header: Header {
            //         frame_id: imu_path_frame_id.clone(),
            //         stamp: imu_odom.header.stamp.clone(),
            //     },
            //     poses: imu_path_poses.iter().cloned().collect(),
            // };
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

                    // imu_path_publisher
                    //     .publish(&pub_imu_path)
                    //     .await
                    //     .expect("Failed to publish imu path"); // TODO this will spam a lot, maybe also throttle like lidar_path
                });
            // #[cfg(feature = "rerun")]
            // if let Some(ref rec) = rec {
            //     let imu_path_strip =
            //         rerun::LineStrip3D::from_iter(imu_path_poses.iter().map(|p| {
            //             glam::Vec3::new(
            //                 p.pose.position.x as f32,
            //                 p.pose.position.y as f32,
            //                 p.pose.position.z as f32,
            //             )
            //         }));
            //     let imu_path_strips = rerun::LineStrips3D::new(std::iter::once(imu_path_strip));
            //     if let Err(err) = rec.log("imu_path", &imu_path_strips) {
            //         error!("Error sending to rerun {err}");
            //     }
            // }

            let iteration_time_ms = eval.get_sum_of_top_level_times_ms();

            const NAME: &str = "Pelorus";

            // Update shared stats if TUI is active
            if let Some(ref stats) = shared_stats {
                let pos = t_world.translation.vector;
                let q = t_world.rotation.quaternion();
                let lv = ekf_state.linear_velocity;
                let av = ekf_state.angular_velocity;

                stats.update_pose(
                    [pos.x, pos.y, pos.z],
                    [q.w, q.i, q.j, q.k],
                    [lv.x, lv.y, lv.z],
                    [av.x, av.y, av.z],
                );

                // Update bias values
                let ab = ekf_state.accel_bias;
                let gb = ekf_state.gyro_bias;
                stats.update_bias([ab.x, ab.y, ab.z], [gb.x, gb.y, gb.z]);

                // Record computation time for this scan iteration
                stats.record_comp_time(iteration_time_ms);
            }

            // Debug output (non-TUI timing metrics)
            if let Some(debug_interval) = debug_render_interval {
                if last_render.elapsed() > debug_interval {
                    last_render = std::time::Instant::now();
                    if shared_stats.is_none() {
                        println!(
                            "{}[2J{}[1;L{}\n{}",
                            27 as char,
                            27 as char,
                            &NAME,
                            eval.render()
                        );
                    }
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

    // Clone shutdown flag for spinner loop
    #[cfg(any(feature = "ratpub", feature = "r2r"))]
    let spinner_shutdown_flag = shutdown_flag.clone();

    #[cfg(feature = "ratpub")]
    {
        // Check shutdown flag periodically instead of parking forever
        loop {
            if let Some(ref flag) = spinner_shutdown_flag {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    info!("Shutdown flag detected in ratpub spinner");
                    break;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        // Wait for processing thread to finish (with timeout)
        let _ = processing_handle.join();
        return Ok(());
    }

    #[cfg(feature = "ros2-client")]
    {
        // ros2-client doesn't have a simple way to break the spinner, so we run it in a thread
        // and check the shutdown flag periodically
        let spinner_shutdown = shutdown_flag.clone();
        let spinner_handle = std::thread::spawn(move || {
            future::block_on(async {
                match node.spinner().unwrap().spin().await {
                    Err(e) => {
                        error!("Node did not exit successfully {e}");
                    }
                    Ok(_) => {}
                }
            });
        });

        // Monitor shutdown flag and wait for processing to complete
        loop {
            if let Some(ref flag) = spinner_shutdown {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    info!("Shutdown flag detected, waiting for processing thread");
                    break;
                }
            }
            if processing_handle.is_finished() {
                info!("Processing thread finished");
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // Join processing thread (should be quick now)
        let _ = processing_handle.join();
        // Note: spinner_handle may still be running but will be cleaned up on process exit
        // We don't wait for it since ros2-client spinner doesn't have clean shutdown
        return Ok(());
    }

    #[cfg(feature = "r2r")]
    {
        let spin_timeout = std::time::Duration::from_millis(10);
        loop {
            node.spin_once(spin_timeout.clone());
            if let Some(ref flag) = spinner_shutdown_flag {
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    info!("Shutdown flag detected in r2r spinner");
                    break;
                }
            }
            if processing_handle.is_finished() {
                info!("Processing thread finished");
                break;
            }
        }
        // Wait for processing thread to finish
        let _ = processing_handle.join();
        return Ok(());
    }

    #[cfg(not(any(feature = "ratpub", feature = "ros2-client", feature = "r2r")))]
    {
        let _ = processing_handle.join();
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize our custom logger that works with both TUI and console
    init_logger();

    let args = Args::parse();

    const DEFAULT_CONFIG_STR: &str = include_str!("../../default.rl");

    match args.command {
        Command::Completions { shell } => {
            let mut cmd = Args::command();
            let bin_name = cmd.get_name().to_string();
            generate(shell, &mut cmd, bin_name, &mut std::io::stdout());
            Ok(())
        }
        Command::Run(args) => {
            match args {
                RunArgs::Online(config_args) => {
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

                    let config = ast_to_config(default_ast, user_ast)?;

                    // Update log level after config has set RUST_LOG
                    update_log_level();

                    let mut show_tui = config.show_tui;

                    if show_tui && !std::io::stdout().is_terminal() {
                        warn!("TUI requested but stdout is not a terminal. Disabling TUI.");
                        show_tui = false;
                    }

                    if show_tui {
                        let shared_stats = std::sync::Arc::new(SharedStats::new());
                        shared_stats.set_mode("Online");
                        let stats_for_tui = shared_stats.clone();

                        shared_stats.set_target_lidar_rate(10.0); // will be updated when first scan arrives

                        let (tui_stop_tx, tui_stop_rx) = std::sync::mpsc::channel();
                        let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel();

                        let tui_thread = std::thread::spawn(move || {
                            let mut tui = StatsTui::new(stats_for_tui)
                                .with_stop_receiver(tui_stop_rx)
                                .with_shutdown_sender(shutdown_tx);
                            let _ = tui.run();
                        });
                        let lio_config = config.clone();
                        let lio_export = config_args.export_trajectory;
                        let lio_sync = config_args.rat;
                        let lio_stats = Some(shared_stats);

                        let lio_shutdown_flag =
                            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                        let lio_shutdown_flag_for_thread = lio_shutdown_flag.clone();

                        let lio_handle = std::thread::spawn(move || -> Result<(), String> {
                            lio_with_ready_signal(
                                lio_config,
                                None,
                                lio_export,
                                lio_sync,
                                lio_stats,
                                Some(lio_shutdown_flag_for_thread),
                            )
                            .map_err(|e| e.to_string())
                        });

                        // Wait for either TUI shutdown or LIO completion
                        loop {
                            if shutdown_rx.try_recv().is_ok() {
                                // TUI requested shutdown -> request LIO to stop
                                lio_shutdown_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                                info!("received shutdown from TUI");
                                break;
                            }
                            if lio_handle.is_finished() {
                                // LIO terminated -> request TUI to stop
                                let _ = tui_stop_tx.send(());
                                info!("lio finished, stopping tui");
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(100));
                        }

                        // Ensure both threads are asked to stop
                        let _ = tui_stop_tx.send(());
                        lio_shutdown_flag.store(true, std::sync::atomic::Ordering::SeqCst);

                        // Join TUI thread first (should be quick)
                        let tui_join_result = tui_thread.join();
                        if tui_join_result.is_err() {
                            warn!("TUI thread panicked");
                        }

                        // Get LIO result
                        match lio_handle.join() {
                            Ok(Ok(())) => Ok(()),
                            Ok(Err(e)) => Err(anyhow!("{}", e).into()),
                            Err(_) => Err(anyhow!("LIO thread panicked").into()),
                        }
                    } else {
                        lio(config, config_args.export_trajectory, config_args.rat)
                    }
                }
                #[cfg(feature = "rat")]
                RunArgs::Offline(offline_args) => {
                    let shared_stats = std::sync::Arc::new(SharedStats::new());
                    shared_stats.set_mode("Offline");
                    let stats_for_tui = shared_stats.clone();
                    // Set default target rate, will be updated by measured rate
                    shared_stats.set_target_lidar_rate(10.0);

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
                        let mut config = ast_to_config(default_ast, user_ast)?;
                        config.show_tui = true; 
                        config
                    };

                    let show_tui = config.show_tui && std::io::stdout().is_terminal();
                    if !show_tui && config.show_tui {
                         warn!("TUI requested for offline mode but stdout is not a terminal. Continuing without visualization.");
                    }

                    // Update log level after config has set RUST_LOG
                    update_log_level();

                    let lidar_topic = &config.prelio_node_params.cloud_src_topic;
                    let imu_topic = &config.prelio_node_params.imu_src_topic;
                    let imu_init_time_ms =
                        (config.prelio_node_params.imu_predictor_init_time_sec * 1000.0) as u64;
                    let scan_init_samples = config.prelio_params.initial_scan_accumulate;
                    let bagfile_path_canon = offline_args.bagfile_path.canonicalize().unwrap();
                    let bagfile_path = bagfile_path_canon.clone().to_string_lossy().to_string();
                    shared_stats.set_bagfile(
                        bagfile_path_canon
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy(),
                    );

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
                    
                    // Start TUI immediately so we can see what's happening
                    let (tui_stop_tx, tui_stop_rx) = std::sync::mpsc::channel();
                    let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel();

                    let tui_thread = if show_tui {
                        Some(std::thread::spawn(move || {
                            let mut tui = StatsTui::new(stats_for_tui)
                                .with_stop_receiver(tui_stop_rx)
                                .with_shutdown_sender(shutdown_tx);
                            let _ = tui.run();
                        }))
                    } else {
                        None
                    };

                    // Define the main logic as a closure/block to easily handle errors and cleanup
                    let run_offline_logic = || -> anyhow::Result<()> {
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
                    let (exit_tx, exit_rx) = std::sync::mpsc::channel();

                    // Thread for stdout
                    let exit_tx_clone = exit_tx.clone();
                    let ready_tx_clone = ready_tx.clone();
                    std::thread::spawn(move || {
                        let reader = BufReader::new(minot_stdout);
                        for line in reader.lines() {
                            match line {
                                Ok(line) => {
                                    info!("[Minot] {}", line);

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
                                    error!("[Minot] Error reading stdout: {}", e);
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
                                    if line.to_lowercase().contains("error") {
                                        error!("[Minot] {}", line);
                                    } else {
                                        info!("[Minot] {}", line);
                                    }

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
                                    error!("[Minot] Error reading stderr: {}", e);
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

                    // start Odometry in thread
                    // Provide an atomic shutdown flag so we can request LIO to shutdown from main/TUI
                    let lio_shutdown_flag =
                        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let lio_shutdown_flag_for_thread = lio_shutdown_flag.clone();
                    let shared_stats_for_lio = shared_stats.clone();

                    let lio_handle = std::thread::spawn(move || {
                        if let Err(e) = lio_with_ready_signal(
                            config,
                            Some(pelorus_ready_tx),
                            offline_args.export_trajectory,
                            true,
                            Some(shared_stats_for_lio),
                            Some(lio_shutdown_flag_for_thread),
                        ) {
                            error!("Odometry failed: {}", e);
                            return Err(format!("Odometry failed: {}", e));
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

                    // Check if process is still alive before writing
                    if let Ok(Some(status)) = minot_child.try_wait() {
                         error!("Minot process has already exited with status: {}", status);
                         return Err(anyhow!("Minot exited prematurely: {}", status));
                    }

                    use std::io::Write;
                    minot_stdin
                        .write_all(b"\n")
                        .map_err(|e| anyhow!("Failed to write to minot stdin: {}", e))?;
                    minot_stdin
                        .flush()
                        .map_err(|e| anyhow!("Failed to flush minot stdin: {}", e))?;

                    /* 
                    // TUI is already running from the start of the block
                    let (tui_stop_tx, tui_stop_rx) = std::sync::mpsc::channel();
                    let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel();
                    
                    let tui_thread = std::thread::spawn(move || {
                        let mut tui = StatsTui::new(
                            stats_for_tui,
                        )
                        .with_stop_receiver(tui_stop_rx)
                        .with_shutdown_sender(shutdown_tx);
                        let _ = tui.run();
                    });
                    */

                    // wait loop - check for various exit conditions
                    loop {
                        if exit_rx.try_recv().is_ok() {
                            // minot finished -> request stop
                            lio_shutdown_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                            info!("minot finished, stopping TUI and LIO");
                            break;
                        }
                        if lio_handle.is_finished() {
                            log::info!("LIO finished, stopping TUI");
                            break;
                        }
                        if shutdown_rx.try_recv().is_ok() {
                            // TUI requested shutdown -> request LIO to stop
                            lio_shutdown_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                            info!("TUI requested shutdown, stopping LIO");
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    
                    // Cleanup
                    lio_shutdown_flag.store(true, std::sync::atomic::Ordering::SeqCst);

                    // Join LIO thread with timeout
                    let join_start = std::time::Instant::now();
                    let join_timeout = std::time::Duration::from_secs(3);
                    loop {
                        if lio_handle.is_finished() {
                            match lio_handle.join() {
                                Ok(Ok(())) => info!("LIO thread joined successfully"),
                                Ok(Err(e)) => warn!("LIO thread exited with error: {}", e),
                                Err(_) => warn!("LIO thread panicked"),
                            }
                            break;
                        }
                        if join_start.elapsed() > join_timeout {
                            warn!("LIO thread did not exit within timeout");
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    };
                    
                    // Kill minot just in case
                    let _ = minot_child.kill();
                    
                    Ok(())
                    };

                    match run_offline_logic() {
                        Ok(_) => {
                            let _ = tui_stop_tx.send(());
                            if let Some(handle) = tui_thread {
                                let _ = handle.join();
                            }
                            Ok(())
                        }
                        Err(e) => {
                            error!("Offline run failed: {}", e);
                            std::thread::sleep(std::time::Duration::from_secs(2)); // Give time for user to read error
                            let _ = tui_stop_tx.send(());
                            if let Some(handle) = tui_thread {
                                let _ = handle.join();
                            }
                            Err(e.into())
                        }
                    }
                }
            }
        }
        Command::Config(cmd) => match cmd {
            ConfigCommand::Generate(args) => {
                match args.kind {
                    GenerateKind::Full => {
                        print!("{}", DEFAULT_CONFIG_STR);
                        Ok(())
                    }
                    GenerateKind::Recommended => {
                        let mut default_ast =
                            ratslang::compile_code(DEFAULT_CONFIG_STR).map_err(|e| {
                                anyhow!("Failed to compile embedded default config: {}", e)
                            })?;

                        // Suppress unused-variable warnings when generating recommended config
                        default_ast.vars = default_ast.vars.with_drop_warning(false);

                        match default_ast.to_string_filtered("recommended") {
                            Some(recommended_config) => {
                                print!("{}", recommended_config);
                                Ok(())
                            }
                            None => Err(anyhow!(
                                "No @recommended annotations found in the default configuration"
                            )
                            .into()),
                        }
                    }
                }
            }
            ConfigCommand::Verify(args) => {
                let default_ast = ratslang::compile_code(DEFAULT_CONFIG_STR)
                    .map_err(|e| anyhow!("Failed to compile embedded default config: {}", e))?;

                let user_ast = if args.config_path.to_str() == Some("-") {
                    use std::io::Read;
                    let mut buffer = String::new();
                    std::io::stdin()
                        .read_to_string(&mut buffer)
                        .map_err(|e| anyhow!("Failed to read from stdin: {}", e))?;
                    ratslang::compile_code(&buffer)
                        .map_err(|e| anyhow!("Failed to compile config from stdin: {}", e))?
                } else {
                    let path = std::path::Path::new(&args.config_path);
                    if !path.exists() {
                        return Err(anyhow!(
                            "Error: Configuration file not found at '{}'",
                            args.config_path.to_string_lossy()
                        )
                        .into());
                    }
                    ratslang::compile_file(&path.to_path_buf(), None, None).map_err(|e| {
                        anyhow!(
                            "Failed to compile user config '{}': {}",
                            args.config_path.to_string_lossy(),
                            e
                        )
                    })?
                };

                let (_, (assigned_vars, evaluated_vars)) =
                    ast_to_config_with_user_usage(default_ast, user_ast)?;

                // Check for unused variables in user config
                let mut has_warnings = false;
                for var in assigned_vars.iter() {
                    if !evaluated_vars.contains(var) {
                        warn!("Variable '{}' is not recognized by Pelorus", var);
                        has_warnings = true;
                    }
                }

                if !has_warnings {
                    println!("Configuration is valid.");
                }

                Ok(())
            }
        },
        #[cfg(feature = "evaluate")]
        Command::Evaluate(paths) => {
            let estimated_path = std::path::PathBuf::from(&paths.estimated_file);
            let ground_truth_path = std::path::PathBuf::from(&paths.reference_file);
            evaluate::visualize_results(&estimated_path, &ground_truth_path)?;
            Ok(())
        }
    }
}
