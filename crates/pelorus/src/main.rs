use nalgebra::Translation3;
use r2r::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{PoseStamped, Quaternion, Transform, TransformStamped, Vector3},
    nav_msgs::msg::{Odometry, Path},
    sensor_msgs::msg::PointCloud2,
    std_msgs::msg::Header,
    RosParams,
};

use tf_r2r::TfBroadcaster;

use lifis_ros::LifisNode;
use ros_pointcloud2::prelude::*;
use small_gicp_sys::*;

pub fn timestamp_to_secs(t: &TimeMsg) -> f64 {
    t.sec as f64 + t.nanosec as f64 / 1_000_000_000.0
}

pub fn rpcl2_time_to_r2r_time(t: &ros_pointcloud2::ros::TimeMsg) -> TimeMsg {
    TimeMsg {
        sec: t.sec,
        nanosec: t.nanosec,
    }
}

#[derive(Debug, Clone, Default)]
pub enum LioMethod {
    SmallGICPCPU,

    #[default]
    SFasterLioCPU,

    JlioGPU,
}

impl From<&str> for LioMethod {
    fn from(s: &str) -> Self {
        match s {
            "small_gicp_cpu" => LioMethod::SmallGICPCPU,
            "sfasterlio_cpu" => LioMethod::SFasterLioCPU,
            "jlio_gpu" => LioMethod::JlioGPU,
            _ => LioMethod::SFasterLioCPU,
        }
    }
}

#[derive(RosParams, Debug, Clone)]
pub struct LioConfigParse {
    pub feedback_fps: u32,
    pub map_frame: String,
    pub registered_cloud_qos: String, // sensor | system_default
    pub lidar_qos: String,
    pub imu_qos: String,
    pub odom_qos: String,
    pub imu_odom_qos: String,
    pub imu_path_qos: String,
    pub path_qos: String,
    pub map_resolution: f64,
    pub method: String,
}

impl Default for LioConfigParse {
    fn default() -> Self {
        Self {
            feedback_fps: 1,
            map_frame: "map".to_string(),
            registered_cloud_qos: "system_default".to_string(),
            odom_qos: "system_default".to_string(),
            imu_odom_qos: "system_default".to_string(),
            imu_path_qos: "system_default".to_string(),
            path_qos: "system_default".to_string(),
            imu_qos: "sensor".to_string(),
            lidar_qos: "sensor".to_string(),
            map_resolution: 0.15,
            method: "sfasterlio_cpu".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LioConfig {
    pub feedback_fps: u32,
    pub registered_cloud_qos: r2r::QosProfile,
    pub lidar_qos: r2r::QosProfile,
    pub imu_qos: r2r::QosProfile,
    pub odom_qos: r2r::QosProfile,
    pub imu_odom_qos: r2r::QosProfile,
    pub imu_path_qos: r2r::QosProfile,
    pub path_qos: r2r::QosProfile,
    pub map_resolution: f64,
}

fn parse_qos(qos: &str) -> r2r::QosProfile {
    match qos {
        "sensor" => r2r::QosProfile::sensor_data(),
        "system_default" => r2r::QosProfile::default(),
        _ => r2r::QosProfile::default(),
    }
}

impl From<LioConfigParse> for LioConfig {
    fn from(parse: LioConfigParse) -> Self {
        Self {
            feedback_fps: parse.feedback_fps,
            map_resolution: parse.map_resolution,
            registered_cloud_qos: parse_qos(&parse.registered_cloud_qos),
            lidar_qos: parse_qos(&parse.lidar_qos),
            imu_qos: parse_qos(&parse.imu_qos),
            odom_qos: parse_qos(&parse.odom_qos),
            imu_odom_qos: parse_qos(&parse.imu_odom_qos),
            imu_path_qos: parse_qos(&parse.imu_path_qos),
            path_qos: parse_qos(&parse.path_qos),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "lio", "lio")?;
    let log_prefix = "lio";
    let config = std::sync::Arc::new(std::sync::Mutex::new(LioConfigParse::default()));
    let (_, _) = node.make_derived_parameter_handler(config.clone())?;
    let config = {
        let config = config.lock().unwrap();
        config.clone()
    };
    let config = LioConfig::from(config);

    let tf_broadcaster = TfBroadcaster::new(&mut node);

    let mut lifis_node = LifisNode::create(lifis_ros::LifisNodeParameters {
        node_name: "lifis",
        imu_qos: config.imu_qos,
        cloud_qos: config.lidar_qos,
        ..Default::default()
    })
    .await?;

    let map_frame = {
        lifis_node
            .state
            .lock()
            .unwrap()
            .config
            .lock()
            .unwrap()
            .map_frame
            .clone()
    };

    let cloud_publisher =
        node.create_publisher::<PointCloud2>("registered", config.registered_cloud_qos)?;
    let odom_publisher = node.create_publisher::<Odometry>("odom", config.odom_qos)?;
    let lidar_path_publisher = node.create_publisher::<Path>("path", config.path_qos)?;
    let imu_odom_publisher = node.create_publisher::<Odometry>("imu_odom", config.imu_odom_qos)?;
    let imu_path_publisher = node.create_publisher::<Path>("imu_path", config.imu_path_qos)?;

    let mut map = Map::new(config.map_resolution);

    #[cfg(debug_assertions)]
    r2r::log_warn!(log_prefix, "Running in dev mode.");

    let mut t_world_lidar = nalgebra::Isometry3::identity();
    let mut t_last_current = nalgebra::Isometry3::identity();

    let mut lidar_path = Path {
        header: Default::default(),
        poses: vec![],
    };

    lidar_path.header.frame_id = map_frame.clone();

    let mut imu_path = Path {
        header: Default::default(),
        poses: vec![],
    };

    imu_path.header.frame_id = map_frame.clone();

    let mut first_scan = true;
    let mut last_scan_time = None;

    let mut avg_iterations = 0.0;
    let mut iteration_updates = 0;

    let path_update_translate_threshold = 0.0; // default 0.15 meters - TODO config

    let mut eval = lifis_ros::lifis::timing::RuntimeEvaluator::new();

    let render_interval = if config.feedback_fps == 0 {
        None
    } else {
        Some(std::time::Duration::from_secs_f64(
            1.0 / config.feedback_fps as f64,
        ))
    };
    let mut last_render = std::time::Instant::now();

    loop {
        node.spin_once(std::time::Duration::from_millis(100));
        match lifis_node.tick().await {
            Ok(Some(lifis_tick)) => {
                eval.start("total").unwrap();
                eval.merge(&lifis_tick.runtime_eval);

                let cloud: PointCloud2Msg = lifis_tick.deskewed_cloud.into();
                let orig_frame = cloud.header.frame_id.clone();
                let orig_time = cloud.header.stamp.clone();
                let current_cloud_time = timestamp_to_secs(&orig_time);
                if let Some(existing_last_scan_time) = last_scan_time {
                    if current_cloud_time < existing_last_scan_time {
                        map.reset(config.map_resolution);
                        r2r::log_info!(log_prefix, "Resetting map");
                        first_scan = true;
                        t_world_lidar = nalgebra::Isometry3::identity();
                        t_last_current = nalgebra::Isometry3::identity();
                    }
                }

                last_scan_time = Some(timestamp_to_secs(&orig_time));

                if first_scan {
                    map.add(&cloud, None);
                    first_scan = false;
                    eval.stop("total").unwrap();
                    // TODO: rpcl2 - from and into for timemsg
                    continue;
                }

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

                    nalgebra::Isometry3::from_parts(
                        initial_guess_translation,
                        initial_guess_rotation,
                    )
                };

                let init_guess_world = t_world_lidar * init_guess;

                eval.start("reg").unwrap();
                let register_pose = map.register(&cloud, &init_guess_world);
                eval.stop("reg").unwrap();

                avg_iterations = (avg_iterations * iteration_updates as f64
                    + register_pose.num_iterations as f64)
                    / (iteration_updates as f64 + 1.0);
                iteration_updates += 1;

                // Transformation from the last pose to the current pose in the local frame.
                t_last_current = t_world_lidar.inverse() * register_pose.transform_target_to_source;
                t_world_lidar = register_pose.transform_target_to_source;

                eval.start("map_update").unwrap();
                map.add(&cloud, Some(register_pose.transform_target_to_source));
                eval.stop("map_update").unwrap();

                let mut lidar_odom = lifis_tick.imu_odom.clone();
                lidar_odom.header.frame_id = map_frame.clone();
                lidar_odom.child_frame_id = cloud.header.frame_id.clone();
                lidar_odom.pose.pose.position.x = t_world_lidar.translation.vector.x;
                lidar_odom.pose.pose.position.y = t_world_lidar.translation.vector.y;
                lidar_odom.pose.pose.position.z = t_world_lidar.translation.vector.z;
                lidar_odom.pose.pose.orientation.x = t_world_lidar.rotation.quaternion().i;
                lidar_odom.pose.pose.orientation.y = t_world_lidar.rotation.quaternion().j;
                lidar_odom.pose.pose.orientation.z = t_world_lidar.rotation.quaternion().k;
                lidar_odom.pose.pose.orientation.w = t_world_lidar.rotation.quaternion().w;

                let changed_lidar_path = match lidar_path.poses.last() {
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

                let mut cloud: PointCloud2 = transform_cloud(cloud, &t_world_lidar).into();

                let map_to_lidar_transform = t_world_lidar.clone();
                cloud.header.frame_id = map_frame.clone();

                let tf = TransformStamped {
                    header: Header {
                        frame_id: map_frame.to_string(),
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

                tf_broadcaster
                    .send_transform(tf)
                    .expect("Failed to send transform");

                cloud_publisher
                    .publish(&cloud)
                    .expect("Failed to publish cloud");

                odom_publisher
                    .publish(&lidar_odom)
                    .expect("Failed to publish odom");

                if changed_lidar_path {
                    lidar_path_publisher
                        .publish(&lidar_path)
                        .expect("Failed to publish path");
                }

                imu_odom_publisher
                    .publish(&imu_odom)
                    .expect("Failed to publish imu odom");

                imu_path_publisher
                    .publish(&imu_path)
                    .expect("Failed to publish imu path"); // TODO this will spam a lot, maybe also throttle like lidar_path

                eval.stop("total").unwrap();

                if let Some(render_interval) = render_interval {
                    if last_render.elapsed() > render_interval {
                        last_render = std::time::Instant::now();
                        r2r::log_info!(
                            log_prefix,
                            "{}[2J{}[1;1H[LIO] - Simple LiDAR Odometry\n{}",
                            27 as char,
                            27 as char,
                            eval.render()
                        );
                    }
                }
            }
            Err(e) => {
                r2r::log_error!("lifis", "{}", e);
            }
            _ => {}
        }
    }
}
