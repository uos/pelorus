use log::{error, info, warn};
use nalgebra::{Isometry3, Translation3};
use r2r::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{Point, PoseStamped, Quaternion, Transform, TransformStamped},
    log_error,
    nav_msgs::msg::{Odometry, Path},
    sensor_msgs::msg::PointCloud2,
    std_msgs::msg::Header,
    visualization_msgs::msg::Marker,
    RosParams,
};

use either::*;
use prelio_ros2::tf::TfBroadcaster;
use rand::Rng;

use pelorus_reg::{sfasterlio::SFasterLio, Registration};
use prelio_ros2::{prelio::imu::ImuMsg, LifisNode};
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

#[allow(dead_code)]
fn lerp_color(c1: (f32, f32, f32), c2: (f32, f32, f32), t: f32) -> (f32, f32, f32) {
    (
        c1.0 * (1.0 - t) + c2.0 * t,
        c1.1 * (1.0 - t) + c2.1 * t,
        c1.2 * (1.0 - t) + c2.2 * t,
    )
}

/*
template<typename T>
auto set_pose6d(const double t, const Matrix<T, 3, 1> &a, const Matrix<T, 3, 1> &g, \
                const Matrix<T, 3, 1> &v, const Matrix<T, 3, 1> &p, const Matrix<T, 3, 3> &R)
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++)  rot_kp.rot[i*3+j] = R(i,j);
    }
    return move(rot_kp);
}
*/

fn lio() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "lio", "lio")?;
    let log_prefix = "pelorus_lio";
    let config = std::sync::Arc::new(std::sync::Mutex::new(LioConfigParse::default()));
    let (_, _) = node.make_derived_parameter_handler(config.clone())?;
    let config = {
        let config = config.lock().unwrap();
        config.clone()
    };
    let config = LioConfig::from(config);

    let tf_broadcaster = TfBroadcaster::new(&mut node);

    let imu_params = pelorus_reg::imu::ImuParameters::default();

    let gravity = 9.81;

    //let map = Arc::new(Mutex::new(Map::new(config.map_resolution)));
    let map = Arc::new(Mutex::new(SFasterLio::new(
        config.map_resolution,
        gravity,
        &imu_params,
        0.001,
        4,
        5,
        5.0,
        false,
        Isometry3::identity(),
        pelorus_reg::IVoxNearbyType::Nearby18,
    )));

    let mean_acc = Arc::new(Mutex::new(None));

    // Initialize the EKF with the first IMU messages
    let imu_pose_predictor_init = {
        let map = map.clone();
        let mean_acc = mean_acc.clone();
        Arc::new(Mutex::new(move |msg: &[ImuMsg]| {
            let msg: Vec<pelorus_reg::ImuMsg> = msg
                .iter()
                .map(|m| pelorus_reg::ImuMsg {
                    timestamp_sec: prelio_ros2::prelio::timestamp_to_secs(&m.stamp),
                    orientation: nalgebra::UnitQuaternion::from_quaternion(
                        nalgebra::Quaternion::new(
                            m.orientation[0],
                            m.orientation[1],
                            m.orientation[2],
                            m.orientation[3],
                        ),
                    ),
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
                })
                .collect();
            let mut map = map.lock().unwrap();
            let init_mean_acc = map.init_imu_ekf(&msg);
            let mut mean_acc = mean_acc.lock().unwrap();
            *mean_acc = Some(init_mean_acc);
        }))
    };

    let (mut lifis_node, synced_rx) = LifisNode::create(
        prelio_ros2::LifisNodeParameters {
            node_name: "lifis",
            imu_qos: config.imu_qos,
            cloud_qos: config.lidar_qos,
            ..Default::default()
        },
        Some(imu_pose_predictor_init),
        {
            let map = map.clone();

            Some(Box::new(
                move |imu_msgs: &[prelio_ros2::prelio::imu::ImuMsg],
                      last_cloud_end_time,
                      cloud_begin_time,
                      cloud_end_time,
                      cloud| {
                    assert!(!imu_msgs.is_empty());
                    assert!(last_cloud_end_time > 0.0);
                    assert!(cloud_begin_time > 0.0);
                    assert!(cloud_end_time > 0.0);

                    // TODO why even use different ImuMsg types?
                    let imus = imu_msgs.iter().map(|m| pelorus_reg::ImuMsg {
                        timestamp_sec: prelio_ros2::prelio::timestamp_to_secs(&m.stamp),
                        orientation: nalgebra::UnitQuaternion::from_quaternion(
                            nalgebra::Quaternion::new(
                                m.orientation[0],
                                m.orientation[1],
                                m.orientation[2],
                                m.orientation[3],
                            ),
                        ),
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
                    });

                    let imus_vec = imus.collect::<Vec<_>>();

                    let mut map = map.lock().unwrap();
                    let cloud_undistorted = map.undistort(
                        &imus_vec,
                        last_cloud_end_time,
                        cloud_begin_time,
                        cloud_end_time,
                        cloud,
                    );

                    //dbg!(&cloud_undistorted);

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
                        angular_velocity: nalgebra::Vector3::default(),
                        lidar_imu_extrinsic_rotation: nalgebra::Matrix3::identity(),
                        lidar_imu_extrinsic_translation: nalgebra::Vector3::default(),
                    }
                },
            ))
        },
    )?;

    let map_frame = {
        let lnode = lifis_node.state.lock().unwrap();
        let config = lnode.config.lock().unwrap();
        config.map_frame.clone()
    };

    let cloud_publisher =
        node.create_publisher::<PointCloud2>("registered", config.registered_cloud_qos)?;
    let odom_publisher = node.create_publisher::<Odometry>("odom", config.odom_qos)?;
    let lidar_path_publisher = node.create_publisher::<Path>("path", config.path_qos)?;
    let imu_odom_publisher = node.create_publisher::<Odometry>("imu_odom", config.imu_odom_qos)?;
    let imu_path_publisher = node.create_publisher::<Path>("imu_path", config.imu_path_qos)?;
    let registered_visual =
        node.create_publisher::<Marker>("registered_viz", r2r::QosProfile::default())?;

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

    let mut last_scan_time = None;
    let mut total_viz_marker = 0;

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
        while let Ok(lifis_tick) = synced_rx.recv() {
            dbg!("--- received lifis tick ---");
            eval.start("total").unwrap();
            //eval.merge(&lifis_tick.runtime_eval);

            let cloud = lifis_tick.deskewed_cloud;
            let orig_frame = lifis_tick.cloud_header.frame_id.clone();
            let orig_time = lifis_tick.cloud_header.stamp.clone();
            let current_cloud_time = timestamp_to_secs(&orig_time);
            if let Some(existing_last_scan_time) = last_scan_time {
                if current_cloud_time < existing_last_scan_time {
                    let mut map = map.lock().unwrap();
                    map.clear_localmap();
                    r2r::log_info!(log_prefix, "Resetting map");
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

            //dbg!(init_guess);

            let init_guess_world = t_world_lidar * init_guess;

            eval.start("reg").unwrap();
            let register_pose = {
                let mut map = map.lock().unwrap();
                match map.register_and_map(&cloud, &init_guess_world) {
                    Ok(pose) => pose,
                    Err(err) => {
                        eval.stop("reg").unwrap();
                        log_error!(log_prefix, "{:?}", err);
                        continue;
                    }
                }
            };

            eval.stop("reg").unwrap();

            //avg_iterations = register_pose.avg_iterations.unwrap_or(0) as f32;
            /* (avg_iterations * iteration_updates as f64
            + register_pose.num_iterations as f64)
            / (iteration_updates as f64 + 1.0); */

            //iteration_updates += 1;

            // Transformation from the last pose to the current pose in the local frame.
            t_last_current = t_world_lidar.inverse() * register_pose.pose;
            t_world_lidar = register_pose.pose;

            /*eval.start("map_update").unwrap();
            map.add(&cloud, Some(register_pose.pose));
            eval.stop("map_update").unwrap();*/

            let mut lidar_odom = lifis_tick.imu_odom.clone();
            lidar_odom.header.frame_id = map_frame.clone();
            lidar_odom.child_frame_id = orig_frame.clone();
            lidar_odom.pose.pose.position.x = t_world_lidar.translation.vector.x;
            lidar_odom.pose.pose.position.y = t_world_lidar.translation.vector.y;
            lidar_odom.pose.pose.position.z = t_world_lidar.translation.vector.z;
            lidar_odom.pose.pose.orientation.x = t_world_lidar.rotation.i;
            lidar_odom.pose.pose.orientation.y = t_world_lidar.rotation.j;
            lidar_odom.pose.pose.orientation.z = t_world_lidar.rotation.k;
            lidar_odom.pose.pose.orientation.w = t_world_lidar.rotation.w;

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

            if let Some(registered_normals) = register_pose.registered_cloud {
                let random_color = {
                    let mut rng = rand::thread_rng();
                    let r: f32 = rng.gen();
                    let g: f32 = rng.gen();
                    let b: f32 = rng.gen();
                    let v = nalgebra::Vector3::new(r, g, b).normalize();
                    let a: f32 = 0.7;
                    (v.x, v.y, v.z, a)
                };

                let mut marr = Marker::default();
                marr.header = Header {
                    frame_id: map_frame.clone(),
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

                registered_visual
                    .publish(&marr)
                    .expect("Failed to publish registered visualization");
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
                    translation: r2r::geometry_msgs::msg::Vector3 {
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

            dbg!(&tf);

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
                        "{}[2J{}[1;1HAccelerated LiDAR Odometry Framework\n{}",
                        27 as char,
                        27 as char,
                        eval.render()
                    );
                }
            }
        }
        dbg!("--- END lifis tick ---");
    });

    let spin_timeout = std::time::Duration::from_millis(10);

    loop {
        node.spin_once(spin_timeout.clone());
        lifis_node.spin_once(spin_timeout.clone());
    }
}

#[cfg(debug_assertions)]
fn main() -> color_eyre::eyre::Result<(), Box<dyn std::error::Error>> {
    color_eyre::install()?;
    env_logger::init();
    warn!("Running in dev mode.");
    lio()
}

#[cfg(not(debug_assertions))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    lio()
}
