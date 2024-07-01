use nalgebra::Translation3;
use r2r::{
    geometry_msgs::msg::{PoseStamped, Quaternion, Transform, TransformStamped, Vector3},
    nav_msgs::msg::{Odometry, Path},
    sensor_msgs::msg::PointCloud2,
};

use tf_r2r::TfBroadcaster;

use lifis_ros::LifisNode;
use ros_pointcloud2::prelude::*;
use small_gicp_rs::*;

pub fn timestamp_to_secs(t: &TimeMsg) -> f64 {
    t.sec as f64 + t.nanosec as f64 / 1_000_000_000.0
}

pub fn rpcl2_time_to_r2r_time(t: &ros_pointcloud2::ros::TimeMsg) -> TimeMsg {
    TimeMsg {
        sec: t.sec,
        nanosec: t.nanosec,
    }
}

const FEEDBACK_FPS: u64 = 5;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "lio", "lio")?;

    let tf_broadcaster = TfBroadcaster::new(&mut node);

    let mut lifis_node = LifisNode::create(lifis_ros::LifisNodeParameters {
        node_name: "lifis",
        cloud_src_topic: "ouster/points",
        imu_src_topic: "ouster/imu",
        ..Default::default()
    })
    .await?;

    let cloud_publisher =
        node.create_publisher::<PointCloud2>("registered", r2r::QosProfile::default())?;
    let odom_publisher = node.create_publisher::<Odometry>("odom", r2r::QosProfile::default())?;
    let lidar_path_publisher = node.create_publisher::<Path>("path", r2r::QosProfile::default())?;
    let imu_odom_publisher =
        node.create_publisher::<Odometry>("imu_odom", r2r::QosProfile::default())?;
    let imu_path_publisher =
        node.create_publisher::<Path>("imu_path", r2r::QosProfile::default())?;

    let map_voxel_size = 0.15;
    let mut map = Map::new(map_voxel_size);

    #[cfg(debug_assertions)]
    r2r::log_warn!("lio", "Running in dev mode.");

    let mut t_world_lidar = nalgebra::Isometry3::identity();
    let mut t_last_current = nalgebra::Isometry3::identity();
    let map_frame = "map".to_string();

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

    let mut eval = lifis_ros::timing::RuntimeEvaluator::new();
    let render_interval = std::time::Duration::from_secs_f64(1.0 / FEEDBACK_FPS as f64);
    let mut last_render = std::time::Instant::now();

    loop {
        node.spin_once(std::time::Duration::from_millis(100));
        match lifis_node.tick().await {
            Ok(Some(lifis_tick)) => {
                eval.start("total").unwrap();
                eval.merge(&lifis_tick.runtime_eval);

                let cloud: PointCloud2Msg = lifis_tick.deskewed_cloud.into();
                //let cloud_size = cloud.dimensions.height * cloud.dimensions.width;
                let orig_frame = cloud.header.frame_id.clone();
                let orig_time = cloud.header.stamp.clone();
                let current_cloud_time = timestamp_to_secs(&orig_time);
                if let Some(existing_last_scan_time) = last_scan_time {
                    if current_cloud_time < existing_last_scan_time {
                        map.reset(map_voxel_size);
                        r2r::log_info!("lio", "Resetting map");
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
                    t_last_current
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

                dbg!(init_guess);

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
                    header: cloud.header.clone().into(),
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

                /*if last_render.elapsed() > render_interval {
                    last_render = std::time::Instant::now();
                    r2r::log_info!(
                        "lio",
                        "{}[2J{}[1;1H[LIO] - Simple LiDAR Odometry\n{}",
                        27 as char,
                        27 as char,
                        eval.render()
                    );
                }*/
            }
            Err(e) => {
                r2r::log_error!("lifis", "{}", e);
            }
            _ => {}
        }
    }
}
