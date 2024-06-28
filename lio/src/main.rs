use nalgebra::Translation3;
use r2r::{
    geometry_msgs::msg::{Quaternion, Transform, TransformStamped, Vector3},
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::PointCloud2,
};

use tf_r2r::TfBroadcaster;

use lifis_ros::LifisNode;
use ros_pointcloud2::prelude::*;
use small_gicp_rs::*;

pub fn timestamp_to_secs(t: &TimeMsg) -> f64 {
    t.sec as f64 + t.nanosec as f64 / 1_000_000_000.0
}

const FEEDBACK_FPS: u64 = 5;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "lio", "")?;

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
    let odom_publisher =
        node.create_publisher::<Odometry>("lifis_odom", r2r::QosProfile::default())?;

    let map_voxel_size = 0.15;
    let mut map = Map::new(map_voxel_size);

    #[cfg(debug_assertions)]
    r2r::log_warn!("lio", "Running in dev mode.");

    let mut t_world_lidar = nalgebra::Isometry3::identity();
    let mut t_last_current = nalgebra::Isometry3::identity();
    let map_frame = "map".to_string();

    let mut first_scan = true;
    let mut last_scan_time = None;

    let mut avg_iterations = 0.0;
    let mut iteration_updates = 0;

    let mut eval = lifis_ros::timing::RuntimeEvaluator::new();
    let render_interval = std::time::Duration::from_secs_f64(1.0 / FEEDBACK_FPS as f64);
    let mut last_render = std::time::Instant::now();

    loop {
        node.spin_once(std::time::Duration::from_millis(100));
        match lifis_node.tick().await {
            Ok(Some((orig_cloud, mut odom, lifis_eval))) => {
                eval.start("total").unwrap();
                eval.merge(&lifis_eval);

                let cloud: PointCloud2Msg = orig_cloud.into();
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
                let init_guess = if odom.pose.pose.position.x == 0.0
                    && odom.pose.pose.position.y == 0.0
                    && odom.pose.pose.position.z == 0.0
                {
                    t_last_current
                } else {
                    let initial_guess_translation = Translation3::new(
                        odom.pose.pose.position.x,
                        odom.pose.pose.position.y,
                        odom.pose.pose.position.z,
                    );

                    let initial_guess_rotation =
                        nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                            odom.pose.pose.orientation.w,
                            odom.pose.pose.orientation.x,
                            odom.pose.pose.orientation.y,
                            odom.pose.pose.orientation.z,
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

                odom.header.frame_id = map_frame.clone();
                odom.child_frame_id = cloud.header.frame_id.clone();
                odom.pose.pose.position.x = t_world_lidar.translation.vector.x;
                odom.pose.pose.position.y = t_world_lidar.translation.vector.y;
                odom.pose.pose.position.z = t_world_lidar.translation.vector.z;
                odom.pose.pose.orientation.x = t_world_lidar.rotation.quaternion().i;
                odom.pose.pose.orientation.y = t_world_lidar.rotation.quaternion().j;
                odom.pose.pose.orientation.z = t_world_lidar.rotation.quaternion().k;
                odom.pose.pose.orientation.w = t_world_lidar.rotation.quaternion().w;

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

                tf_broadcaster
                    .send_transform(tf)
                    .expect("Failed to send transform");

                cloud_publisher
                    .publish(&cloud)
                    .expect("Failed to publish cloud");
                odom_publisher
                    .publish(&odom)
                    .expect("Failed to publish odom");

                eval.stop("total").unwrap();

                if last_render.elapsed() > render_interval {
                    last_render = std::time::Instant::now();
                    r2r::log_info!(
                        "lio",
                        "{}[2J{}[1;1H[LIO] - Simple LiDAR Odometry\n{}",
                        27 as char,
                        27 as char,
                        eval.render()
                    );
                }
            }
            Err(e) => {
                r2r::log_error!("lifis", "{}", e);
            }
            _ => {}
        }
    }
}
