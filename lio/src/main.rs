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
    let odom_publisher = node.create_publisher::<Odometry>("odom", r2r::QosProfile::default())?;

    let map_voxel_size = 0.15;
    let mut map = Map::new(map_voxel_size);

    #[cfg(debug_assertions)]
    r2r::log_warn!("lio", "Running in dev mode.");

    let mut t_world_lidar = nalgebra::Isometry3::identity();
    let mut t_last_current = nalgebra::Isometry3::identity();
    let map_frame = "map".to_string();

    let mut first_scan = true;
    let mut last_scan_time = None;

    loop {
        node.spin_once(std::time::Duration::from_millis(100));
        match lifis_node.tick().await {
            Ok(Some((orig_cloud, mut odom))) => {
                let cloud: PointCloud2Msg = orig_cloud.into();
                let current_cloud_time = timestamp_to_secs(&cloud.header.stamp);
                if let Some(existing_last_scan_time) = last_scan_time {
                    if current_cloud_time < existing_last_scan_time {
                        map.reset(map_voxel_size);
                        r2r::log_info!("lio", "Resetting map");
                        first_scan = true;
                        t_world_lidar = nalgebra::Isometry3::identity();
                        t_last_current = nalgebra::Isometry3::identity();
                    }
                }

                last_scan_time = Some(timestamp_to_secs(&cloud.header.stamp));

                if first_scan {
                    map.add(&cloud, None);
                    first_scan = false;
                    // TODO: rpcl2 - from and into for timemsg
                    continue;
                }

                let init_guess = nalgebra::Isometry3::identity(); // TODO from tick odom
                let init_guess_world = t_world_lidar * t_last_current * init_guess;

                //let start_time = std::time::Instant::now();
                let register_pose = map.register(&cloud, &init_guess_world);
                //let elapsed = start_time.elapsed();
                //r2r::log_info!("lio", "Registration took: {:?}", elapsed);

                t_last_current = t_world_lidar.inverse() * register_pose.transform_target_to_source;
                t_world_lidar = register_pose.transform_target_to_source;

                //let start_time = std::time::Instant::now();
                map.add(&cloud, Some(register_pose.transform_target_to_source));
                //let elapsed = start_time.elapsed();
                //r2r::log_info!("lio", "Adding cloud took: {:?}", elapsed);

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
                let orig_frame = cloud.header.frame_id.clone();
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
            }
            Err(e) => {
                r2r::log_error!("lifis", "{}", e);
            }
            _ => {}
        }
    }
}
