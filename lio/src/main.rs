use r2r::{nav_msgs::msg::Odometry, sensor_msgs::msg::PointCloud2};
use ros_pointcloud2::prelude::*;

use small_gicp_rs::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "lio", "")?;

    let mut lifis_node = lifis_ros::LifisNode::create(lifis_ros::LifisNodeParameters {
        node_name: "lifis",
        cloud_src_topic: "ouster/points",
        imu_src_topic: "ouster/imu",
        ..Default::default()
    })
    .await?;

    let cloud_publisher =
        node.create_publisher::<PointCloud2>("registered", r2r::QosProfile::default())?;
    let odom_publisher = node.create_publisher::<Odometry>("odom", r2r::QosProfile::default())?;

    let voxel_size = 0.15;
    let mut map = Map::new(voxel_size);

    #[cfg(debug_assertions)]
    r2r::log_warn!("lio", "Running in dev mode.");

    let mut t_world_lidar = nalgebra::Isometry3::identity();
    //let mut t_last_current = nalgebra::Isometry3::identity();
    let map_frame = "map".to_string();

    let mut first_scan = true;

    loop {
        node.spin_once(std::time::Duration::from_millis(100));
        match lifis_node.tick().await {
            Ok(Some((cloud, mut odom))) => {
                let cloud: PointCloud2Msg = cloud.into();
                if first_scan {
                    map.add(&cloud, None);
                    first_scan = false;
                }
                let init_guess = nalgebra::Isometry3::identity(); // TODO from tick odom
                let init_guess_world = t_world_lidar * init_guess;

                let start_time = std::time::Instant::now();
                let register_pose = map.register(&cloud, &init_guess_world);
                let elapsed = start_time.elapsed();
                r2r::log_info!("lio", "Registration took: {:?}", elapsed);

                t_world_lidar = register_pose;

                dbg!(t_world_lidar);

                let start_time = std::time::Instant::now();
                map.add(&cloud, Some(t_world_lidar));
                let elapsed = start_time.elapsed();
                r2r::log_info!("lio", "Adding cloud took: {:?}", elapsed);

                odom.pose.pose.position.x = t_world_lidar.translation.vector.x;
                odom.pose.pose.position.y = t_world_lidar.translation.vector.y;
                odom.pose.pose.position.z = t_world_lidar.translation.vector.z;
                odom.pose.pose.orientation.x = t_world_lidar.rotation.quaternion().i;
                odom.pose.pose.orientation.y = t_world_lidar.rotation.quaternion().j;
                odom.pose.pose.orientation.z = t_world_lidar.rotation.quaternion().k;
                odom.pose.pose.orientation.w = t_world_lidar.rotation.quaternion().w;

                let mut cloud: PointCloud2 = transform_cloud(cloud, &t_world_lidar).into();

                cloud.header.frame_id = map_frame.clone();

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
