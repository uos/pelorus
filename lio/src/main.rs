#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use r2r::{nav_msgs::msg::Odometry, sensor_msgs::msg::PointCloud2};
use ros_pointcloud2::prelude::*;

struct Map {
    c_map: *mut std::os::raw::c_void,
}

impl From<nalgebra::Isometry3<f64>> for LioIsometry3d {
    fn from(iso: nalgebra::Isometry3<f64>) -> Self {
        LioIsometry3d {
            x: iso.translation.vector.x,
            y: iso.translation.vector.y,
            z: iso.translation.vector.z,
            q_i: iso.rotation.quaternion().i,
            q_j: iso.rotation.quaternion().j,
            q_k: iso.rotation.quaternion().k,
            q_w: iso.rotation.quaternion().w,
        }
    }
}

impl From<LioIsometry3d> for nalgebra::Isometry3<f64> {
    fn from(iso: LioIsometry3d) -> Self {
        nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(iso.x, iso.y, iso.z),
            nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
                iso.q_w, iso.q_i, iso.q_j, iso.q_k,
            )),
        )
    }
}

impl Map {
    fn new(voxel_size: f64) -> Self {
        Map {
            c_map: unsafe { init_map(voxel_size) },
        }
    }

    /// Assuming PointCloud2 to be in XYZ format, so can copy directly to C++.
    /// Data gets copied on C++ side, so no need to worry about lifetimes.
    fn add(&mut self, cloud: &PointCloud2, pose: Option<nalgebra::Isometry3<f64>>) {
        let pose = pose.unwrap_or(nalgebra::Isometry3::identity());

        unsafe {
            add_to_map(
                self.c_map,
                cloud.data.as_ptr() as *const std::os::raw::c_void,
                (cloud.width * cloud.height) as i32,
                pose.into(),
            )
        };
    }

    fn register(
        &mut self,
        cloud: &PointCloud2,
        init_guess: &nalgebra::Isometry3<f64>,
    ) -> nalgebra::Isometry3<f64> {
        let mut out_transform: LioIsometry3d = nalgebra::Isometry3::identity().into();
        let point_buffer = cloud.data.as_ptr() as *const std::os::raw::c_void;

        unsafe {
            align(
                self.c_map,
                point_buffer,
                (cloud.width * cloud.height) as i32,
                (*init_guess).into(),
                &mut out_transform,
            );
        }

        out_transform.into()
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        unsafe {
            drop_map(self.c_map);
        }
    }
}

fn transform_cloud(cloud: PointCloud2, pose: &nalgebra::Isometry3<f64>) -> PointCloud2 {
    let cloud: ros_pointcloud2::PointCloud2Msg = cloud.into();
    let it = cloud
        .try_into_vec()
        .unwrap()
        .into_iter()
        .map(|p: PointXYZ| {
            let t_p =
                pose.transform_point(&nalgebra::Point3::new(p.x as f64, p.y as f64, p.z as f64));
            PointXYZ {
                x: t_p.x as f32,
                y: t_p.y as f32,
                z: t_p.z as f32,
            }
        })
        .collect();

    PointCloud2Msg::try_from_vec(it).unwrap().into()
}

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
    let mut t_last_current = nalgebra::Isometry3::identity();
    let map_frame = "map".to_string();

    let mut first_scan = true;

    loop {
        node.spin_once(std::time::Duration::from_millis(100));
        match lifis_node.tick().await {
            Ok(Some((mut cloud, mut odom))) => {
                if first_scan {
                    map.add(&cloud, None);
                    first_scan = false;
                }
                let init_guess = nalgebra::Isometry3::identity(); // TODO from tick odom
                let world_pred = t_world_lidar * t_last_current * init_guess;

                let start_time = std::time::Instant::now();
                let register_pose = map.register(&cloud, &world_pred);
                let elapsed = start_time.elapsed();
                r2r::log_info!("lio", "Registration took: {:?}", elapsed);

                t_last_current = t_world_lidar.inverse() * register_pose;
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

                cloud.header.frame_id = "map".to_string();

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
