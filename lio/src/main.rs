#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use r2r::{nav_msgs::msg::Odometry, sensor_msgs::msg::PointCloud2};
use ros_pointcloud2::prelude::*;

struct Map {
    c_map: *mut std::os::raw::c_void,
}

impl Map {
    fn new() -> Self {
        Map {
            c_map: unsafe { init_map() },
        }
    }

    /// Assuming PointCloud2 to be in XYZ format, so can copy directly to C++.
    /// Data gets copied on C++ side, so no need to worry about lifetimes.
    fn add(&mut self, cloud: &PointCloud2) {
        unsafe {
            // TODO should also have the pose with it, somehow not showing
            add_to_map(
                self.c_map,
                cloud.data.as_ptr() as *mut std::os::raw::c_void,
                cloud.width as i32,
            );
        }
    }

    fn register(&mut self, cloud: &PointCloud2) -> nalgebra::Isometry3<f64> {
        let res = unsafe {
            let translation = vec![0.0; 3];
            let rotation = vec![0.0; 4];
            align(
                self.c_map,
                cloud.data.as_ptr() as *mut std::os::raw::c_void,
                cloud.width as i32,
                translation.as_ptr() as *mut f64,
                rotation.as_ptr() as *mut f64,
            );
            (translation, rotation)
        };

        nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(res.0[0], res.0[1], res.0[2]),
            nalgebra::UnitQuaternion::from_euler_angles(res.1[0], res.1[1], res.1[2]),
        )
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
    let it = cloud.try_into_par_iter().unwrap().map(|p: PointXYZ| {
        let t_p = pose.transform_point(&nalgebra::Point3::new(p.x as f64, p.y as f64, p.z as f64));
        PointXYZ {
            x: t_p.x as f32,
            y: t_p.y as f32,
            z: t_p.z as f32,
        }
    });

    PointCloud2Msg::try_from_par_iter(it).unwrap().into()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "lifis", "")?;

    let mut lifis_node = lifis_ros::LifisNode::create(lifis_ros::LifisNodeParameters {
        node_name: "lifis",
        cloud_src_topic: "pointcloud",
        imu_src_topic: "imu",
        ..Default::default()
    })
    .await?;

    let cloud_publisher =
        node.create_publisher::<PointCloud2>("registered", r2r::QosProfile::default())?;
    let odom_publisher = node.create_publisher::<Odometry>("odom", r2r::QosProfile::default())?;

    let mut map = Map::new();

    #[cfg(debug_assertions)]
    r2r::log_warn!("lifis", "Running in dev mode.");

    loop {
        node.spin_once(std::time::Duration::from_millis(100));
        match lifis_node.tick().await {
            Ok(Some((cloud, mut odom))) => {
                let pose = map.register(&cloud);
                map.add(&cloud); // TODO pose where?

                let cloud = transform_cloud(cloud, &pose);

                odom.pose.pose.position.x = pose.translation.vector.x;
                odom.pose.pose.position.y = pose.translation.vector.y;
                odom.pose.pose.position.z = pose.translation.vector.z;
                odom.pose.pose.orientation.x = pose.rotation.quaternion().i;
                odom.pose.pose.orientation.y = pose.rotation.quaternion().j;
                odom.pose.pose.orientation.z = pose.rotation.quaternion().k;
                odom.pose.pose.orientation.w = pose.rotation.quaternion().w;

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
