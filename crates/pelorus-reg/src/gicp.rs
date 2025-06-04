// use crate::Registration;

// pub struct Gicp {
//     map: small_gicp_sys::Map,
// }

// impl Gicp {
//     pub fn new(voxel_size: f64) -> Self {
//         Self {
//             map: small_gicp_sys::Map::new(voxel_size),
//         }
//     }
// }

// impl Registration for Gicp {
//     fn register_and_map(
//         &mut self,
//         data: crate::Mode,
//     ) -> anyhow::Result<crate::CloudRegisterResult> {
//         todo!()
//     }

//     fn requested_register_and_map_cloud_type(&self) -> crate::PointXYZLocatedBufferView {
//         todo!()
//     }

//     // fn imu_ekf_predict(
//     //     &mut self,
//     //     acc: [f64; 3],
//     //     gyro: [f64; 3],
//     //     dt: f64,
//     // ) -> crate::ImuEkfPrediction {
//     //     todo!()
//     // }

//     fn get_state(&self) -> crate::ImuEkfPrediction {
//         todo!()
//     }

//     fn undistort(
//         &mut self,
//         imu_msgs: &[crate::ImuMsg],
//         last_cloud_end_time: f64,
//         cloud_begin_time: f64,
//         cloud_end_time: f64,
//         cloud: &[prelio::PointXYZT],
//         lidar_vendor: Option<prelio::LidarVendor>,
//     ) -> Vec<ros_pointcloud2::prelude::PointXYZ> {
//         todo!()
//     }
// }
