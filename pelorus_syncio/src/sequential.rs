//! Input of the sequential pipelining mode.
//!
//! [`SequentialSource`] holds the cloud, IMU and wheel subscriptions and hands
//! out one message at a time to the thread that calls
//! [`next`](SequentialSource::next). No subscriber threads are started, so the
//! caller sees the messages in arrival order and processes each before it
//! takes the next one.

use std::sync::Arc;
use std::time::Duration;

use mt_pubsub::Node;
use pelorus_core::{ImuMsg, OdomMsg, SyncNodeParameters, SyncParameters};
use pelorus_sync::PreprocessingPointCloud;
use ros_pointcloud2::ros::HeaderMsg;
use ros2_interfaces_jazzy_rkyv::nav_msgs::msg::Odometry;
use ros2_interfaces_jazzy_rkyv::sensor_msgs::msg::Imu;

use crate::{
    MtCloudSubscriber, imu_to_sync, mt_cloud_source, mt_cloud_subscriber, mt_odom_to_sync,
    next_mt_cloud, qos_to_mt, wheel_to_imu_rotation,
};

/// One received message, converted for the synchronizer.
pub enum SequentialMessage {
    Cloud {
        header: HeaderMsg,
        cloud: Box<dyn PreprocessingPointCloud>,
    },
    Imu(ImuMsg),
    Wheel(OdomMsg),
}

/// The subscriptions of the sequential mode, read on the caller's thread.
pub struct SequentialSource {
    rt: Arc<tokio::runtime::Runtime>,
    local_only: bool,
    cloud: MtCloudSubscriber,
    imu: Option<mt_pubsub::Subscriber<Imu>>,
    wheel: mt_pubsub::Subscriber<Odometry>,
    wheel_frame_to_imu: nalgebra::Matrix3<f64>,
}

impl SequentialSource {
    /// Subscribe to the configured topics. Without `imu_enabled` the IMU topic
    /// is not subscribed.
    pub fn new(
        node: &Node,
        rt: Arc<tokio::runtime::Runtime>,
        local_only: bool,
        node_params: &SyncNodeParameters,
        params: &SyncParameters,
        imu_enabled: bool,
    ) -> anyhow::Result<Self> {
        let (cloud, imu, wheel) = rt.block_on(async {
            let cloud = mt_cloud_subscriber(node, node_params).await?;
            let imu = if imu_enabled {
                Some(
                    node.create_subscriber::<Imu>(
                        node_params.imu_src_topic.clone(),
                        100,
                        qos_to_mt(node_params.imu_qos),
                    )
                    .await?,
                )
            } else {
                None
            };
            let wheel = node
                .create_subscriber::<Odometry>(
                    node_params.wheel_src_topic.clone(),
                    10,
                    qos_to_mt(node_params.wheel_qos),
                )
                .await?;
            Ok::<_, anyhow::Error>((cloud, imu, wheel))
        })?;
        Ok(Self {
            rt,
            local_only,
            cloud,
            imu,
            wheel,
            wheel_frame_to_imu: wheel_to_imu_rotation(params),
        })
    }

    /// The next message, waiting at most `timeout`. Returns `None` when the
    /// timeout passed without a message.
    pub fn next(&mut self, timeout: Duration) -> anyhow::Result<Option<SequentialMessage>> {
        enum Received {
            Cloud(crate::IncomingMtPointCloud),
            Imu(Imu),
            Wheel(Box<Odometry>),
            Timeout,
            Closed,
        }
        let Self {
            rt,
            local_only,
            cloud,
            imu,
            wheel,
            ..
        } = self;
        let received = rt.block_on(async {
            let imu_next = async {
                match imu.as_mut() {
                    Some(imu) => imu.next().await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                // Clouds first, so a cloud and the IMU behind it arrive in the
                // order Minot sent them.
                biased;
                cloud = next_mt_cloud(cloud, *local_only) => cloud.map_or(Received::Closed, Received::Cloud),
                imu = imu_next => imu.map_or(Received::Closed, Received::Imu),
                odom = wheel.next() => odom.map_or(Received::Closed, |odom| Received::Wheel(Box::new(odom))),
                () = tokio::time::sleep(timeout) => Received::Timeout,
            }
        });
        Ok(match received {
            Received::Cloud(msg) => {
                let (header, cloud) = mt_cloud_source(msg)?;
                Some(SequentialMessage::Cloud { header, cloud })
            }
            Received::Imu(msg) => Some(SequentialMessage::Imu(imu_to_sync(msg))),
            Received::Wheel(msg) => Some(SequentialMessage::Wheel(mt_odom_to_sync(
                *msg,
                self.wheel_frame_to_imu,
            ))),
            Received::Timeout => None,
            Received::Closed => anyhow::bail!("a sequential input subscription closed"),
        })
    }
}

/// Ask Minot for the next IMU message during the IMU initialization.
#[cfg(feature = "rat")]
pub fn request_imu_init_step(synchronous_mode: bool) {
    crate::publish_minot_trigger(synchronous_mode, crate::MINOT_IMU_INIT_TICK);
}

/// Tell Minot the IMU initialization is complete, which starts the replay of
/// clouds.
#[cfg(feature = "rat")]
pub fn request_lio_start(synchronous_mode: bool) {
    crate::publish_minot_trigger(synchronous_mode, crate::MINOT_LIO_START_TICK);
}
