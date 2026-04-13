//! TF2 support for mt_pubsub backend using transforms crate.
//!
//! This module provides TF broadcaster and listener by wrapping the
//! middleware-independent `transforms` crate with ROS2 message handling.

use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use log::error;
use nalgebra::geometry::{Isometry3, Translation3, UnitQuaternion};
use transforms::geometry::{Quaternion, Transform, Vector3};
use transforms::time::Timestamp;
use transforms::Registry;

const TF_TOPIC: &str = "/tf";
const TF_STATIC_TOPIC: &str = "/tf_static";

/// TF Broadcaster for mt_pubsub.
pub struct TfBroadcaster {
    _publisher_node: Option<Arc<mt_pubsub::Node>>,
    publisher: Option<mt_pubsub::Publisher<ros2_interfaces_jazzy_rkyv::tf2_msgs::msg::TFMessage>>,
    static_publisher: Option<mt_pubsub::Publisher<ros2_interfaces_jazzy_rkyv::tf2_msgs::msg::TFMessage>>,
}

impl TfBroadcaster {
    /// Create a new TF broadcaster.
    pub fn new(_node: &Arc<mt_pubsub::Node>, rt: Arc<tokio::runtime::Runtime>) -> Self {
        let (publisher_node, publisher, static_publisher) = rt.block_on(async {
            let node_name = format!("tf_pub_{}", std::process::id());
            let config = mt_pubsub::NodeConfig {
                name: node_name,
                mode: mt_pubsub::Qos::BestEffort,
                coord_mode: mt_pubsub::CoordMode::AutoStart,
            };

            match mt_pubsub::Node::create(config).await {
                Ok(pub_node) => {
                    let node = Arc::new(pub_node);
                    let dyn_pub = node
                        .create_publisher::<ros2_interfaces_jazzy_rkyv::tf2_msgs::msg::TFMessage>(
                            TF_TOPIC.to_owned(),
                            mt_pubsub::Qos::BestEffort,
                        )
                        .await
                        .ok();
                    let static_pub = node
                        .create_publisher::<ros2_interfaces_jazzy_rkyv::tf2_msgs::msg::TFMessage>(
                            TF_STATIC_TOPIC.to_owned(),
                            mt_pubsub::Qos::BestEffort,
                        )
                        .await
                        .ok();
                    (Some(node), dyn_pub, static_pub)
                }
                Err(e) => {
                    error!("Failed to create TF publisher node: {}", e);
                    (None, None, None)
                }
            }
        });

        Self {
            _publisher_node: publisher_node,
            publisher,
            static_publisher,
        }
    }

    /// Broadcast a transform asynchronously.
    pub async fn send_transform(&self, transform: ros2_interfaces_jazzy_rkyv::geometry_msgs::msg::TransformStamped) -> Result<(), TfError> {
        if let Some(ref publisher) = self.publisher {
            let msg = ros2_interfaces_jazzy_rkyv::tf2_msgs::msg::TFMessage {
                transforms: vec![transform],
            };
            publisher.publish(&msg).await.map_err(|e| TfError::Publish(e.to_string()))
        } else {
            Ok(())
        }
    }

    /// Broadcast a static transform asynchronously.
    pub async fn send_static_transform(&self, transform: ros2_interfaces_jazzy_rkyv::geometry_msgs::msg::TransformStamped) -> Result<(), TfError> {
        if let Some(ref publisher) = self.static_publisher {
            let msg = ros2_interfaces_jazzy_rkyv::tf2_msgs::msg::TFMessage {
                transforms: vec![transform],
            };
            publisher.publish(&msg).await.map_err(|e| TfError::Publish(e.to_string()))
        } else {
            Ok(())
        }
    }
}

/// TF Listener for mt_pubsub.
pub struct TfListener {
    registry: Arc<RwLock<Registry<Timestamp>>>,
}

impl TfListener {
    /// Create a new TF listener.
    pub fn new(node: &mt_pubsub::Node, rt: Arc<tokio::runtime::Runtime>) -> Self {
        let registry = Arc::new(RwLock::new(Registry::new(Duration::from_secs(10))));

        // Subscribe to /tf
        let mut dynamic_sub = rt.block_on(async {
            node.create_subscriber::<ros2_interfaces_jazzy_rkyv::tf2_msgs::msg::TFMessage>(
                TF_TOPIC.to_owned(),
                100,
                mt_pubsub::Qos::BestEffort,
            )
            .await
            .expect("Failed to subscribe to /tf")
        });

        let registry_dyn = registry.clone();
        thread::spawn(move || {
            futures_lite::future::block_on(async {
                while Arc::strong_count(&registry_dyn) > 1 {
                    while let Some(msg) = dynamic_sub.next().await {
                        if let Ok(mut reg) = registry_dyn.write() {
                            for tf in msg.transforms {
                                let transform = ros_to_transform(&tf);
                                reg.add_transform(transform);
                            }
                        }
                    }
                }
            });
        });

        // Subscribe to /tf_static
        let mut static_sub = rt.block_on(async {
            node.create_subscriber::<ros2_interfaces_jazzy_rkyv::tf2_msgs::msg::TFMessage>(
                TF_STATIC_TOPIC.to_owned(),
                100,
                mt_pubsub::Qos::BestEffort,
            )
            .await
            .expect("Failed to subscribe to /tf_static")
        });

        let registry_static = registry.clone();
        thread::spawn(move || {
            futures_lite::future::block_on(async {
                while Arc::strong_count(&registry_static) > 1 {
                    while let Some(msg) = static_sub.next().await {
                        if let Ok(mut reg) = registry_static.write() {
                            for tf in msg.transforms {
                                let transform = ros_to_transform(&tf);
                                reg.add_transform(transform);
                            }
                        }
                    }
                }
            });
        });

        Self { registry }
    }

    /// Look up a transform at a specific time.
    pub fn lookup_transform(
        &self,
        source_frame: &str,
        target_frame: &str,
        time: ros2_interfaces_jazzy_rkyv::builtin_interfaces::msg::Time,
    ) -> Result<ros2_interfaces_jazzy_rkyv::geometry_msgs::msg::TransformStamped, TfError> {
        let timestamp = ros_time_to_timestamp(&time);

        let reg = self.registry.read().map_err(|_| TfError::Lock)?;

        // Handle zero time (get latest)
        let ts = if time.sec == 0 && time.nanosec == 0 {
            Timestamp::now()
        } else {
            timestamp
        };

        let transform = reg.get_transform(source_frame, target_frame, ts)
            .map_err(|e| TfError::Lookup(e.to_string()))?;

        Ok(transform_to_ros(&transform, time))
    }
}

/// Convert ROS TransformStamped to transforms::Transform.
fn ros_to_transform(tf: &ros2_interfaces_jazzy_rkyv::geometry_msgs::msg::TransformStamped) -> Transform {
    Transform {
        translation: Vector3::new(
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ),
        rotation: Quaternion {
            w: tf.transform.rotation.w,
            x: tf.transform.rotation.x,
            y: tf.transform.rotation.y,
            z: tf.transform.rotation.z,
        },
        timestamp: ros_time_to_timestamp(&tf.header.stamp),
        parent: tf.header.frame_id.clone(),
        child: tf.child_frame_id.clone(),
    }
}

/// Convert transforms::Transform to ROS TransformStamped.
fn transform_to_ros(tf: &Transform, time: ros2_interfaces_jazzy_rkyv::builtin_interfaces::msg::Time) -> ros2_interfaces_jazzy_rkyv::geometry_msgs::msg::TransformStamped {
    ros2_interfaces_jazzy_rkyv::geometry_msgs::msg::TransformStamped {
        header: ros2_interfaces_jazzy_rkyv::std_msgs::msg::Header {
            stamp: time,
            frame_id: tf.parent.clone(),
        },
        child_frame_id: tf.child.clone(),
        transform: ros2_interfaces_jazzy_rkyv::geometry_msgs::msg::Transform {
            translation: ros2_interfaces_jazzy_rkyv::geometry_msgs::msg::Vector3 {
                x: tf.translation.x,
                y: tf.translation.y,
                z: tf.translation.z,
            },
            rotation: ros2_interfaces_jazzy_rkyv::geometry_msgs::msg::Quaternion {
                x: tf.rotation.x,
                y: tf.rotation.y,
                z: tf.rotation.z,
                w: tf.rotation.w,
            },
        },
    }
}

/// Convert ROS Time to Timestamp.
fn ros_time_to_timestamp(time: &ros2_interfaces_jazzy_rkyv::builtin_interfaces::msg::Time) -> Timestamp {
    let nanos = (time.sec as u128) * 1_000_000_000 + (time.nanosec as u128);
    Timestamp { t: nanos }
}

/// Convert a ROS Transform to an Isometry3.
pub fn isometry_from_transform(tf: &ros2_interfaces_jazzy_rkyv::geometry_msgs::msg::Transform) -> Isometry3<f64> {
    let trans = Translation3::new(tf.translation.x, tf.translation.y, tf.translation.z);
    let rot = UnitQuaternion::new_normalize(nalgebra::geometry::Quaternion::new(
        tf.rotation.w,
        tf.rotation.x,
        tf.rotation.y,
        tf.rotation.z,
    ));
    Isometry3::from_parts(trans, rot)
}

/// TF error type.
#[derive(Debug, thiserror::Error)]
pub enum TfError {
    #[error("Failed to acquire lock")]
    Lock,
    #[error("Transform lookup failed: {0}")]
    Lookup(String),
    #[error("Publish failed: {0}")]
    Publish(String),
}
