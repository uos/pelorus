//! TF2 support for ros2-client backend using transforms crate.
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

/// TF Broadcaster for ros2-client.
pub struct TfBroadcaster {
    publisher: ros2_client::Publisher<ros2_interfaces_jazzy_serde::tf2_msgs::msg::TFMessage>,
    static_publisher: ros2_client::Publisher<ros2_interfaces_jazzy_serde::tf2_msgs::msg::TFMessage>,
}

impl TfBroadcaster {
    /// Create a new TF broadcaster.
    #[track_caller]
    pub fn new(node: &mut ros2_client::Node) -> Self {
        let tf_topic = node
            .create_topic(
                &ros2_client::Name::new("/", "tf").unwrap(),
                ros2_client::MessageTypeName::new("tf2_msgs", "TFMessage"),
                &ros2_client::DEFAULT_PUBLISHER_QOS,
            )
            .expect("Failed to create /tf topic");

        let tf_static_topic = node
            .create_topic(
                &ros2_client::Name::new("/", "tf_static").unwrap(),
                ros2_client::MessageTypeName::new("tf2_msgs", "TFMessage"),
                &ros2_client::ros2::QosPolicyBuilder::new()
                    .durability(ros2_client::ros2::policy::Durability::TransientLocal)
                    .reliability(ros2_client::ros2::policy::Reliability::Reliable {
                        max_blocking_time: ros2_client::ros2::Duration::INFINITE,
                    })
                    .history(ros2_client::ros2::policy::History::KeepLast { depth: 1 })
                    .build(),
            )
            .expect("Failed to create /tf_static topic");

        let publisher = node
            .create_publisher(&tf_topic, None)
            .expect("Failed to create /tf publisher");

        let static_publisher = node
            .create_publisher(&tf_static_topic, None)
            .expect("Failed to create /tf_static publisher");

        Self {
            publisher,
            static_publisher,
        }
    }

    /// Broadcast a transform.
    pub fn send_transform(
        &self,
        transform: ros2_interfaces_jazzy_serde::geometry_msgs::msg::TransformStamped,
    ) -> Result<(), TfError> {
        let msg = ros2_interfaces_jazzy_serde::tf2_msgs::msg::TFMessage {
            transforms: vec![transform],
        };
        self.publisher
            .publish(msg)
            .map_err(|e| TfError::Publish(e.to_string()))
    }

    /// Broadcast a static transform.
    pub fn send_static_transform(
        &self,
        transform: ros2_interfaces_jazzy_serde::geometry_msgs::msg::TransformStamped,
    ) -> Result<(), TfError> {
        let msg = ros2_interfaces_jazzy_serde::tf2_msgs::msg::TFMessage {
            transforms: vec![transform],
        };
        self.static_publisher
            .publish(msg)
            .map_err(|e| TfError::Publish(e.to_string()))
    }
}

/// TF Listener for ros2-client.
pub struct TfListener {
    registry: Arc<RwLock<Registry<Timestamp>>>,
}

impl TfListener {
    /// Create a new TF listener.
    #[track_caller]
    pub fn new(node: &mut ros2_client::Node) -> Self {
        let registry = Arc::new(RwLock::new(Registry::new(Duration::from_secs(10))));

        // Subscribe to /tf
        let tf_topic = node
            .create_topic(
                &ros2_client::Name::new("/", "tf").unwrap(),
                ros2_client::MessageTypeName::new("tf2_msgs", "TFMessage"),
                &ros2_client::DEFAULT_SUBSCRIPTION_QOS,
            )
            .expect("Failed to create /tf topic");

        let dynamic_sub = node
            .create_subscription::<ros2_interfaces_jazzy_serde::tf2_msgs::msg::TFMessage>(
                &tf_topic, None,
            )
            .expect("Failed to subscribe to /tf");

        let registry_dyn = registry.clone();
        thread::spawn(move || {
            while Arc::strong_count(&registry_dyn) > 1 {
                match dynamic_sub.take() {
                    Ok(Some((msg, _))) => {
                        if let Ok(mut reg) = registry_dyn.write() {
                            for tf in msg.transforms {
                                let transform = ros_to_transform(&tf);
                                reg.add_transform(transform);
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(e) => error!("Error reading from /tf: {}", e),
                }
            }
        });

        // Subscribe to /tf_static
        let tf_static_topic = node
            .create_topic(
                &ros2_client::Name::new("/", "tf_static").unwrap(),
                ros2_client::MessageTypeName::new("tf2_msgs", "TFMessage"),
                &ros2_client::ros2::QosPolicyBuilder::new()
                    .durability(ros2_client::ros2::policy::Durability::TransientLocal)
                    .reliability(ros2_client::ros2::policy::Reliability::Reliable {
                        max_blocking_time: ros2_client::ros2::Duration::INFINITE,
                    })
                    .history(ros2_client::ros2::policy::History::KeepLast { depth: 1 })
                    .build(),
            )
            .expect("Failed to create /tf_static topic");

        let static_sub = node
            .create_subscription::<ros2_interfaces_jazzy_serde::tf2_msgs::msg::TFMessage>(
                &tf_static_topic,
                None,
            )
            .expect("Failed to subscribe to /tf_static");

        let registry_static = registry.clone();
        thread::spawn(move || {
            while Arc::strong_count(&registry_static) > 1 {
                match static_sub.take() {
                    Ok(Some((msg, _))) => {
                        if let Ok(mut reg) = registry_static.write() {
                            for tf in msg.transforms {
                                let transform = ros_to_transform(&tf);
                                reg.add_transform(transform);
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(e) => error!("Error reading from /tf_static: {}", e),
                }
            }
        });

        Self { registry }
    }

    /// Look up a transform at a specific time.
    pub fn lookup_transform(
        &self,
        source_frame: &str,
        target_frame: &str,
        time: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time,
    ) -> Result<ros2_interfaces_jazzy_serde::geometry_msgs::msg::TransformStamped, TfError> {
        let timestamp = ros_time_to_timestamp(&time);

        let reg = self.registry.read().map_err(|_| TfError::Lock)?;

        // Handle zero time (get latest)
        let ts = if time.sec == 0 && time.nanosec == 0 {
            Timestamp::now()
        } else {
            timestamp
        };

        let transform = reg
            .get_transform(source_frame, target_frame, ts)
            .map_err(|e| TfError::Lookup(e.to_string()))?;

        Ok(transform_to_ros(&transform, time))
    }
}

/// Convert ROS TransformStamped to transforms::Transform.
fn ros_to_transform(
    tf: &ros2_interfaces_jazzy_serde::geometry_msgs::msg::TransformStamped,
) -> Transform {
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
fn transform_to_ros(
    tf: &Transform,
    time: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time,
) -> ros2_interfaces_jazzy_serde::geometry_msgs::msg::TransformStamped {
    ros2_interfaces_jazzy_serde::geometry_msgs::msg::TransformStamped {
        header: ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
            stamp: time,
            frame_id: tf.parent.clone(),
        },
        child_frame_id: tf.child.clone(),
        transform: ros2_interfaces_jazzy_serde::geometry_msgs::msg::Transform {
            translation: ros2_interfaces_jazzy_serde::geometry_msgs::msg::Vector3 {
                x: tf.translation.x,
                y: tf.translation.y,
                z: tf.translation.z,
            },
            rotation: ros2_interfaces_jazzy_serde::geometry_msgs::msg::Quaternion {
                x: tf.rotation.x,
                y: tf.rotation.y,
                z: tf.rotation.z,
                w: tf.rotation.w,
            },
        },
    }
}

/// Convert ROS Time to Timestamp.
fn ros_time_to_timestamp(
    time: &ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time,
) -> Timestamp {
    let nanos = (time.sec as u128) * 1_000_000_000 + (time.nanosec as u128);
    Timestamp { t: nanos }
}

/// Convert a ROS Transform to an Isometry3.
pub fn isometry_from_transform(
    tf: &ros2_interfaces_jazzy_serde::geometry_msgs::msg::Transform,
) -> Isometry3<f64> {
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
