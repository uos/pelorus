//! TF2 support for r2r backend using transforms crate.
//!
//! This module provides TF broadcaster and listener by wrapping the
//! middleware-independent `transforms` crate with ROS2 message handling.

use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use futures_lite::StreamExt;
use log::error;
use nalgebra::geometry::{Isometry3, Translation3, UnitQuaternion};
use transforms::geometry::{Quaternion, Transform, Vector3};
use transforms::time::Timestamp;
use transforms::Registry;

/// TF Broadcaster for r2r.
pub struct TfBroadcaster {
    publisher: r2r::Publisher<r2r::tf2_msgs::msg::TFMessage>,
    static_publisher: r2r::Publisher<r2r::tf2_msgs::msg::TFMessage>,
}

impl TfBroadcaster {
    /// Create a new TF broadcaster.
    #[track_caller]
    pub fn new(node: &mut r2r::Node) -> Self {
        let publisher = node
            .create_publisher("/tf", r2r::QosProfile::default())
            .expect("Failed to create /tf publisher");

        let static_publisher = node
            .create_publisher(
                "/tf_static",
                r2r::QosProfile::parameters()
                    .durability(r2r::qos::DurabilityPolicy::TransientLocal)
                    .reliability(r2r::qos::ReliabilityPolicy::Reliable)
                    .history(r2r::qos::HistoryPolicy::KeepLast)
                    .keep_last(1),
            )
            .expect("Failed to create /tf_static publisher");

        Self { publisher, static_publisher }
    }

    /// Broadcast a transform.
    pub fn send_transform(&self, transform: r2r::geometry_msgs::msg::TransformStamped) -> Result<(), TfError> {
        let msg = r2r::tf2_msgs::msg::TFMessage {
            transforms: vec![transform],
        };
        self.publisher.publish(&msg).map_err(|e| TfError::Publish(e.to_string()))
    }

    /// Broadcast a static transform.
    pub fn send_static_transform(&self, transform: r2r::geometry_msgs::msg::TransformStamped) -> Result<(), TfError> {
        let msg = r2r::tf2_msgs::msg::TFMessage {
            transforms: vec![transform],
        };
        self.static_publisher.publish(&msg).map_err(|e| TfError::Publish(e.to_string()))
    }
}

/// TF Listener for r2r.
pub struct TfListener {
    registry: Arc<RwLock<Registry<Timestamp>>>,
}

impl TfListener {
    /// Create a new TF listener.
    #[track_caller]
    pub fn new(node: &mut r2r::Node) -> Self {
        let registry = Arc::new(RwLock::new(Registry::new(Duration::from_secs(10))));

        // Subscribe to /tf
        let mut dynamic_sub = node
            .subscribe::<r2r::tf2_msgs::msg::TFMessage>("/tf", r2r::QosProfile::default())
            .expect("Failed to subscribe to /tf");

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
        let mut static_sub = node
            .subscribe::<r2r::tf2_msgs::msg::TFMessage>(
                "/tf_static",
                r2r::QosProfile::parameters()
                    .durability(r2r::qos::DurabilityPolicy::TransientLocal)
                    .reliability(r2r::qos::ReliabilityPolicy::Reliable)
                    .history(r2r::qos::HistoryPolicy::KeepLast)
                    .keep_last(1),
            )
            .expect("Failed to subscribe to /tf_static");

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
        time: r2r::builtin_interfaces::msg::Time,
    ) -> Result<r2r::geometry_msgs::msg::TransformStamped, TfError> {
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
fn ros_to_transform(tf: &r2r::geometry_msgs::msg::TransformStamped) -> Transform {
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
fn transform_to_ros(tf: &Transform, time: r2r::builtin_interfaces::msg::Time) -> r2r::geometry_msgs::msg::TransformStamped {
    r2r::geometry_msgs::msg::TransformStamped {
        header: r2r::std_msgs::msg::Header {
            stamp: time,
            frame_id: tf.parent.clone(),
        },
        child_frame_id: tf.child.clone(),
        transform: r2r::geometry_msgs::msg::Transform {
            translation: r2r::geometry_msgs::msg::Vector3 {
                x: tf.translation.x,
                y: tf.translation.y,
                z: tf.translation.z,
            },
            rotation: r2r::geometry_msgs::msg::Quaternion {
                x: tf.rotation.x,
                y: tf.rotation.y,
                z: tf.rotation.z,
                w: tf.rotation.w,
            },
        },
    }
}

/// Convert ROS Time to Timestamp.
fn ros_time_to_timestamp(time: &r2r::builtin_interfaces::msg::Time) -> Timestamp {
    let nanos = (time.sec as u128) * 1_000_000_000 + (time.nanosec as u128);
    Timestamp { t: nanos }
}

/// Convert a ROS Transform to an Isometry3.
pub fn isometry_from_transform(tf: &r2r::geometry_msgs::msg::Transform) -> Isometry3<f64> {
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
