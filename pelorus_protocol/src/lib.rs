use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 2;
pub const REGISTERED_FRAME_TOPIC: &str = "pelorus/registered_frame";
pub const DOWNSTREAM_CONTROL_SERVICE: &str = "pelorus/downstream_control";
pub const DOWNSTREAM_READY_TOPIC: &str = "pelorus/downstream_ready";
/// Topic carrying registered clouds as plain `PointCloud2`.
pub const REGISTERED_CLOUD_TOPIC: &str = "registered";

/// Companion topic carrying the same clouds Cloudini-compressed.
///
/// The handshake selects which typed topic carries traffic. Consumers subscribe
/// to both topics before registering because registration releases bag replay.
pub fn compressed_cloud_topic(base: &str) -> String {
    format!("{base}/compressed")
}

#[derive(
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
)]
pub enum CloudEncoding {
    PointCloud2,
    CompressedPointCloud2,
}

impl CloudEncoding {
    /// Required fallback for protocol participants.
    pub const BASELINE: Self = Self::PointCloud2;
}

/// Cloudini's compressed ROS message with mt_pubsub-compatible rkyv fields.
///
/// Shared by producers and consumers to keep the wire representation identical.
///
/// This crate avoids Cloudini and `ros_pointcloud2` dependencies so viewers can
/// select compatible versions. Each endpoint converts this field-for-field
/// representation locally.
#[derive(
    Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize, Debug, Clone, PartialEq,
)]
pub struct CompressedPointCloud2 {
    pub header: CompressedHeaderMsg,
    pub height: u32,
    pub width: u32,
    pub fields: Vec<CompressedPointFieldMsg>,
    pub is_bigendian: bool,
    pub point_step: u32,
    pub row_step: u32,
    pub compressed_data: Vec<u8>,
    pub is_dense: bool,
    pub format: String,
}

#[derive(
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
)]
pub struct CompressedTimeMsg {
    pub sec: i32,
    pub nanosec: u32,
}

#[derive(
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
    Serialize,
    Deserialize,
    Debug,
    Clone,
    PartialEq,
    Eq,
    Default,
)]
pub struct CompressedHeaderMsg {
    pub seq: u32,
    pub stamp: CompressedTimeMsg,
    pub frame_id: String,
}

#[derive(
    Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize, Debug, Clone, PartialEq, Eq,
)]
pub struct CompressedPointFieldMsg {
    pub name: String,
    pub offset: u32,
    pub datatype: u8,
    pub count: u32,
}

/// Published after Pelorus has published the matching registered cloud and
/// corrected Lidar odometry.
#[derive(
    Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize, Debug, Clone, PartialEq, Eq,
)]
pub struct RegisteredFrameMarker {
    pub protocol_version: u32,
    pub scan_id: u64,
    pub timestamp_sec: i32,
    pub timestamp_nanosec: u32,
    pub cloud_encoding: CloudEncoding,
    pub cloud_frame: String,
    pub odometry_frame: String,
}

/// Who releases the steady-state replay trigger once bootstrap is complete.
#[derive(
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
)]
pub enum TickOwner {
    /// Pelorus keeps its normal offline replay clock.
    Pelorus,
    /// The downstream consumer gates Minot by acknowledging each completed frame.
    Consumer,
}
#[derive(
    Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize, Debug, Clone, PartialEq, Eq,
)]
pub enum DownstreamRequest {
    GetCapabilities,
    Register {
        consumer: String,
        protocol_version: u32,
        tick_owner: TickOwner,
        /// Cloud encodings this consumer can actually decode, best first.
        ///
        /// Pelorus selects the first compatible encoding and reports it in
        /// [`DownstreamResponse::cloud_encoding`]. Baseline support is required.
        accepted_cloud_encodings: Vec<CloudEncoding>,
    },
    /// The registered consumer finished processing one authoritative frame.
    CompleteFrame {
        consumer: String,
        scan_id: u64,
    },
    GetStatus,
    Shutdown {
        consumer: String,
    },
}

/// Repeated by Pelorus after its managed downstream service subscriber is
/// active and until the user registers successfully.
#[derive(
    Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize, Debug, Clone, PartialEq, Eq,
)]
pub struct DownstreamReady {
    pub protocol_version: u32,
    pub process_id: u32,
    pub control_service: String,
}

#[derive(
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
)]
pub enum ReplayState {
    Starting,
    WaitingForConsumer,
    Running,
    EndOfBag,
    ShuttingDown,
    Failed,
}
#[derive(
    Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize, Debug, Clone, PartialEq, Eq,
)]
pub struct DownstreamResponse {
    pub protocol_version: u32,
    pub replay_state: ReplayState,
    pub tick_owner: TickOwner,
    pub accepted: bool,
    /// This request released the bag replay held back for the consumer.
    pub unlocked_replay: bool,
    /// The encoding registered clouds will actually use for this session.
    ///
    /// On `GetCapabilities` this is what Pelorus would prefer; on an accepted
    /// `Register` it is binding for the rest of the run. The consumer reads
    /// clouds from the matching topic and ignores the other.
    pub cloud_encoding: CloudEncoding,
    pub message: String,
}

/// State machine for Pelorus managed replay.
#[derive(Debug, Clone)]
pub struct DownstreamGate {
    consumer: Option<String>,
    tick_owner: TickOwner,
    replay_state: ReplayState,
    /// What Pelorus is configured and built to publish. Downgraded per session
    /// to whatever the registered consumer can decode.
    preferred_cloud_encoding: CloudEncoding,
    negotiated_cloud_encoding: CloudEncoding,
    last_completed_scan_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownstreamEffect {
    pub response: DownstreamResponse,
    /// Release the bag replay that was held until a consumer attached.
    pub unlock_replay: bool,
    pub request_shutdown: bool,
    /// Grant one replay credit after validation.
    pub completed_frame: Option<u64>,
}

impl Default for DownstreamGate {
    fn default() -> Self {
        Self::new(CloudEncoding::BASELINE)
    }
}

impl DownstreamGate {
    /// Creates a gate with this build's preferred cloud encoding.
    pub fn new(preferred_cloud_encoding: CloudEncoding) -> Self {
        Self {
            consumer: None,
            tick_owner: TickOwner::Pelorus,
            replay_state: ReplayState::WaitingForConsumer,
            preferred_cloud_encoding,
            negotiated_cloud_encoding: CloudEncoding::BASELINE,
            last_completed_scan_id: None,
        }
    }

    pub fn has_consumer(&self) -> bool {
        self.consumer.is_some()
    }

    pub fn tick_owner(&self) -> TickOwner {
        self.tick_owner
    }

    /// The encoding to publish registered clouds with. Meaningful once a
    /// consumer has registered; the baseline before that.
    pub fn negotiated_cloud_encoding(&self) -> CloudEncoding {
        self.negotiated_cloud_encoding
    }

    /// Highest encoding both sides support, always defined because the
    /// baseline is mandatory for everyone.
    fn negotiate(&self, accepted: &[CloudEncoding]) -> CloudEncoding {
        if accepted.contains(&self.preferred_cloud_encoding) {
            self.preferred_cloud_encoding
        } else {
            CloudEncoding::BASELINE
        }
    }

    pub fn handle(&mut self, request: DownstreamRequest) -> DownstreamEffect {
        let mut unlock_replay = false;
        let mut request_shutdown = false;
        let mut completed_frame = None;
        let mut reported_cloud_encoding = if self.has_consumer() {
            self.negotiated_cloud_encoding
        } else {
            self.preferred_cloud_encoding
        };
        let (accepted, message) = match request {
            DownstreamRequest::GetCapabilities => (true, "demand-driven managed replay supported"),
            DownstreamRequest::Register {
                consumer,
                protocol_version,
                tick_owner,
                accepted_cloud_encodings,
            } => {
                if protocol_version != PROTOCOL_VERSION {
                    (false, "protocol version mismatch")
                } else if !accepted_cloud_encodings.contains(&CloudEncoding::BASELINE) {
                    // Baseline support guarantees a compatible encoding.
                    (false, "consumer must accept the baseline cloud encoding")
                } else if self
                    .consumer
                    .as_ref()
                    .is_some_and(|current| current != &consumer)
                {
                    (false, "another consumer is registered")
                } else if self.consumer.is_some() && self.tick_owner != tick_owner {
                    (false, "tick owner cannot change after registration")
                } else {
                    let first_registration = self.consumer.is_none();
                    self.consumer = Some(consumer);
                    self.tick_owner = tick_owner;
                    self.negotiated_cloud_encoding = self.negotiate(&accepted_cloud_encodings);
                    self.replay_state = ReplayState::Running;
                    // Registration releases replay after subscribers are ready.
                    unlock_replay = first_registration;
                    reported_cloud_encoding = self.negotiated_cloud_encoding;
                    (true, "consumer registered")
                }
            }
            DownstreamRequest::CompleteFrame { consumer, scan_id } => {
                if self.consumer.as_deref() != Some(consumer.as_str()) {
                    (false, "consumer is not registered")
                } else if self.tick_owner != TickOwner::Consumer {
                    (false, "consumer does not own frame acknowledgements")
                } else if self
                    .last_completed_scan_id
                    .is_some_and(|last_scan_id| scan_id <= last_scan_id)
                {
                    (false, "frame completions must be strictly increasing")
                } else {
                    self.last_completed_scan_id = Some(scan_id);
                    completed_frame = Some(scan_id);
                    (true, "frame completion accepted")
                }
            }
            DownstreamRequest::GetStatus => (true, "status"),
            DownstreamRequest::Shutdown { consumer } => {
                if self.consumer.as_deref() != Some(consumer.as_str()) {
                    (false, "consumer is not registered")
                } else {
                    self.replay_state = ReplayState::ShuttingDown;
                    request_shutdown = true;
                    (true, "shutdown requested")
                }
            }
        };

        DownstreamEffect {
            response: DownstreamResponse {
                protocol_version: PROTOCOL_VERSION,
                replay_state: self.replay_state,
                tick_owner: self.tick_owner,
                accepted,
                unlocked_replay: unlock_replay,
                cloud_encoding: reported_cloud_encoding,
                message: message.to_owned(),
            },
            unlock_replay,
            request_shutdown,
            completed_frame,
        }
    }
}

#[cfg(test)]
mod cloud_encoding_tests {
    use super::*;

    fn register(gate: &mut DownstreamGate, accepted: Vec<CloudEncoding>) -> DownstreamResponse {
        gate.handle(DownstreamRequest::Register {
            consumer: "downstream".to_owned(),
            protocol_version: PROTOCOL_VERSION,
            tick_owner: TickOwner::Consumer,
            accepted_cloud_encodings: accepted,
        })
        .response
    }

    #[test]
    fn compression_is_used_only_when_both_sides_have_it() {
        let mut gate = DownstreamGate::new(CloudEncoding::CompressedPointCloud2);
        let response = register(
            &mut gate,
            vec![
                CloudEncoding::CompressedPointCloud2,
                CloudEncoding::PointCloud2,
            ],
        );

        assert!(response.accepted);
        assert_eq!(
            response.cloud_encoding,
            CloudEncoding::CompressedPointCloud2
        );
        assert_eq!(
            gate.negotiated_cloud_encoding(),
            CloudEncoding::CompressedPointCloud2
        );
    }

    #[test]
    fn a_consumer_without_compression_downgrades_instead_of_failing() {
        let mut gate = DownstreamGate::new(CloudEncoding::CompressedPointCloud2);
        let response = register(&mut gate, vec![CloudEncoding::PointCloud2]);

        assert!(response.accepted);
        assert_eq!(response.cloud_encoding, CloudEncoding::PointCloud2);
    }

    #[test]
    fn dropping_the_baseline_is_refused() {
        let mut gate = DownstreamGate::new(CloudEncoding::PointCloud2);
        let response = register(&mut gate, vec![CloudEncoding::CompressedPointCloud2]);

        assert!(!response.accepted);
        assert!(!gate.has_consumer());
    }

    #[test]
    fn capabilities_advertise_the_preference_before_anyone_registers() {
        let mut gate = DownstreamGate::new(CloudEncoding::CompressedPointCloud2);

        let capabilities = gate.handle(DownstreamRequest::GetCapabilities).response;
        assert_eq!(
            capabilities.cloud_encoding,
            CloudEncoding::CompressedPointCloud2
        );

        register(&mut gate, vec![CloudEncoding::PointCloud2]);
        let status = gate.handle(DownstreamRequest::GetStatus).response;
        assert_eq!(status.cloud_encoding, CloudEncoding::PointCloud2);
    }

    #[test]
    fn registered_consumer_can_complete_a_frame() {
        let mut gate = DownstreamGate::new(CloudEncoding::PointCloud2);
        assert!(register(&mut gate, vec![CloudEncoding::PointCloud2]).accepted);

        let effect = gate.handle(DownstreamRequest::CompleteFrame {
            consumer: "downstream".to_owned(),
            scan_id: 42,
        });

        assert!(effect.response.accepted);
        assert_eq!(effect.completed_frame, Some(42));
    }

    #[test]
    fn unregistered_consumer_cannot_complete_a_frame() {
        let mut gate = DownstreamGate::new(CloudEncoding::PointCloud2);
        let effect = gate.handle(DownstreamRequest::CompleteFrame {
            consumer: "stranger".to_owned(),
            scan_id: 42,
        });

        assert!(!effect.response.accepted);
        assert_eq!(effect.completed_frame, None);
    }

    #[test]
    fn frame_completions_must_be_strictly_increasing() {
        let mut gate = DownstreamGate::new(CloudEncoding::PointCloud2);
        assert!(register(&mut gate, vec![CloudEncoding::PointCloud2]).accepted);
        assert!(
            gate.handle(DownstreamRequest::CompleteFrame {
                consumer: "downstream".to_owned(),
                scan_id: 42,
            })
            .response
            .accepted
        );

        for scan_id in [42, 41] {
            let effect = gate.handle(DownstreamRequest::CompleteFrame {
                consumer: "downstream".to_owned(),
                scan_id,
            });
            assert!(!effect.response.accepted);
            assert_eq!(effect.completed_frame, None);
        }
    }
}
