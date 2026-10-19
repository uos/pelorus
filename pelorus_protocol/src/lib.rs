use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 3;
pub const REGISTERED_FRAME_TOPIC: &str = "pelorus/registered_frame";
pub const DOWNSTREAM_CONTROL_SERVICE: &str = "pelorus/downstream_control";
pub const DOWNSTREAM_READY_TOPIC: &str = "pelorus/downstream_ready";
pub const REPLAY_FINISHED_TOPIC: &str = "pelorus/replay_finished";
pub const STREAM_ENDED_TOPIC: &str = "pelorus/stream_ended";
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

/// Published once by Pelorus when a managed replay has reached the end of the
/// bag and every scan Minot replayed has been registered or given up on.
///
/// No frame follows. The consumer finishes once it has completed
/// `last_scan_id`, or at once when it is `None`, and then sends
/// [`DownstreamRequest::Shutdown`]. Pelorus stays up until it does, so the
/// consumer never has to guess from silence that the replay is over.
#[derive(
    Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize, Debug, Clone, PartialEq, Eq,
)]
pub struct ReplayFinished {
    pub protocol_version: u32,
    /// The last authoritative frame Pelorus published, if any.
    pub last_scan_id: Option<u64>,
}

/// Published once per forwarded topic (`--forward-topic`) when Minot has
/// replayed the last message of it in the bag, before that message itself.
/// A topic the bag does not have ends with zero messages, right at the start.
///
/// Topics travel separately, so the messages may still be on their way: a
/// consumer that counts what it receives on `topic` has the complete stream
/// once it has `messages` of them, and can stop waiting for more without
/// waiting for the end of the whole replay.
#[derive(
    Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize, Debug, Clone, PartialEq, Eq,
)]
pub struct StreamEnded {
    pub protocol_version: u32,
    pub topic: String,
    pub messages: u64,
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
    /// On `GetCapabilities` this is what Pelorus would prefer. On an accepted
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

    /// The bag is fully replayed and no further frame will be published.
    pub fn finish_replay(&mut self) {
        if self.replay_state == ReplayState::Running {
            self.replay_state = ReplayState::EndOfBag;
        }
    }

    pub fn has_consumer(&self) -> bool {
        self.consumer.is_some()
    }

    pub fn tick_owner(&self) -> TickOwner {
        self.tick_owner
    }

    /// The encoding to publish registered clouds with. Meaningful once a
    /// consumer has registered. The baseline before that.
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
