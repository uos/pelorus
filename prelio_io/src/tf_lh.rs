/// Inlined tf_r2r crate as single module for R2R generated message compatibility over major versions.
///
/// tf_r2r is written by Arjo Chakravarty <arjo129@gmail.com>, Michael Maerz <maerz@magazino.eu> and Mitsuharu Kojima <kojima@smilerobotics.com>.
/// See this [repository](https://github.com/smilerobotics/tf_rosrust) or [crates.io](https://crates.io/crates/tf_r2r) for the original.
use nalgebra::geometry::{Isometry3, Translation3, UnitQuaternion};
use ratpub::{Node, Publisher};
use ros2_interfaces_jazzy_rkyv::{
    builtin_interfaces::msg::{Duration, Time},
    tf2_msgs::msg::TFMessage,
};
use ros2_interfaces_jazzy_rkyv::{
    geometry_msgs::msg::{Pose, Quaternion, Transform, TransformStamped, Vector3},
    std_msgs::msg::Header,
};
use std::collections::{HashMap, HashSet, VecDeque, hash_map::Entry};
use std::sync::{Arc, RwLock};
use thiserror::Error;

pub struct TfBroadcaster {
    publisher: Option<Publisher<TFMessage>>,
}
const TF_STATIC_TOPIC: &str = "/tf_static";
const TF_TOPIC: &str = "/tf";

impl TfBroadcaster {
    /// Create a new TfBroadcaster
    // TODO broadcasting is mute atm because a publisher can not reach the subscriber on the same node. They share the socket :)
    #[track_caller]
    pub fn new(_node: &Arc<Node>, _rt: Arc<tokio::runtime::Runtime>) -> Self {
        // Self {
        //     publisher: rt
        //         .block_on(async { node.create_publisher(TF_TOPIC.to_owned()).await.unwrap() }),
        // }
        Self { publisher: None }
    }

    /// Broadcast transform
    pub async fn send_transform(&self, tf: TransformStamped) -> Result<(), TfError> {
        let tf_message = TFMessage {
            transforms: vec![tf],
        };
        // TODO: handle error correctly
        if let Some(p) = self.publisher.as_ref() {
            p.publish(&tf_message)
                .await
                .map_err(|err| TfError::R2r(err.to_string()))?
        }
        Ok(())
    }
}

/// Enumerates the different types of errors
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TfError {
    /// Error due to looking up too far in the past. I.E the information is no longer available in the TF Cache.
    #[error("tf_r2r: AttemptedLookupInPast {:?} < {:?}",.0, .1)]
    AttemptedLookupInPast(Time, Box<TransformStamped>),
    /// Error due to the transform not yet being available.
    #[error("tf_r2r: AttemptedLookupInFuture {:?} < {:?}",.0, .1)]
    AttemptedLookUpInFuture(Box<TransformStamped>, Time),
    /// There is no path between the from and to frame.
    #[error("tf_r2r: CouldNotFindTransform {} -> {} ({:?})", .0, .1, .2)]
    CouldNotFindTransform(String, String, HashMap<String, HashSet<String>>),
    /// In the event that a write is simultaneously happening with a read of the same tf buffer
    #[error("tf_r2r: CouldNotAcquireLock")]
    CouldNotAcquireLock,
    /// Error of r2r
    #[error("tf_r2r: r2r error {:?}", .0)]
    R2r(String),
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub(crate) struct TfGraphNode {
    pub(crate) child: String,
    pub(crate) parent: String,
}

fn get_nanos(dur: Duration) -> i64 {
    i64::from(dur.sec) * 1_000_000_000 + i64::from(dur.nanosec)
}

fn binary_search_time(chain: &[TransformStamped], time: &Time) -> Result<usize, usize> {
    chain.binary_search_by(|element| {
        time_as_ns_i64(&element.header.stamp).cmp(&time_as_ns_i64(time))
    })
}

#[derive(Clone, Debug)]
pub(crate) struct TfIndividualTransformChain {
    cache_duration: Duration,
    static_tf: bool,
    // TODO: Implement a circular buffer. Current method is slow.
    pub(crate) transform_chain: Vec<TransformStamped>,
}

impl TfIndividualTransformChain {
    pub(crate) fn new(static_tf: bool, cache_duration: Duration) -> Self {
        Self {
            cache_duration,
            transform_chain: Vec::new(),
            static_tf,
        }
    }

    fn newest_stamp(&self) -> Option<Time> {
        self.transform_chain.last().map(|x| x.header.stamp.clone())
    }

    pub(crate) fn add_to_buffer(&mut self, msg: TransformStamped) {
        let index = binary_search_time(&self.transform_chain, &msg.header.stamp)
            .unwrap_or_else(|index| index);
        self.transform_chain.insert(index, msg.clone());

        if let Some(newest_stamp) = self.newest_stamp() {
            if is_time_later(
                &newest_stamp,
                &add_time_and_duration(&time_from_nanosec(0), &self.cache_duration),
            ) {
                let time_to_keep = sub_duration_from_time(&newest_stamp, &self.cache_duration);
                let index =
                    binary_search_time(&self.transform_chain, &time_to_keep).unwrap_or_else(|x| x);
                self.transform_chain.drain(..index);
            }
        }
    }

    /// If timestamp is zero, return the latest transform.
    pub(crate) fn get_closest_transform(&self, time: &Time) -> Result<TransformStamped, TfError> {
        if time_as_ns_i64(time) == 0 {
            return Ok(self.transform_chain.last().unwrap().clone());
        }

        if self.static_tf {
            return Ok(self.transform_chain.last().unwrap().clone());
        }

        match binary_search_time(&self.transform_chain, time) {
            Ok(x) => Ok(self.transform_chain.get(x).unwrap().clone()),
            Err(x) => {
                if x == 0 {
                    return Err(TfError::AttemptedLookupInPast(
                        time.clone(),
                        Box::new(self.transform_chain.first().unwrap().clone()),
                    ));
                }
                if x >= self.transform_chain.len() {
                    return Err(TfError::AttemptedLookUpInFuture(
                        Box::new(self.transform_chain.last().unwrap().clone()),
                        time.clone(),
                    ));
                }
                let tf1 = self.transform_chain.get(x - 1).unwrap().clone().transform;
                let tf2 = self.transform_chain.get(x).unwrap().clone().transform;
                let time1 = self
                    .transform_chain
                    .get(x - 1)
                    .unwrap()
                    .header
                    .stamp
                    .clone();
                let time2 = self.transform_chain.get(x).unwrap().header.stamp.clone();
                let header = self.transform_chain.get(x).unwrap().header.clone();
                let child_frame = self.transform_chain.get(x).unwrap().child_frame_id.clone();
                let total_duration = get_nanos(sub_time_and_time(&time2, &time1)) as f64;
                let desired_duration = get_nanos(sub_time_and_time(time, &time1)) as f64;
                let weight = 1.0 - desired_duration / total_duration;
                let final_tf = interpolate(tf1, tf2, weight);
                let ros_msg = to_transform_stamped(final_tf, header.frame_id, child_frame, time);
                Ok(ros_msg)
            }
        }
    }

    pub(crate) fn has_valid_transform(&self, time: &Time) -> bool {
        if self.transform_chain.is_empty() {
            return false;
        }

        if self.static_tf {
            return true;
        }

        let first = self.transform_chain.first().unwrap();
        let last = self.transform_chain.last().unwrap();

        time_as_ns_i64(time) == 0
            || is_time_in_range_eq(time, &first.header.stamp, &last.header.stamp)
    }
}

pub struct TfListener {
    buffer: Arc<RwLock<TfBuffer>>,
}

impl TfListener {
    /// Create a new TfListener
    #[track_caller]
    pub fn new(node: &Node, rt: Arc<tokio::runtime::Runtime>) -> Self {
        Self::new_with_buffer(node, TfBuffer::new(), rt)
    }

    #[track_caller]
    pub fn new_with_buffer(
        node: &Node,
        tf_buffer: TfBuffer,
        rt: Arc<tokio::runtime::Runtime>,
    ) -> Self {
        let buff = Arc::new(RwLock::new(tf_buffer));

        let mut dynamic_subscriber = rt.block_on(async {
            node.create_subscriber::<TFMessage>(TF_TOPIC.to_owned(), 100)
                .await
                .unwrap()
        });

        let buff_for_dynamic_sub = buff.clone();
        std::thread::spawn(move || {
            futures_lite::future::block_on(async {
                while Arc::strong_count(&buff_for_dynamic_sub) > 1 {
                    if let Some(tf) = dynamic_subscriber.next().await {
                        buff_for_dynamic_sub
                            .write()
                            .unwrap()
                            .handle_incoming_transforms(tf, false);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            });
        });

        let mut static_subscriber = rt.block_on(async {
            node.create_subscriber::<TFMessage>(TF_STATIC_TOPIC.to_owned(), 100)
                .await
                .unwrap()
        });

        let buff_for_static_sub = buff.clone();
        std::thread::spawn(move || {
            futures_lite::future::block_on(async {
                while Arc::strong_count(&buff_for_static_sub) > 1 {
                    if let Some(tf) = static_subscriber.next().await {
                        buff_for_static_sub
                            .write()
                            .unwrap()
                            .handle_incoming_transforms(tf, true);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            });
        });

        TfListener { buffer: buff }
    }

    /// Looks up a transform within the tree at a given time.
    pub fn lookup_transform(
        &self,
        from: &str,
        to: &str,
        time: Time,
    ) -> Result<TransformStamped, TfError> {
        self.buffer
            .read()
            .unwrap()
            .lookup_transform(from, to, &time)
    }

    /// Looks up a transform within the tree at a given time.
    pub fn lookup_transform_with_time_travel(
        &self,
        from: &str,
        time1: Time,
        to: &str,
        time2: Time,
        fixed_frame: &str,
    ) -> Result<TransformStamped, TfError> {
        self.buffer
            .read()
            .unwrap()
            .lookup_transform_with_time_travel(from, time1, to, time2, fixed_frame)
    }
}

pub fn isometry_from_pose(pose: &Pose) -> Isometry3<f64> {
    let trans = Translation3::new(pose.position.x, pose.position.y, pose.position.z);
    let rot = UnitQuaternion::new_normalize(nalgebra::geometry::Quaternion::new(
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
    ));

    Isometry3::from_parts(trans, rot)
}

pub fn isometry_from_transform(tf: &Transform) -> Isometry3<f64> {
    let trans = Translation3::new(tf.translation.x, tf.translation.y, tf.translation.z);
    let rot = UnitQuaternion::new_normalize(nalgebra::geometry::Quaternion::new(
        tf.rotation.w,
        tf.rotation.x,
        tf.rotation.y,
        tf.rotation.z,
    ));

    Isometry3::from_parts(trans, rot)
}

pub fn isometry_to_transform(iso: Isometry3<f64>) -> Transform {
    Transform {
        translation: Vector3 {
            x: iso.translation.x,
            y: iso.translation.y,
            z: iso.translation.z,
        },
        rotation: Quaternion {
            x: iso.rotation.i,
            y: iso.rotation.j,
            z: iso.rotation.k,
            w: iso.rotation.w,
        },
    }
}

pub fn get_inverse(trans: &TransformStamped) -> TransformStamped {
    TransformStamped {
        header: Header {
            stamp: trans.header.stamp.clone(),
            frame_id: trans.child_frame_id.clone(),
        },
        child_frame_id: trans.header.frame_id.clone(),
        transform: isometry_to_transform(
            isometry_from_transform(&trans.transform).clone().inverse(),
        ),
    }
}

///Chain multiple transforms together. Takes in a vector of transforms. The vector should be in order of desired transformations
pub fn chain_transforms(transforms: &[Transform]) -> Transform {
    let mut final_transform = Isometry3::identity();
    for t in transforms {
        let tf = isometry_from_transform(t);
        final_transform *= tf;
    }
    isometry_to_transform(final_transform)
}

pub fn interpolate(t1: Transform, t2: Transform, weight: f64) -> Transform {
    let r1 = nalgebra::geometry::Quaternion::new(
        t1.rotation.w,
        t1.rotation.x,
        t1.rotation.y,
        t1.rotation.z,
    );
    let r2 = nalgebra::geometry::Quaternion::new(
        t2.rotation.w,
        t2.rotation.x,
        t2.rotation.y,
        t2.rotation.z,
    );
    let r1 = UnitQuaternion::from_quaternion(r1);
    let r2 = UnitQuaternion::from_quaternion(r2);
    let res = r1.try_slerp(&r2, weight, 1e-9);
    match res {
        Some(qt) => Transform {
            translation: Vector3 {
                x: t1.translation.x * weight + t2.translation.x * (1.0 - weight),
                y: t1.translation.y * weight + t2.translation.y * (1.0 - weight),
                z: t1.translation.z * weight + t2.translation.z * (1.0 - weight),
            },
            rotation: Quaternion {
                x: qt.coords[0],
                y: qt.coords[1],
                z: qt.coords[2],
                w: qt.coords[3],
            },
        },
        None => {
            if weight > 0.5 {
                Transform {
                    translation: Vector3 {
                        x: t1.translation.x * weight + t2.translation.x * (1.0 - weight),
                        y: t1.translation.y * weight + t2.translation.y * (1.0 - weight),
                        z: t1.translation.z * weight + t2.translation.z * (1.0 - weight),
                    },
                    rotation: t1.rotation,
                }
            } else {
                Transform {
                    translation: Vector3 {
                        x: t1.translation.x * weight + t2.translation.x * (1.0 - weight),
                        y: t1.translation.y * weight + t2.translation.y * (1.0 - weight),
                        z: t1.translation.z * weight + t2.translation.z * (1.0 - weight),
                    },
                    rotation: t2.rotation,
                }
            }
        }
    }
}

pub(crate) fn to_transform_stamped(
    tf: Transform,
    from: std::string::String,
    to: std::string::String,
    time: &ros2_interfaces_jazzy_rkyv::builtin_interfaces::msg::Time,
) -> TransformStamped {
    TransformStamped {
        header: Header {
            frame_id: from,
            stamp: time.clone(),
        },
        child_frame_id: to,
        transform: tf,
    }
}

const BILLION: i64 = 1_000_000_000;

pub(crate) fn time_from_nanosec(t: i64) -> Time {
    Time {
        sec: (t / BILLION) as i32,
        nanosec: (t % BILLION) as u32,
    }
}

/// target - delta
pub(crate) fn sub_time_and_time(target: &Time, delta: &Time) -> Duration {
    if delta.sec.is_positive() {
        if target.nanosec >= delta.nanosec {
            Duration {
                sec: target.sec - delta.sec,
                nanosec: target.nanosec - delta.nanosec,
            }
        } else {
            Duration {
                sec: target.sec - delta.sec - 1,
                nanosec: BILLION as u32 + target.nanosec - delta.nanosec,
            }
        }
    } else {
        let nanosec = target.nanosec + delta.nanosec;
        let sec = target.sec - delta.sec + nanosec as i32 / BILLION as i32;
        let nanosec = nanosec % BILLION as u32;
        Duration { sec, nanosec }
    }
}

pub(crate) fn add_time_and_duration(t: &Time, d: &Duration) -> Time {
    let sec = t.sec + d.sec;
    let nanosec = t.nanosec + d.nanosec;

    Time {
        sec: sec + nanosec as i32 / BILLION as i32,
        nanosec: nanosec % BILLION as u32,
    }
}

pub(crate) fn sub_duration_from_time(t: &Time, d: &Duration) -> Time {
    let mut sec = t.sec - d.sec;
    let mut nanosec = t.nanosec as i64 - d.nanosec as i64;

    if nanosec.is_negative() {
        sec -= 1;
        nanosec += BILLION;
    }

    Time {
        sec,
        nanosec: nanosec as u32,
    }
}

pub(crate) fn time_as_ns_i64(t: &Time) -> i64 {
    t.sec as i64 * BILLION + t.nanosec as i64
}

pub(crate) fn is_time_in_range_eq(target: &Time, min: &Time, max: &Time) -> bool {
    let target_i64 = target.sec as i64 * BILLION + target.nanosec as i64;
    let min_i64 = min.sec as i64 * BILLION + min.nanosec as i64;
    let max_i64 = max.sec as i64 * BILLION + max.nanosec as i64;

    target_i64 >= min_i64 && target_i64 <= max_i64
}

pub(crate) fn is_time_later(target: &Time, past: &Time) -> bool {
    let target_i64 = target.sec as i64 * BILLION + target.nanosec as i64;
    let past_i64 = past.sec as i64 * BILLION + past.nanosec as i64;

    target_i64 > past_i64
}

#[derive(Clone, Debug)]
pub struct TfBuffer {
    child_transform_index: HashMap<String, HashSet<String>>,
    transform_data: HashMap<TfGraphNode, TfIndividualTransformChain>,
    cache_duration: Duration,
}

const DEFAULT_CACHE_DURATION_SECONDS: i32 = 10;

impl TfBuffer {
    pub(crate) fn new() -> Self {
        Self::new_with_duration(Duration {
            sec: DEFAULT_CACHE_DURATION_SECONDS,
            nanosec: 0,
        })
    }

    pub fn new_with_duration(cache_duration: Duration) -> Self {
        TfBuffer {
            child_transform_index: HashMap::new(),
            transform_data: HashMap::new(),
            cache_duration,
        }
    }

    pub(crate) fn handle_incoming_transforms(&mut self, transforms: TFMessage, static_tf: bool) {
        for transform in transforms.transforms {
            self.add_transform(&transform, static_tf);
            self.add_transform(&get_inverse(&transform), static_tf);
        }
    }

    fn add_transform(&mut self, transform: &TransformStamped, static_tf: bool) {
        //TODO: Detect is new transform will create a loop
        self.child_transform_index
            .entry(transform.header.frame_id.clone())
            .or_default()
            .insert(transform.child_frame_id.clone());

        let key = TfGraphNode {
            child: transform.child_frame_id.clone(),
            parent: transform.header.frame_id.clone(),
        };

        match self.transform_data.entry(key) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(TfIndividualTransformChain::new(
                static_tf,
                self.cache_duration.clone(),
            )),
        }
        .add_to_buffer(transform.clone());
    }

    /// Retrieves the transform path
    fn retrieve_transform_path(
        &self,
        from: String,
        to: String,
        time: &Time,
    ) -> Result<Vec<String>, TfError> {
        let mut res = vec![];
        let mut frontier: VecDeque<String> = VecDeque::new();
        let mut visited: HashSet<String> = HashSet::new();
        let mut parents: HashMap<String, String> = HashMap::new();
        visited.insert(from.clone());
        frontier.push_front(from.clone());

        while !frontier.is_empty() {
            let current_node = frontier.pop_front().unwrap();
            if current_node == to {
                break;
            }
            if let Some(children) = self.child_transform_index.get(&current_node) {
                for v in children {
                    if visited.contains(v) {
                        continue;
                    }

                    if self
                        .transform_data
                        .get(&TfGraphNode {
                            child: v.clone(),
                            parent: current_node.clone(),
                        })
                        .is_some_and(|chain| chain.has_valid_transform(time))
                    {
                        parents.insert(v.to_string(), current_node.clone());
                        frontier.push_front(v.to_string());
                        visited.insert(v.to_string());
                    }
                }
            }
        }
        let mut r = to.clone();
        while r != from {
            res.push(r.clone());
            let parent = parents.get(&r);

            match parent {
                Some(x) => r = x.to_string(),
                None => {
                    return Err(TfError::CouldNotFindTransform(
                        from,
                        to,
                        self.child_transform_index.clone(),
                    ));
                }
            }
        }
        res.reverse();
        Ok(res)
    }

    /// Looks up a transform within the tree at a given time.
    pub fn lookup_transform(
        &self,
        from: &str,
        to: &str,
        time: &Time,
    ) -> Result<TransformStamped, TfError> {
        let from = from.to_string();
        let to = to.to_string();
        let path = self.retrieve_transform_path(from.clone(), to.clone(), time);

        match path {
            Ok(path) => {
                let mut tf_list: Vec<Transform> = Vec::new();
                let mut first = from.clone();
                for intermediate in path {
                    let node = TfGraphNode {
                        child: intermediate.clone(),
                        parent: first.clone(),
                    };
                    let time_cache = self.transform_data.get(&node).unwrap();
                    let transform = time_cache.get_closest_transform(time);
                    match transform {
                        Err(e) => return Err(e),
                        Ok(x) => {
                            tf_list.push(x.transform);
                        }
                    }
                    first.clone_from(&intermediate);
                }
                let final_tf = chain_transforms(&tf_list);
                let msg = TransformStamped {
                    child_frame_id: to,
                    header: Header {
                        frame_id: from,
                        stamp: time.clone(),
                    },
                    transform: final_tf,
                };
                Ok(msg)
            }
            Err(x) => Err(x),
        }
    }

    pub(crate) fn lookup_transform_with_time_travel(
        &self,
        to: &str,
        time2: Time,
        from: &str,
        time1: Time,
        fixed_frame: &str,
    ) -> Result<TransformStamped, TfError> {
        let tf1 = self.lookup_transform(from, fixed_frame, &time1)?;
        let tf2 = self.lookup_transform(to, fixed_frame, &time2)?;
        let transforms = get_inverse(&tf1);
        let result = chain_transforms(&[tf2.transform, transforms.transform]);
        Ok(to_transform_stamped(
            result,
            from.to_string(),
            to.to_string(),
            &time1,
        ))
    }
}

#[cfg(test)]
mod test {
    use ros2_interfaces_jazzy_rkyv::{
        builtin_interfaces::msg::Time,
        geometry_msgs::msg::{Quaternion, Vector3},
    };

    use super::*;

    const PARENT: &str = "parent";
    const CHILD0: &str = "child0";
    const CHILD1: &str = "child1";

    use super::TfGraphNode;
    use std::collections::HashMap;

    #[test]
    fn test() {
        let mut hash_map = HashMap::new();
        hash_map.insert(
            TfGraphNode {
                child: "child0".to_owned(),
                parent: "parent".to_owned(),
            },
            0,
        );
        assert_eq!(hash_map.len(), 1);
        hash_map.insert(
            TfGraphNode {
                child: "child0".to_owned(),
                parent: "parent".to_owned(),
            },
            1,
        );
        assert_eq!(hash_map.len(), 1);

        let mut hash_map = HashMap::new();
        hash_map.insert(
            TfGraphNode {
                child: "child0".to_owned(),
                parent: "parent0".to_owned(),
            },
            0,
        );
        assert_eq!(hash_map.len(), 1);
        hash_map.insert(
            TfGraphNode {
                child: "parent0".to_owned(),
                parent: "child0".to_owned(),
            },
            0,
        );
        assert_eq!(hash_map.len(), 2);
        hash_map.insert(
            TfGraphNode {
                child: "child0".to_owned(),
                parent: "parent1".to_owned(),
            },
            0,
        );
        assert_eq!(hash_map.len(), 3);
    }

    /// This function builds a tree consisting of the following items:
    /// * a world coordinate frame
    /// * an item in the world frame at (1,0,0)
    /// * base_link of a robot starting at (0,0,0) and progressing at (0,t,0) where t is time in seconds
    /// * a camera which is (0.5, 0, 0) from the base_link
    fn build_test_tree(buffer: &mut TfBuffer, time: f64) {
        let nsecs = ((time - ((time.floor() as i64) as f64)) * 1E9) as u32;

        let world_to_item = TransformStamped {
            child_frame_id: "item".to_string(),
            header: Header {
                frame_id: "world".to_string(),
                stamp: Time {
                    sec: time.floor() as i32,
                    nanosec: nsecs,
                },
            },
            transform: Transform {
                rotation: Quaternion {
                    x: 0f64,
                    y: 0f64,
                    z: 0f64,
                    w: 1f64,
                },
                translation: Vector3 {
                    x: 1f64,
                    y: 0f64,
                    z: 0f64,
                },
            },
        };
        buffer.add_transform(&world_to_item, true);
        buffer.add_transform(&get_inverse(&world_to_item), true);

        let world_to_base_link = TransformStamped {
            child_frame_id: "base_link".to_string(),
            header: Header {
                frame_id: "world".to_string(),
                stamp: Time {
                    sec: time.floor() as i32,
                    nanosec: nsecs,
                },
            },
            transform: Transform {
                rotation: Quaternion {
                    x: 0f64,
                    y: 0f64,
                    z: 0f64,
                    w: 1f64,
                },
                translation: Vector3 {
                    x: 0f64,
                    y: time,
                    z: 0f64,
                },
            },
        };
        buffer.add_transform(&world_to_base_link, false);
        buffer.add_transform(&get_inverse(&world_to_base_link), false);

        let base_link_to_camera = TransformStamped {
            child_frame_id: "camera".to_string(),
            header: Header {
                frame_id: "base_link".to_string(),
                stamp: Time {
                    sec: time.floor() as i32,
                    nanosec: nsecs,
                },
            },
            transform: Transform {
                rotation: Quaternion {
                    x: 0f64,
                    y: 0f64,
                    z: 0f64,
                    w: 1f64,
                },
                translation: Vector3 {
                    x: 0.5f64,
                    y: 0f64,
                    z: 0f64,
                },
            },
        };
        buffer.add_transform(&base_link_to_camera, true);
        buffer.add_transform(&get_inverse(&base_link_to_camera), true);
    }

    /// Tests a basic lookup
    #[test]
    fn test_basic_tf_lookup() {
        let mut tf_buffer = TfBuffer::new();
        build_test_tree(&mut tf_buffer, 0f64);
        let res = tf_buffer.lookup_transform("camera", "item", &time_from_nanosec(0));
        let expected = TransformStamped {
            child_frame_id: "item".to_string(),
            header: Header {
                frame_id: "camera".to_string(),
                stamp: time_from_nanosec(0),
            },
            transform: Transform {
                rotation: Quaternion {
                    x: 0f64,
                    y: 0f64,
                    z: 0f64,
                    w: 1f64,
                },
                translation: Vector3 {
                    x: 0.5f64,
                    y: 0f64,
                    z: 0f64,
                },
            },
        };
        assert_eq!(res.unwrap(), expected);
    }

    /// Tests an interpolated lookup.
    #[test]
    fn test_basic_tf_interpolation() {
        let mut tf_buffer = TfBuffer::new();
        build_test_tree(&mut tf_buffer, 0f64);
        build_test_tree(&mut tf_buffer, 1f64);
        let res = tf_buffer.lookup_transform("camera", "item", &time_from_nanosec(700_000_000));
        let expected = TransformStamped {
            child_frame_id: "item".to_string(),
            header: Header {
                frame_id: "camera".to_string(),
                stamp: time_from_nanosec(700_000_000),
            },
            transform: Transform {
                rotation: Quaternion {
                    x: 0f64,
                    y: 0f64,
                    z: 0f64,
                    w: 1f64,
                },
                translation: Vector3 {
                    x: 0.5f64,
                    y: -0.7f64,
                    z: 0f64,
                },
            },
        };
        assert_eq!(res.unwrap(), expected);
    }

    /// Tests an interpolated lookup.
    #[test]
    fn test_basic_tf_time_travel() {
        let mut tf_buffer = TfBuffer::new();
        build_test_tree(&mut tf_buffer, 0f64);
        build_test_tree(&mut tf_buffer, 1f64);
        let res = tf_buffer.lookup_transform_with_time_travel(
            "camera",
            time_from_nanosec(400_000_000),
            "camera",
            time_from_nanosec(700_000_000),
            "item",
        );
        let expected = TransformStamped {
            child_frame_id: "camera".to_string(),
            header: Header {
                frame_id: "camera".to_string(),
                stamp: time_from_nanosec(700_000_000),
            },
            transform: Transform {
                rotation: Quaternion {
                    x: 0f64,
                    y: 0f64,
                    z: 0f64,
                    w: 1f64,
                },
                translation: Vector3 {
                    x: 0f64,
                    y: 0.3f64,
                    z: 0f64,
                },
            },
        };
        assert_approx_eq(res.unwrap(), expected);
    }

    #[test]
    fn test_add_transform() {
        let mut tf_buffer = TfBuffer::new();
        let transform00 = TransformStamped {
            header: Header {
                frame_id: PARENT.to_string(),
                stamp: time_from_nanosec(0),
            },
            child_frame_id: CHILD0.to_string(),
            ..Default::default()
        };
        let transform01 = TransformStamped {
            header: Header {
                frame_id: PARENT.to_string(),
                stamp: Time { sec: 1, nanosec: 0 },
            },
            child_frame_id: CHILD0.to_string(),
            ..Default::default()
        };
        let transform1 = TransformStamped {
            header: Header {
                frame_id: PARENT.to_string(),
                ..Default::default()
            },
            child_frame_id: CHILD1.to_string(),
            ..Default::default()
        };
        let transform0_key = TfGraphNode {
            child: CHILD0.to_owned(),
            parent: PARENT.to_owned(),
        };
        let transform1_key = TfGraphNode {
            child: CHILD1.to_owned(),
            parent: PARENT.to_owned(),
        };
        let static_tf = true;
        tf_buffer.add_transform(&transform00, static_tf);
        assert_eq!(tf_buffer.child_transform_index.len(), 1);
        assert!(tf_buffer.child_transform_index.contains_key(PARENT));
        let children = tf_buffer.child_transform_index.get(PARENT).unwrap();
        assert_eq!(children.len(), 1);
        assert!(children.contains(CHILD0));
        assert_eq!(tf_buffer.transform_data.len(), 1);
        assert!(tf_buffer.transform_data.contains_key(&transform0_key));
        let data = tf_buffer.transform_data.get(&transform0_key);
        assert!(data.is_some());
        assert_eq!(data.unwrap().transform_chain.len(), 1);

        tf_buffer.add_transform(&transform01, static_tf);
        assert_eq!(tf_buffer.child_transform_index.len(), 1);
        assert!(tf_buffer.child_transform_index.contains_key(PARENT));
        let children = tf_buffer.child_transform_index.get(PARENT).unwrap();
        assert_eq!(children.len(), 1);
        assert!(children.contains(CHILD0));
        assert_eq!(tf_buffer.transform_data.len(), 1);
        assert!(tf_buffer.transform_data.contains_key(&transform0_key));
        let data = tf_buffer.transform_data.get(&transform0_key);
        assert!(data.is_some());
        assert_eq!(data.unwrap().transform_chain.len(), 2);

        tf_buffer.add_transform(&transform1, static_tf);
        assert_eq!(tf_buffer.child_transform_index.len(), 1);
        assert!(tf_buffer.child_transform_index.contains_key(PARENT));
        let children = tf_buffer.child_transform_index.get(PARENT).unwrap();
        assert_eq!(children.len(), 2);
        assert!(children.contains(CHILD0));
        assert!(children.contains(CHILD1));
        assert_eq!(tf_buffer.transform_data.len(), 2);
        assert!(tf_buffer.transform_data.contains_key(&transform0_key));
        assert!(tf_buffer.transform_data.contains_key(&transform1_key));
        let data = tf_buffer.transform_data.get(&transform0_key);
        assert!(data.is_some());
        assert_eq!(data.unwrap().transform_chain.len(), 2);
        let data = tf_buffer.transform_data.get(&transform1_key);
        assert!(data.is_some());
        assert_eq!(data.unwrap().transform_chain.len(), 1);
    }

    #[test]
    fn test_cache_duration() {
        let mut tf_buffer = TfBuffer::new_with_duration(Duration { sec: 1, nanosec: 0 });
        let transform00 = TransformStamped {
            header: Header {
                frame_id: PARENT.to_string(),
                stamp: time_from_nanosec(0),
            },
            child_frame_id: CHILD0.to_string(),
            ..Default::default()
        };
        let transform01 = TransformStamped {
            header: Header {
                frame_id: PARENT.to_string(),
                stamp: Time { sec: 1, nanosec: 0 },
            },
            child_frame_id: CHILD0.to_string(),
            ..Default::default()
        };
        let transform02 = TransformStamped {
            header: Header {
                frame_id: PARENT.to_string(),
                stamp: Time { sec: 2, nanosec: 0 },
            },
            child_frame_id: CHILD0.to_string(),
            ..Default::default()
        };
        let transform0_key = TfGraphNode {
            child: CHILD0.to_owned(),
            parent: PARENT.to_owned(),
        };

        let static_tf = true;
        tf_buffer.add_transform(&transform00, static_tf);
        assert_eq!(tf_buffer.child_transform_index.len(), 1);
        assert_eq!(tf_buffer.transform_data.len(), 1);
        assert!(tf_buffer.transform_data.contains_key(&transform0_key));
        let data = tf_buffer.transform_data.get(&transform0_key);
        assert!(data.is_some());
        assert_eq!(data.unwrap().transform_chain.len(), 1);
        assert_eq!(
            data.unwrap().transform_chain[0].header.stamp,
            time_from_nanosec(0)
        );

        tf_buffer.add_transform(&transform01, static_tf);
        assert_eq!(tf_buffer.child_transform_index.len(), 1);
        assert_eq!(tf_buffer.transform_data.len(), 1);
        assert!(tf_buffer.transform_data.contains_key(&transform0_key));
        let data = tf_buffer.transform_data.get(&transform0_key);
        assert!(data.is_some());
        assert_eq!(data.unwrap().transform_chain.len(), 2);
        assert_eq!(
            data.unwrap().transform_chain[0].header.stamp,
            time_from_nanosec(0)
        );
        assert_eq!(
            data.unwrap().transform_chain[1].header.stamp,
            time_from_nanosec(1_000_000_000)
        );

        tf_buffer.add_transform(&transform02, static_tf);
        assert_eq!(tf_buffer.child_transform_index.len(), 1);
        assert_eq!(tf_buffer.transform_data.len(), 1);
        assert!(tf_buffer.transform_data.contains_key(&transform0_key));
        let data = tf_buffer.transform_data.get(&transform0_key);
        assert!(data.is_some());
        assert_eq!(data.unwrap().transform_chain.len(), 2);
        assert_eq!(
            data.unwrap().transform_chain[0].header.stamp,
            time_from_nanosec(1_000_000_000)
        );
        assert_eq!(
            data.unwrap().transform_chain[1].header.stamp,
            time_from_nanosec(2_000_000_000)
        );
    }

    fn assert_approx_eq(msg1: TransformStamped, msg2: TransformStamped) {
        assert_eq!(msg1.header, msg2.header);
        assert_eq!(msg1.child_frame_id, msg2.child_frame_id);

        assert!((msg1.transform.rotation.x - msg2.transform.rotation.x).abs() < 1e-9);
        assert!((msg1.transform.rotation.y - msg2.transform.rotation.y).abs() < 1e-9);
        assert!((msg1.transform.rotation.z - msg2.transform.rotation.z).abs() < 1e-9);
        assert!((msg1.transform.rotation.w - msg2.transform.rotation.w).abs() < 1e-9);

        assert!((msg1.transform.translation.x - msg2.transform.translation.x).abs() < 1e-9);
        assert!((msg1.transform.translation.y - msg2.transform.translation.y).abs() < 1e-9);
        assert!((msg1.transform.translation.z - msg2.transform.translation.z).abs() < 1e-9);
    }

    #[test]
    fn test_time_from_nanosec() {
        const EXPECTED_TIME: Time = Time {
            sec: 1,
            nanosec: 234_567_890,
        };
        let nanosec = 1_234_567_890;
        let time = time_from_nanosec(nanosec);

        assert_eq!(time.sec, EXPECTED_TIME.sec);
        assert_eq!(time.nanosec, EXPECTED_TIME.nanosec);
    }

    #[test]
    fn test_sub_time_and_time() {
        const EXPECTED_DIF1: Time = Time {
            sec: 0,
            nanosec: 111_111_011,
        };
        const EXPECTED_DIF2: Time = Time {
            sec: 0,
            nanosec: 888_889_889,
        };
        const EXPECTED_DIF3: Time = Time {
            sec: 20,
            nanosec: 691_357_802,
        };

        let (t1, t2, t3, t4, _, _) = times_and_durations_for_test();

        let dif1 = sub_time_and_time(&t2, &t1);
        let dif2 = sub_time_and_time(&t2, &t3);
        let dif3 = sub_time_and_time(&t2, &t4);

        assert_eq!(dif1.sec, EXPECTED_DIF1.sec);
        assert_eq!(dif1.nanosec, EXPECTED_DIF1.nanosec);
        assert_eq!(dif2.sec, EXPECTED_DIF2.sec);
        assert_eq!(dif2.nanosec, EXPECTED_DIF2.nanosec);
        assert_eq!(dif3.sec, EXPECTED_DIF3.sec);
        assert_eq!(dif3.nanosec, EXPECTED_DIF3.nanosec);
    }

    #[test]
    fn test_add_time_and_duration() {
        const EXPECTED_SUM1: Time = Time {
            sec: 11,
            nanosec: 334_567_890,
        };
        const EXPECTED_SUM2: Time = Time {
            sec: 12,
            nanosec: 234_567_889,
        };

        let (t1, _, _, _, d1, d2) = times_and_durations_for_test();

        let sum1 = add_time_and_duration(&t1, &d1);
        let sum2 = add_time_and_duration(&t1, &d2);

        assert_eq!(sum1.sec, EXPECTED_SUM1.sec);
        assert_eq!(sum1.nanosec, EXPECTED_SUM1.nanosec);
        assert_eq!(sum2.sec, EXPECTED_SUM2.sec);
        assert_eq!(sum2.nanosec, EXPECTED_SUM2.nanosec);
    }

    #[test]
    fn test_sub_duration_from_time() {
        const EXPECTED_DIF1: Time = Time {
            sec: 9,
            nanosec: 134_567_890,
        };
        const EXPECTED_DIF2: Time = Time {
            sec: 8,
            nanosec: 234_567_891,
        };

        let (t1, _, _, _, d1, d2) = times_and_durations_for_test();

        let dif1 = sub_duration_from_time(&t1, &d1);
        let dif2 = sub_duration_from_time(&t1, &d2);

        assert_eq!(dif1.sec, EXPECTED_DIF1.sec);
        assert_eq!(dif1.nanosec, EXPECTED_DIF1.nanosec);
        assert_eq!(dif2.sec, EXPECTED_DIF2.sec);
        assert_eq!(dif2.nanosec, EXPECTED_DIF2.nanosec);
    }

    #[test]
    fn test_time_as_ns_i64() {
        const EXPECTED: i64 = 10_234_567_890;
        let (t1, _, _, _, _, _) = times_and_durations_for_test();

        let time_as_nanosec = time_as_ns_i64(&t1);

        assert_eq!(time_as_nanosec, EXPECTED);
    }

    #[test]
    fn test_is_time_in_range_eq() {
        let (t1, t2, t3, _, _, _) = times_and_durations_for_test();

        assert!(is_time_in_range_eq(&t1, &t3, &t2));
        assert!(!is_time_in_range_eq(&t2, &t3, &t1));
    }

    fn times_and_durations_for_test() -> (Time, Time, Time, Time, Duration, Duration) {
        let time1 = Time {
            sec: 10,
            nanosec: 234_567_890,
        };
        let time2 = Time {
            sec: 10,
            nanosec: 345_678_901,
        };
        let time3 = Time {
            sec: 9,
            nanosec: 456_789_012,
        };
        let time4 = Time {
            sec: -10,
            nanosec: 345_678_901,
        };
        let duration1 = Duration {
            sec: 1,
            nanosec: 100_000_000,
        };
        let duration2 = Duration {
            sec: 1,
            nanosec: 999_999_999,
        };

        (time1, time2, time3, time4, duration1, duration2)
    }
    #[test]
    fn test_basic_translation_chaining() {
        let tf1 = Transform {
            translation: Vector3 {
                x: 1f64,
                y: 1f64,
                z: 0f64,
            },
            rotation: Quaternion {
                x: 0f64,
                y: 0f64,
                z: 0f64,
                w: 1f64,
            },
        };
        let expected_tf = Transform {
            translation: Vector3 {
                x: 2f64,
                y: 2f64,
                z: 0f64,
            },
            rotation: Quaternion {
                x: 0f64,
                y: 0f64,
                z: 0f64,
                w: 1f64,
            },
        };
        let transform_chain = vec![tf1.clone(), tf1];
        let res = chain_transforms(&transform_chain);
        assert_eq!(res, expected_tf);
    }

    #[test]
    fn test_basic_interpolation() {
        let tf1 = Transform {
            translation: Vector3 {
                x: 1f64,
                y: 1f64,
                z: 0f64,
            },
            rotation: Quaternion {
                x: 0f64,
                y: 0f64,
                z: 0f64,
                w: 1f64,
            },
        };
        let tf2 = Transform {
            translation: Vector3 {
                x: 2f64,
                y: 2f64,
                z: 0f64,
            },
            rotation: Quaternion {
                x: 0f64,
                y: 0f64,
                z: 0f64,
                w: 1f64,
            },
        };
        let expected = Transform {
            translation: Vector3 {
                x: 1.5f64,
                y: 1.5f64,
                z: 0f64,
            },
            rotation: Quaternion {
                x: 0f64,
                y: 0f64,
                z: 0f64,
                w: 1f64,
            },
        };
        assert_eq!(interpolate(tf1, tf2, 0.5), expected);
    }

    /// Tests a case in which the tree structure changes dynamically
    /// time 1-2(sec): [base] -> [camera1] -> [marker] -> [target]
    /// time 3-4(sec): [base] -> [camera2] -> [marker] -> [target]
    /// time 5-6(sec): [base] -> [camera1] -> [marker] -> [target]
    #[test]
    fn test_dynamic_tree() {
        let mut tf_buffer = TfBuffer::new();

        let base_to_camera1 = TransformStamped {
            child_frame_id: "camera1".to_string(),
            header: Header {
                frame_id: "base".to_string(),
                stamp: Time { sec: 1, nanosec: 0 },
            },
            transform: Transform {
                rotation: Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                translation: Vector3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
            },
        };
        tf_buffer.add_transform(&base_to_camera1, true);
        tf_buffer.add_transform(&get_inverse(&base_to_camera1), true);

        let base_to_camera2 = TransformStamped {
            child_frame_id: "camera2".to_string(),
            header: Header {
                frame_id: "base".to_string(),
                stamp: Time { sec: 1, nanosec: 0 },
            },
            transform: Transform {
                rotation: Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                    w: 0.0,
                },
                translation: Vector3 {
                    x: -1.0,
                    y: 0.0,
                    z: 0.0,
                },
            },
        };
        tf_buffer.add_transform(&base_to_camera2, true);
        tf_buffer.add_transform(&get_inverse(&base_to_camera2), true);

        let marker_to_target = TransformStamped {
            child_frame_id: "target".to_string(),
            header: Header {
                frame_id: "marker".to_string(),
                stamp: Time { sec: 1, nanosec: 0 },
            },
            transform: Transform {
                rotation: Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                translation: Vector3 {
                    x: -0.5,
                    y: 0.0,
                    z: 0.0,
                },
            },
        };
        tf_buffer.add_transform(&marker_to_target, true);
        tf_buffer.add_transform(&get_inverse(&marker_to_target), true);

        let mut camera1_to_marker = TransformStamped {
            child_frame_id: "marker".to_string(),
            header: Header {
                frame_id: "camera1".to_string(),
                stamp: Time { sec: 1, nanosec: 0 },
            },
            transform: Transform {
                rotation: Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                translation: Vector3 {
                    x: 1.0,
                    y: 1.0,
                    z: 0.0,
                },
            },
        };
        tf_buffer.add_transform(&camera1_to_marker, false);
        tf_buffer.add_transform(&get_inverse(&camera1_to_marker), false);

        camera1_to_marker.header.stamp.sec = 2;
        camera1_to_marker.transform.translation.y = -1.0;
        tf_buffer.add_transform(&camera1_to_marker, false);
        tf_buffer.add_transform(&get_inverse(&camera1_to_marker), false);

        let mut camera2_to_marker = TransformStamped {
            child_frame_id: "marker".to_string(),
            header: Header {
                frame_id: "camera2".to_string(),
                stamp: Time { sec: 3, nanosec: 0 },
            },
            transform: Transform {
                rotation: Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                translation: Vector3 {
                    x: 1.0,
                    y: 1.0,
                    z: 0.0,
                },
            },
        };
        tf_buffer.add_transform(&camera2_to_marker, false);
        tf_buffer.add_transform(&get_inverse(&camera2_to_marker), false);

        camera2_to_marker.header.stamp.sec = 4;
        camera2_to_marker.transform.translation.y = -1.0;
        tf_buffer.add_transform(&camera2_to_marker, false);
        tf_buffer.add_transform(&get_inverse(&camera2_to_marker), false);

        let result = tf_buffer.lookup_transform("base", "target", &Time { sec: 1, nanosec: 0 });
        assert_eq!(
            result.unwrap().transform.translation,
            Vector3 {
                x: 1.5,
                y: 1.0,
                z: 0.0
            }
        );

        let result = tf_buffer.lookup_transform(
            "base",
            "target",
            &Time {
                sec: 1,
                nanosec: 500_000_000,
            },
        );
        assert_eq!(
            result.unwrap().transform.translation,
            Vector3 {
                x: 1.5,
                y: 0.0,
                z: 0.0
            }
        );

        let result = tf_buffer.lookup_transform("base", "target", &Time { sec: 2, nanosec: 0 });
        assert_eq!(
            result.unwrap().transform.translation,
            Vector3 {
                x: 1.5,
                y: -1.0,
                z: 0.0
            }
        );

        let result = tf_buffer.lookup_transform(
            "base",
            "target",
            &Time {
                sec: 2,
                nanosec: 500_000_000,
            },
        );
        assert!(result.is_err());

        let result = tf_buffer.lookup_transform("base", "target", &Time { sec: 3, nanosec: 0 });
        assert_eq!(
            result.unwrap().transform.translation,
            Vector3 {
                x: -1.5,
                y: -1.0,
                z: 0.0
            }
        );

        let result = tf_buffer.lookup_transform(
            "base",
            "target",
            &Time {
                sec: 3,
                nanosec: 500_000_000,
            },
        );
        assert_eq!(
            result.unwrap().transform.translation,
            Vector3 {
                x: -1.5,
                y: -0.0,
                z: 0.0
            }
        );

        let result = tf_buffer.lookup_transform("base", "target", &Time { sec: 4, nanosec: 0 });
        assert_eq!(
            result.unwrap().transform.translation,
            Vector3 {
                x: -1.5,
                y: 1.0,
                z: 0.0
            }
        );

        let result = tf_buffer.lookup_transform(
            "base",
            "target",
            &Time {
                sec: 4,
                nanosec: 500_000_000,
            },
        );
        assert!(result.is_err());

        camera1_to_marker.header.stamp.sec = 5;
        camera1_to_marker.transform.translation.x = 0.5;
        camera1_to_marker.transform.translation.y = 1.0;
        tf_buffer.add_transform(&camera1_to_marker, false);
        tf_buffer.add_transform(&get_inverse(&camera1_to_marker), false);

        camera1_to_marker.header.stamp.sec = 6;
        camera1_to_marker.transform.translation.y = -1.0;
        tf_buffer.add_transform(&camera1_to_marker, false);
        tf_buffer.add_transform(&get_inverse(&camera1_to_marker), false);

        let result = tf_buffer.lookup_transform("base", "target", &Time { sec: 5, nanosec: 0 });
        assert_eq!(
            result.unwrap().transform.translation,
            Vector3 {
                x: 1.0,
                y: 1.0,
                z: 0.0
            }
        );

        let result = tf_buffer.lookup_transform(
            "base",
            "target",
            &Time {
                sec: 5,
                nanosec: 500_000_000,
            },
        );
        assert_eq!(
            result.unwrap().transform.translation,
            Vector3 {
                x: 1.0,
                y: 0.0,
                z: 0.0
            }
        );

        let result = tf_buffer.lookup_transform("base", "target", &Time { sec: 6, nanosec: 0 });
        assert_eq!(
            result.unwrap().transform.translation,
            Vector3 {
                x: 1.0,
                y: -1.0,
                z: 0.0
            }
        );
    }
}
