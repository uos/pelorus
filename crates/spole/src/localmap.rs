#[cfg(feature = "parallel")]
use rayon::slice::ParallelSliceMut;

use crate::VoxelDistribution;

// Saving the hash to know what to delete
#[derive(Clone)]
struct Value((i32, VoxelDistribution));

// LeastRecentlyUsedCacheIncrementalVoxelDistributionHashmap
pub struct Livodish {
    // all distributions in general
    buffer: Vec<Option<Value>>,
    add_cursor: usize,
    // ordering for lru. first index here is the buffer[idx] that should be removed next.
    // on removal, it always deletes the next idx from lru_cursor by setting it to none.
    // the indexes are built from the sorting
    lru_mapping: Vec<Option<usize>>,
    lru_cursor: usize,
    // lookup for hash to voxel idx in buffer
    lookup: rustc_hash::FxHashMap<i32, usize>,
    capacity: usize,
    sensitivity: f32,
    min_dist: f32,
}

impl Livodish {
    pub fn new(capacity: usize, sensitivity: f32, min_dist: f32) -> Self {
        Self {
            buffer: vec![None; capacity],
            lru_mapping: vec![None; capacity],
            add_cursor: 0,
            lookup: rustc_hash::FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            lru_cursor: 0,
            capacity,
            sensitivity,
            min_dist,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    pub fn size(&self) -> usize {
        self.add_cursor
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.lookup.clear();
        self.lru_mapping.clear();
        self.add_cursor = 0;
        self.lru_cursor = 0;
    }

    /// Insert at the beginning, since the ones in the beginning are the ones least used
    pub fn add(&mut self, dist: VoxelDistribution, hash: i32, scan_number: usize) {
        let entry = self.lookup.get_mut(&hash);

        match entry {
            Some(buff_idx) => {
                // exists, update
                let entry = unsafe {
                    self.buffer
                        .get_unchecked_mut(*buff_idx)
                        .as_mut()
                        .expect("unresolved ref")
                };

                entry
                    .0
                    .1
                    .update(&dist, self.sensitivity, self.min_dist, scan_number);
            }
            None => {
                let value = Value((hash, dist));
                if self.add_cursor >= self.capacity {
                    // full but new, make space
                    if self.lru_cursor > self.capacity {
                        log::warn!(
                            "Replacing too many points inside LRU Cache. Increase the map capacity!"
                        );
                        return;
                    }
                    let lru_never_run = self.lru_mapping.is_empty();
                    if lru_never_run {
                        self.build(scan_number);
                    }
                    let must_be_deleted = unsafe {
                        self.lru_mapping
                            .get_unchecked_mut(self.lru_cursor)
                            .expect("none at cursor but full map")
                    };
                    self.lru_cursor += 1;
                    let del = unsafe {
                        self.buffer
                            .get_unchecked_mut(must_be_deleted)
                            .as_mut()
                            .expect("unresolved ref")
                    };
                    let del_key = del.0.0;
                    self.lookup
                        .remove(&del_key)
                        .expect("voxel not found in lookup");

                    // add new inplace
                    *del = value;
                    // lru_mapping keeps idx, since inplace update
                    // lookup keeps index but changes hash
                    self.lookup.insert(hash, must_be_deleted);

                    // self.lru_cursor += 1; // TODO check should be wrong
                } else {
                    // map not at capacity, clean add
                    unsafe { self.buffer.get_unchecked_mut(self.add_cursor) }.replace(value);
                    unsafe { self.lru_mapping.get_unchecked_mut(self.add_cursor) }
                        .replace(self.add_cursor);
                    self.lookup.insert(hash, self.add_cursor);
                    self.add_cursor += 1;
                }
            }
        }
    }

    pub fn get_map(&self) -> Vec<nalgebra::Point3<f32>> {
        self.buffer
            .iter()
            .filter_map(|v| Some(v.as_ref()?.0.1.mean))
            .collect()
    }

    pub fn find(&self, key: &i32) -> Option<VoxelDistribution> {
        let idx = self.lookup.get(key)?;
        unsafe { self.buffer.get_unchecked(*idx).as_ref() }.map(|v| v.0.1)
    }

    /// Re-arranges the map to be ready for new inserts.
    pub fn build(&mut self, scan_number: usize) {
        if self.add_cursor >= self.capacity {
            self.lru_cursor = 0;
            let cmp = |lh: &Option<usize>, rh: &Option<usize>| match (lh, rh) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (Some(_), None) => std::cmp::Ordering::Greater,
                (Some(l), Some(r)) => {
                    let l = &self.buffer[*l].as_ref().expect("unresolved ref").0.1;
                    let r = &self.buffer[*r].as_ref().expect("unresolved ref").0.1;

                    let calculate_score = |last_updated_at_scan: usize, point_count: u64| -> f64 {
                        if point_count == 0 {
                            return f64::INFINITY;
                        }
                        let temporal_dist = scan_number.saturating_sub(last_updated_at_scan);
                        (temporal_dist as f64) / (point_count as f64)
                    };

                    let score_l = calculate_score(l.last_updated_at_scan, l.point_count);
                    let score_r = calculate_score(r.last_updated_at_scan, r.point_count);
                    score_r
                        .partial_cmp(&score_l)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
            };
            #[cfg(feature = "parallel")]
            {
                self.lru_mapping.par_sort_unstable_by(cmp);
            }
            #[cfg(not(feature = "parallel"))]
            {
                self.lru_mapping.sort_unstable_by(cmp);
            }
        }
    }
}
