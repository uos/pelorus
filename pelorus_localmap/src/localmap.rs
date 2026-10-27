#[cfg(feature = "rayon")]
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};

use crate::VoxelDistribution;
use alloc::{vec, vec::Vec};
use pelorus_core::adaptive_execution::ExecutionMode;

#[cfg(feature = "rayon")]
fn sort_unstable_by<T, F>(slice: &mut [T], cmp: F, mode: ExecutionMode)
where
    T: Send,
    F: Fn(&T, &T) -> core::cmp::Ordering + Sync,
{
    match mode {
        ExecutionMode::Sequential => slice.sort_unstable_by(cmp),
        ExecutionMode::Rayon => slice.par_sort_unstable_by(cmp),
    }
}

#[cfg(not(feature = "rayon"))]
fn sort_unstable_by<T, F>(slice: &mut [T], cmp: F, _mode: ExecutionMode)
where
    F: Fn(&T, &T) -> core::cmp::Ordering,
{
    slice.sort_unstable_by(cmp);
}

// Saving the hash to know what to delete
#[derive(Clone)]
struct Value((i32, VoxelDistribution));

#[derive(Copy, Clone, Default)]
struct SortKey {
    index: usize,
    temporal_dist: u64, // Numerator
    #[cfg_attr(feature = "lru-eviction", allow(dead_code))]
    point_count: u64, // Denominator
}

#[derive(Clone, Copy, Debug)]
pub struct VoxelDistributionLru {
    pub distribution: VoxelDistribution,
    pub lru_score: f32,
}

// LeastRelevantCacheIncrementalVoxelDistributionHashmap
pub struct Livodish {
    // all distributions in general
    buffer: Vec<Option<Value>>,
    add_cursor: usize,
    // ordering for lru. first index here is the buffer[idx] that should be removed next.
    // on removal, it always deletes the next idx from lru_cursor by setting it to none.
    // the indexes are built from the sorting
    lru_mapping: Vec<Option<usize>>,
    lru_mapping_keys: Vec<SortKey>,
    lru_cursor: usize,
    // lookup for hash to voxel idx in buffer
    lookup: crate::FxHashMap<i32, usize>,
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
            lookup: crate::FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            lru_cursor: 0,
            capacity,
            sensitivity,
            min_dist,
            lru_mapping_keys: vec![Default::default(); capacity],
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
    pub fn add(
        &mut self,
        dist: VoxelDistribution,
        hash: i32,
        scan_number: usize,
        execution_mode: ExecutionMode,
    ) {
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
                        self.build(scan_number, execution_mode);
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
                    self.lookup.insert(hash, must_be_deleted);
                } else {
                    // map not at capacity, clean add
                    self.buffer
                        .get_mut(self.add_cursor)
                        .expect("map must have been unexpectedly cleared before")
                        .replace(value);
                    self.lru_mapping
                        .get_mut(self.add_cursor)
                        .expect("map must have been unexpectedly cleared before")
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

    pub fn get_distributions(&self) -> Vec<VoxelDistribution> {
        self.buffer
            .iter()
            .filter_map(|v| Some(v.as_ref()?.0.1))
            .collect()
    }

    /// Export the stored keys as well as the distributions. Re-hashing a mean
    /// can change its key after weighted updates move it across a cell boundary.
    pub fn keyed_distributions(&self) -> impl Iterator<Item = (i32, &VoxelDistribution)> {
        self.buffer.iter().filter_map(|entry| {
            let Value((key, distribution)) = entry.as_ref()?;
            Some((*key, distribution))
        })
    }

    pub fn get_distributions_with_lru_scores(&self) -> Vec<VoxelDistributionLru> {
        let mut ranks = vec![self.capacity; self.capacity];
        for (rank, opt_idx) in self.lru_mapping.iter().enumerate() {
            if let Some(idx) = opt_idx
                && *idx < ranks.len()
            {
                ranks[*idx] = rank;
            }
        }

        let denom = self.capacity.max(1) as f32;
        self.buffer
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| {
                let distribution = v.as_ref()?.0.1;
                let rank = ranks.get(idx).copied().unwrap_or(self.capacity) as f32;
                Some(VoxelDistributionLru {
                    distribution,
                    lru_score: (rank / denom).clamp(0.0, 1.0),
                })
            })
            .collect()
    }

    pub fn find(&self, key: &i32) -> Option<VoxelDistribution> {
        let idx = self.lookup.get(key)?;
        unsafe { self.buffer.get_unchecked(*idx).as_ref() }.map(|v| v.0.1)
    }

    /// Re-arranges the map to be ready for new inserts.
    pub fn build(&mut self, scan_number: usize, execution_mode: ExecutionMode) {
        if self.add_cursor < self.capacity {
            return;
        }
        self.lru_cursor = 0;

        let update_key = |(&opt_idx, key): (&Option<usize>, &mut SortKey)| {
            opt_idx.map(|idx| {
                // SAFETY: safe because of the first line of this function.
                let item = unsafe {
                    &self
                        .buffer
                        .get_unchecked(idx)
                        .as_ref()
                        .unwrap_unchecked()
                        .0
                        .1
                };

                let temporal_dist = (scan_number.saturating_sub(item.last_updated_at_scan)) as u64;

                *key = SortKey {
                    index: idx,
                    temporal_dist,
                    point_count: item.point_count,
                };
            })
        };

        #[cfg(feature = "rayon")]
        match execution_mode {
            ExecutionMode::Sequential => {
                self.lru_mapping
                    .iter()
                    .zip(self.lru_mapping_keys.iter_mut())
                    .filter_map(update_key)
                    .for_each(drop);
            }
            ExecutionMode::Rayon => {
                self.lru_mapping
                    .par_iter()
                    .zip(self.lru_mapping_keys.par_iter_mut())
                    .filter_map(update_key)
                    .for_each(drop);
            }
        }
        #[cfg(not(feature = "rayon"))]
        self.lru_mapping
            .iter()
            .zip(self.lru_mapping_keys.iter_mut())
            .filter_map(update_key)
            .for_each(drop);

        // Ablation: plain LRU, evict the voxel updated longest ago first, ignoring support.
        #[cfg(feature = "lru-eviction")]
        let cmp = |a: &SortKey, b: &SortKey| b.temporal_dist.cmp(&a.temporal_dist);

        #[cfg(not(feature = "lru-eviction"))]
        let cmp = |a: &SortKey, b: &SortKey| {
            if a.point_count == 0 && b.point_count == 0 {
                return core::cmp::Ordering::Equal;
            }
            if a.point_count == 0 {
                return core::cmp::Ordering::Less; // a is Infinite -> comes first
            }
            if b.point_count == 0 {
                return core::cmp::Ordering::Greater;
            }

            // prevent overflow
            let left_val = (a.temporal_dist as u128) * (b.point_count as u128);
            let right_val = (b.temporal_dist as u128) * (a.point_count as u128);

            // descending sort
            right_val.cmp(&left_val)
        };

        sort_unstable_by(&mut self.lru_mapping_keys, cmp, execution_mode);

        for (i, key) in self.lru_mapping_keys.iter().enumerate() {
            unsafe { *self.lru_mapping.get_unchecked_mut(i) = Some(key.index) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Old but well supported (S = 10 / 100), recent but sparse (S = 2 / 1), then a new voxel.
    fn evicted_map() -> Livodish {
        let mut map = Livodish::new(2, 1.0, 0.0);
        for (hash, last_updated_at_scan, point_count) in [(0, 0, 100), (1, 8, 1)] {
            let dist = VoxelDistribution {
                point_count,
                last_updated_at_scan,
                ..VoxelDistribution::new()
            };
            map.add(dist, hash, last_updated_at_scan, ExecutionMode::Sequential);
        }
        map.build(10, ExecutionMode::Sequential);
        let fresh = VoxelDistribution {
            point_count: 1,
            last_updated_at_scan: 10,
            ..VoxelDistribution::new()
        };
        map.add(fresh, 2, 10, ExecutionMode::Sequential);
        map
    }

    #[cfg(not(feature = "lru-eviction"))]
    #[test]
    fn evicts_highest_age_per_support() {
        let map = evicted_map();
        assert!(map.find(&0).is_some());
        assert!(map.find(&1).is_none());
        assert!(map.find(&2).is_some());
    }

    #[cfg(feature = "lru-eviction")]
    #[test]
    fn evicts_least_recently_updated() {
        let map = evicted_map();
        assert!(map.find(&0).is_none());
        assert!(map.find(&1).is_some());
        assert!(map.find(&2).is_some());
    }
}
