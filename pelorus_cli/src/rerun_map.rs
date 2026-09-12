use std::collections::{HashMap, HashSet};

use pelorus_core::RerunMapConfig;
use rerun::external::glam;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct ChunkKey(i64, i64, i64);

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct VoxelKey(i64, i64, i64);

#[derive(Clone, Copy, Debug)]
struct MapPoint {
    position: glam::Vec3,
    color: [u8; 3],
}

#[derive(Debug, Default)]
struct Chunk {
    points: HashMap<VoxelKey, MapPoint>,
    last_updated_scan: u64,
    dirty: bool,
}

/// A bounded visualization-only map assembled incrementally from registered scans.
///
/// This deliberately does not inspect the registration backend's map. Stable spatial
/// entity paths let Rerun replace only chunks that changed instead of retaining one
/// visible entity for every scan.
pub struct RerunChunkMap {
    config: RerunMapConfig,
    chunks: HashMap<ChunkKey, Chunk>,
    pending_clears: HashSet<ChunkKey>,
    last_publish_time: Option<f64>,
}

impl RerunChunkMap {
    pub fn new(config: RerunMapConfig) -> Self {
        Self {
            config,
            chunks: HashMap::new(),
            pending_clears: HashSet::new(),
            last_publish_time: None,
        }
    }

    pub fn add_scan(&mut self, positions: &[glam::Vec3], colors: &[[u8; 3]], scan_id: u64) {
        for (position, color) in positions.iter().zip(colors) {
            if !position.is_finite() {
                continue;
            }

            let chunk_key = self.chunk_key(*position);
            if !self.chunks.contains_key(&chunk_key) {
                self.make_room_for_chunk();
                self.pending_clears.remove(&chunk_key);
            }

            let voxel_key = self.voxel_key(*position);
            let chunk = self.chunks.entry(chunk_key).or_default();
            if !chunk.points.contains_key(&voxel_key)
                && chunk.points.len() >= self.config.max_points_per_chunk
            {
                continue;
            }
            chunk.points.insert(
                voxel_key,
                MapPoint {
                    position: *position,
                    color: *color,
                },
            );
            chunk.last_updated_scan = scan_id;
            chunk.dirty = true;
        }
    }

    pub fn publish_if_due(&mut self, rec: &rerun::RecordingStream, timestamp: f64) {
        let due = self
            .last_publish_time
            .map(|last| timestamp < last || timestamp - last >= self.config.publish_interval_sec)
            .unwrap_or(true);
        if !due {
            return;
        }
        self.last_publish_time = Some(timestamp);

        for key in std::mem::take(&mut self.pending_clears) {
            if let Err(err) = rec.log(Self::entity_path(key), &rerun::Clear::flat()) {
                log::error!("Error clearing evicted Rerun map chunk: {err}");
                self.pending_clears.insert(key);
            }
        }

        let dirty_keys: Vec<_> = self
            .chunks
            .iter()
            .filter_map(|(key, chunk)| chunk.dirty.then_some(*key))
            .collect();
        for key in dirty_keys {
            let Some(chunk) = self.chunks.get(&key) else {
                continue;
            };
            let mut positions = Vec::with_capacity(chunk.points.len());
            let mut colors = Vec::with_capacity(chunk.points.len());
            for point in chunk.points.values() {
                positions.push(point.position);
                colors.push(point.color);
            }

            let result = rec.log(
                Self::entity_path(key),
                &rerun::Points3D::update_fields()
                    .with_positions(positions)
                    .with_colors(colors)
                    .with_radii([rerun::Radius::new_ui_points(1.0)]),
            );
            match result {
                Ok(()) => {
                    if let Some(chunk) = self.chunks.get_mut(&key) {
                        chunk.dirty = false;
                    }
                }
                Err(err) => log::error!("Error publishing Rerun map chunk: {err}"),
            }
        }
    }

    fn make_room_for_chunk(&mut self) {
        if self.chunks.len() < self.config.max_chunks {
            return;
        }
        let Some(oldest) = self
            .chunks
            .iter()
            .min_by_key(|(_, chunk)| chunk.last_updated_scan)
            .map(|(key, _)| *key)
        else {
            return;
        };
        self.chunks.remove(&oldest);
        self.pending_clears.insert(oldest);
    }

    fn chunk_key(&self, point: glam::Vec3) -> ChunkKey {
        ChunkKey(
            (point.x / self.config.chunk_size).floor() as i64,
            (point.y / self.config.chunk_size).floor() as i64,
            (point.z / self.config.chunk_size).floor() as i64,
        )
    }

    fn voxel_key(&self, point: glam::Vec3) -> VoxelKey {
        VoxelKey(
            (point.x / self.config.voxel_size).floor() as i64,
            (point.y / self.config.voxel_size).floor() as i64,
            (point.z / self.config.voxel_size).floor() as i64,
        )
    }

    fn entity_path(key: ChunkKey) -> String {
        format!("world/map/chunks/x{}_y{}_z{}", key.0, key.1, key.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RerunMapConfig {
        RerunMapConfig {
            enabled: true,
            voxel_size: 1.0,
            chunk_size: 10.0,
            max_chunks: 2,
            max_points_per_chunk: 2,
            publish_interval_sec: 0.5,
        }
    }

    #[test]
    fn repeated_voxels_replace_instead_of_accumulating() {
        let mut map = RerunChunkMap::new(config());
        map.add_scan(&[glam::vec3(1.1, 2.0, 3.0)], &[[1, 2, 3]], 1);
        map.add_scan(&[glam::vec3(1.8, 2.0, 3.0)], &[[4, 5, 6]], 2);

        assert_eq!(map.chunks.len(), 1);
        assert_eq!(map.chunks.values().next().unwrap().points.len(), 1);
    }

    #[test]
    fn chunks_are_bounded_and_evictions_are_cleared() {
        let mut map = RerunChunkMap::new(config());
        map.add_scan(&[glam::vec3(1.0, 0.0, 0.0)], &[[0, 0, 0]], 1);
        map.add_scan(&[glam::vec3(11.0, 0.0, 0.0)], &[[0, 0, 0]], 2);
        map.add_scan(&[glam::vec3(21.0, 0.0, 0.0)], &[[0, 0, 0]], 3);

        assert_eq!(map.chunks.len(), 2);
        assert!(map.pending_clears.contains(&ChunkKey(0, 0, 0)));
    }

    #[test]
    fn points_per_chunk_are_bounded_but_existing_voxels_still_update() {
        let mut map = RerunChunkMap::new(config());
        map.add_scan(
            &[
                glam::vec3(1.0, 0.0, 0.0),
                glam::vec3(2.0, 0.0, 0.0),
                glam::vec3(3.0, 0.0, 0.0),
            ],
            &[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            1,
        );
        map.add_scan(&[glam::vec3(1.5, 0.0, 0.0)], &[[9, 9, 9]], 2);

        let chunk = map.chunks.values().next().unwrap();
        assert_eq!(chunk.points.len(), 2);
        assert_eq!(
            chunk.points.get(&VoxelKey(1, 0, 0)).unwrap().color,
            [9, 9, 9]
        );
    }

    #[test]
    fn negative_coordinates_use_floor_for_stable_chunks() {
        let map = RerunChunkMap::new(config());
        assert_eq!(
            map.chunk_key(glam::vec3(-0.1, 0.0, 0.0)),
            ChunkKey(-1, 0, 0)
        );
        assert_eq!(
            map.chunk_key(glam::vec3(-10.1, 0.0, 0.0)),
            ChunkKey(-2, 0, 0)
        );
    }
}
