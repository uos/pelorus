pub const COMMON: &str = include_str!("common.comp");
pub const HASHMAP: &str = include_str!("hashmap.comp");
pub const LOCALMAP: &str = include_str!("localmap.comp");
pub const REORDER: &str = include_str!("reorder.comp");
pub const PLANES: &str = include_str!("planes.comp");
pub const COMPACT_PLANES: &str = include_str!("compact_planes.comp");
pub const PLANE_COUNT: &str = include_str!("plane_count.comp");
pub const PORTABLE_SORT: &str = include_str!("portable_sort.comp");
pub const RLE: &str = include_str!("rle.comp");
pub const LRU_SCORE: &str = include_str!("lru_score.comp");
pub const CLASSIFY_KEYS: &str = include_str!("classify_keys.comp");
pub const EXCLUSIVE_SCAN: &str = include_str!("exclusive_scan.comp");
pub const ASSIGN_NEW_TARGETS: &str = include_str!("assign_new_targets.comp");
pub const FINALIZE_CURSORS: &str = include_str!("finalize_cursors.comp");
pub const SCAN_BLOCK_SUMS: &str = include_str!("scan_block_sums.comp");
pub const ADD_BLOCK_PREFIX: &str = include_str!("add_block_prefix.comp");
pub const DOWNSAMPLE: &str = include_str!("downsample.comp");

pub fn source_with_common(body: &str) -> String {
    let body_without_version = body
        .trim_start()
        .strip_prefix("#version 450")
        .map(str::trim_start)
        .unwrap_or(body);
    format!("{}\n{}", COMMON, body_without_version)
}
