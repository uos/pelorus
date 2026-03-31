pub const COMMON: &str = include_str!("common.comp");
pub const HASHMAP: &str = include_str!("hashmap.comp");
pub const LOCALMAP: &str = include_str!("localmap.comp");
pub const REORDER: &str = include_str!("reorder.comp");
pub const PLANES: &str = include_str!("planes.comp");
pub const RLE: &str = include_str!("rle.comp");
pub const ASSIGN_INDICES: &str = include_str!("assign_indices.comp");
pub const LRU_SCORE: &str = include_str!("lru_score.comp");

pub fn source_with_common(body: &str) -> String {
    let body_without_version = body
        .trim_start()
        .strip_prefix("#version 450")
        .map(str::trim_start)
        .unwrap_or(body);
    format!("{}\n{}", COMMON, body_without_version)
}
