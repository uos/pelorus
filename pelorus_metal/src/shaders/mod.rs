pub const COMMON: &str = include_str!("common.metal");
pub const HASHMAP: &str = include_str!("hashmap.metal");
pub const LOCALMAP: &str = include_str!("localmap.metal");
pub const PLANES: &str = include_str!("planes.metal");

/// Concatenate all shader sources into a single MSL compilation unit.
/// Order matters: common → hashmap → localmap → planes.
pub fn full_source() -> String {
    format!("{}\n{}\n{}\n{}", COMMON, HASHMAP, LOCALMAP, PLANES)
}
