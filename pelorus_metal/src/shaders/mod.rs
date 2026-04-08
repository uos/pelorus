pub const COMMON: &str = include_str!("common.metal");
pub const HASHMAP: &str = include_str!("hashmap.metal");
pub const LOCALMAP: &str = include_str!("localmap.metal");
pub const PLANES: &str = include_str!("planes.metal");
pub const BITONIC: &str = include_str!("bitonic.metal");
pub const RADIX_SORT: &str = include_str!("radix_sort.metal");

pub fn full_source() -> String {
    // Order matters
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        COMMON, HASHMAP, LOCALMAP, PLANES, BITONIC, RADIX_SORT
    )
}
