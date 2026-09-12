use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

#[cfg(feature = "vulkan_enable")]
#[test]
fn downsample_shaders_compile() {
    let mut compiler = shaderc::Compiler::new().expect("failed creating shader compiler");
    let downsample_src = crate::shaders::source_with_common(crate::shaders::DOWNSAMPLE);

    let cases = [
        (
            "downsample_build_keys",
            downsample_src.as_str(),
            vec![("BUILD_KEYS", None)],
        ),
        (
            "downsample_medoids",
            downsample_src.as_str(),
            vec![("MEDOIDS", None)],
        ),
        (
            "downsample_timestamp_keys",
            downsample_src.as_str(),
            vec![("TIMESTAMP_KEYS", None)],
        ),
    ];

    for (name, source, defines) in cases {
        crate::pipelines::compile_shader(&mut compiler, source, name, &defines);
    }
}

#[cfg(feature = "vulkan_enable")]
#[test]
fn portable_sort_matches_unsigned_key_value_order() {
    let stream = crate::stream::Stream::new();
    if !stream.is_vulkan() {
        return;
    }
    let crate::stream::StreamBackend::Vulkan(context) = &stream.backend else {
        unreachable!();
    };

    let count = 100_000usize;
    let capacity = count.next_power_of_two();
    let source_keys: Vec<i32> = (0..count)
        .map(|i| {
            let value = (i.wrapping_mul(2654435761usize) ^ (i / 7)) as u32;
            (value % 97).wrapping_mul(0x01010101) as i32
        })
        .collect();
    let source_values: Vec<u32> = (0..count as u32).collect();
    let mut keys = crate::buffer::VulkanBuffer::new(&stream, size_of::<i32>(), capacity);
    let mut values = crate::buffer::VulkanBuffer::new(&stream, size_of::<u32>(), capacity);
    keys.upload(&stream, &source_keys);
    values.upload(&stream, &source_values);
    let dummy = crate::buffer::VulkanBuffer::new(&stream, size_of::<u32>(), 1);

    let compute = crate::pipelines::build_pipelines(Arc::clone(context));
    let resources = Mutex::new(crate::compute_session::ComputeCommandResources::new(
        context, &compute,
    ));
    let mut session = crate::compute_session::ComputeCommandSession::begin(
        context,
        &resources,
        crate::map::vk_buffer_handle(&dummy),
    );
    crate::compute_session::record_portable_sort(
        &mut session,
        &compute,
        crate::map::vk_buffer_handle(&dummy),
        crate::map::vk_buffer_handle(&keys),
        crate::map::vk_buffer_handle(&values),
        count as u32,
        capacity as u32,
    );
    session.submit();

    let sorted_keys: Vec<i32> = keys.download(&stream);
    let sorted_values: Vec<u32> = values.download(&stream);
    let mut expected: Vec<(i32, u32)> = source_keys.into_iter().zip(source_values).collect();
    expected.sort_unstable_by_key(|(key, value)| (*key as u32, *value));
    let actual: Vec<(i32, u32)> = sorted_keys.into_iter().zip(sorted_values).collect();
    assert_eq!(actual, expected);
}

const EMPTY_KEY: i32 = i32::MIN;
const INVALID_INDEX: u32 = u32::MAX;

fn assign_indices_cuda_style(
    unique_keys: &[i32],
    existing: &HashMap<i32, u32>,
    dist_cap: u32,
    add_cursor_start: u32,
    lru_mapping: &[u32],
) -> (Vec<u32>, Vec<u32>, Vec<i32>, u32, u32) {
    let mut target_indices = Vec::with_capacity(unique_keys.len());
    let mut update_flags = Vec::with_capacity(unique_keys.len());
    let mut dist_keys = vec![EMPTY_KEY; dist_cap as usize];
    let mut add_cursor = add_cursor_start;
    let mut lru_cursor = 0u32;

    for &key in unique_keys {
        if key == EMPTY_KEY {
            target_indices.push(INVALID_INDEX);
            update_flags.push(0);
            continue;
        }

        if let Some(&idx) = existing.get(&key) {
            target_indices.push(idx);
            update_flags.push(1);
            continue;
        }

        let old_add = add_cursor;
        add_cursor = add_cursor.saturating_add(1);

        if old_add < dist_cap {
            target_indices.push(old_add);
            update_flags.push(0);
            dist_keys[old_add as usize] = key;
            continue;
        }

        let lru_idx = lru_cursor;
        lru_cursor = lru_cursor.saturating_add(1);
        if lru_idx < dist_cap {
            let victim = lru_mapping[lru_idx as usize];
            target_indices.push(victim);
            update_flags.push(0);
            dist_keys[victim as usize] = key;
        } else {
            target_indices.push(INVALID_INDEX);
            update_flags.push(0);
        }
    }

    (
        target_indices,
        update_flags,
        dist_keys,
        add_cursor,
        lru_cursor,
    )
}

#[test]
fn map_full_assign_stops_when_lru_candidates_exhausted() {
    let existing = HashMap::new();
    let lru_mapping = vec![2u32, 1u32, 0u32, 3u32];
    let (targets, flags, dist_keys, add_cursor, lru_cursor) =
        assign_indices_cuda_style(&[11, 22, 33, 44, 55, 66], &existing, 4, 4, &lru_mapping);

    assert_eq!(
        targets,
        vec![2, 1, 0, 3, INVALID_INDEX, INVALID_INDEX],
        "replacements must not wrap and overwrite earlier victims in a single scan"
    );
    assert_eq!(flags, vec![0, 0, 0, 0, 0, 0]);
    assert_eq!(add_cursor, 10);
    assert_eq!(lru_cursor, 6);
    assert_eq!(dist_keys[2], 11);
    assert_eq!(dist_keys[1], 22);
    assert_eq!(dist_keys[0], 33);
    assert_eq!(dist_keys[3], 44);
}

#[test]
fn map_full_assign_resets_lru_consumption_each_scan() {
    let existing = HashMap::new();
    let lru_mapping = vec![3u32, 2u32, 1u32, 0u32];

    let (first_targets, _, _, _, first_lru_cursor) =
        assign_indices_cuda_style(&[10, 20], &existing, 4, 4, &lru_mapping);
    let (second_targets, _, _, _, second_lru_cursor) =
        assign_indices_cuda_style(&[30, 40], &existing, 4, 4, &lru_mapping);

    assert_eq!(first_targets, vec![3, 2]);
    assert_eq!(second_targets, vec![3, 2]);
    assert_eq!(first_lru_cursor, 2);
    assert_eq!(second_lru_cursor, 2);
}
