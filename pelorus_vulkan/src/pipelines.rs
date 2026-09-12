use crate::context::VulkanContext;
use crate::shaders;
use ash::{Device, vk};
use bytemuck::{Pod, Zeroable};
use std::ffi::CString;
use std::mem::size_of;
use std::sync::Arc;

pub(crate) const EMPTY_KEY: i32 = i32::MIN;
pub(crate) const THREADS_PER_GROUP: u32 = 256;
pub(crate) const DESCRIPTOR_BINDING_COUNT: u32 = 18;
pub(crate) const MAX_DISPATCHES_PER_SESSION: u32 = 1024;
/// How long `submit` polls the completion fence before blocking. Sized so a typical
/// submission is caught by the poll, without burning a core when one runs long.
pub(crate) const SUBMIT_SPIN_ITERATIONS: u32 = 20_000;

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct GpuVoxelDist {
    pub(crate) lock_flag: i32,
    pub(crate) transform: [f32; 9],
    pub(crate) point_count: u32,
    pub(crate) mean: [f32; 3],
    pub(crate) covariance: [f32; 9],
    pub(crate) last_scan: u32,
}

const _: () = assert!(size_of::<GpuVoxelDist>() == 96);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct PushConstants {
    pub(crate) pose_quat: [f32; 4],
    pub(crate) pose_trans: [f32; 3],
    pub(crate) _pad0: f32,
    pub(crate) lidar_to_imu_quat: [f32; 4],
    pub(crate) lidar_to_imu_pos: [f32; 3],
    pub(crate) _pad1: f32,
    pub(crate) resolution: f32,
    pub(crate) scan_number: u32,
    pub(crate) sensitivity: f32,
    pub(crate) min_dist: f32,
    pub(crate) unique_key_count: u32,
    pub(crate) table_cap: u32,
    pub(crate) dist_cap: u32,
    pub(crate) nbtype: u32,
    pub(crate) sphericity_threshold: f32,
    pub(crate) j: u32,
    pub(crate) k: u32,
    pub(crate) n: u32,
    pub(crate) cap: u32,
    pub(crate) valid_size: u32,
}

impl Default for PushConstants {
    fn default() -> Self {
        Self {
            pose_quat: [0.0, 0.0, 0.0, 1.0],
            pose_trans: [0.0; 3],
            _pad0: 0.0,
            lidar_to_imu_quat: [0.0, 0.0, 0.0, 1.0],
            lidar_to_imu_pos: [0.0; 3],
            _pad1: 0.0,
            resolution: 1.0,
            scan_number: 0,
            sensitivity: 0.1,
            min_dist: 0.0,
            unique_key_count: 0,
            table_cap: 0,
            dist_cap: 0,
            nbtype: 26,
            sphericity_threshold: 0.5,
            j: 0,
            k: 0,
            n: 0,
            cap: 0,
            valid_size: 0,
        }
    }
}

pub(crate) struct VulkanComputePipelines {
    pub(crate) context: Arc<VulkanContext>,
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) hashmap_init: vk::Pipeline,
    pub(crate) rebuild_hashmap_keyed: vk::Pipeline,
    pub(crate) calculate_keys: vk::Pipeline,
    pub(crate) reorder_points: vk::Pipeline,
    pub(crate) build_voxels: vk::Pipeline,
    pub(crate) plane_krnl: vk::Pipeline,
    pub(crate) compact_planes: vk::Pipeline,
    pub(crate) plane_count: vk::Pipeline,
    pub(crate) portable_sort: vk::Pipeline,
    pub(crate) rle: vk::Pipeline,
    pub(crate) lru_score: vk::Pipeline,
    pub(crate) classify_keys: vk::Pipeline,
    pub(crate) exclusive_scan: vk::Pipeline,
    pub(crate) assign_new_targets: vk::Pipeline,
    pub(crate) finalize_cursors: vk::Pipeline,
    pub(crate) scan_block_sums: vk::Pipeline,
    pub(crate) add_block_prefix: vk::Pipeline,
    pub(crate) downsample_build_keys: vk::Pipeline,
    pub(crate) downsample_medoids: vk::Pipeline,
    pub(crate) downsample_timestamp_keys: vk::Pipeline,
    shader_modules: Vec<vk::ShaderModule>,
}

impl Drop for VulkanComputePipelines {
    fn drop(&mut self) {
        if !self.context.is_alive() {
            return;
        }
        unsafe {
            let device = &self.context.device;
            device.destroy_pipeline(self.hashmap_init, None);
            device.destroy_pipeline(self.rebuild_hashmap_keyed, None);
            device.destroy_pipeline(self.calculate_keys, None);
            device.destroy_pipeline(self.reorder_points, None);
            device.destroy_pipeline(self.build_voxels, None);
            device.destroy_pipeline(self.plane_krnl, None);
            device.destroy_pipeline(self.compact_planes, None);
            device.destroy_pipeline(self.plane_count, None);
            device.destroy_pipeline(self.portable_sort, None);
            device.destroy_pipeline(self.rle, None);
            device.destroy_pipeline(self.lru_score, None);
            device.destroy_pipeline(self.classify_keys, None);
            device.destroy_pipeline(self.exclusive_scan, None);
            device.destroy_pipeline(self.assign_new_targets, None);
            device.destroy_pipeline(self.finalize_cursors, None);
            device.destroy_pipeline(self.scan_block_sums, None);
            device.destroy_pipeline(self.add_block_prefix, None);
            device.destroy_pipeline(self.downsample_build_keys, None);
            device.destroy_pipeline(self.downsample_medoids, None);
            device.destroy_pipeline(self.downsample_timestamp_keys, None);
            for module in self.shader_modules.drain(..) {
                device.destroy_shader_module(module, None);
            }
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

#[cfg(feature = "vulkan_enable")]
pub(crate) fn compile_shader(
    compiler: &mut shaderc::Compiler,
    source: &str,
    name: &str,
    defines: &[(&str, Option<&str>)],
) -> Vec<u32> {
    let mut options = shaderc::CompileOptions::new().expect("failed creating shaderc options");
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_2 as u32,
    );
    for (key, value) in defines {
        options.add_macro_definition(key, *value);
    }
    let artifact = compiler
        .compile_into_spirv(
            source,
            shaderc::ShaderKind::Compute,
            name,
            "main",
            Some(&options),
        )
        .unwrap_or_else(|e| panic!("failed compiling {name}: {e}"));
    artifact.as_binary().to_vec()
}

fn create_shader_module(device: &Device, code: &[u32]) -> vk::ShaderModule {
    let ci = vk::ShaderModuleCreateInfo::default().code(code);
    unsafe {
        device
            .create_shader_module(&ci, None)
            .expect("failed creating shader module")
    }
}

fn create_compute_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    module: vk::ShaderModule,
) -> vk::Pipeline {
    let entry = CString::new("main").expect("invalid entry point");
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(entry.as_c_str());
    let ci = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);
    unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&ci), None)
            .expect("failed creating compute pipeline")[0]
    }
}

#[cfg(feature = "vulkan_enable")]
pub(crate) fn build_pipelines(context: Arc<VulkanContext>) -> Arc<VulkanComputePipelines> {
    let bindings: Vec<_> = (0..DESCRIPTOR_BINDING_COUNT)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let descriptor_set_layout = unsafe {
        context
            .device
            .create_descriptor_set_layout(&dsl_ci, None)
            .expect("failed creating descriptor set layout")
    };

    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(size_of::<PushConstants>() as u32);
    let pl_ci = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_set_layout))
        .push_constant_ranges(std::slice::from_ref(&push_range));
    let pipeline_layout = unsafe {
        context
            .device
            .create_pipeline_layout(&pl_ci, None)
            .expect("failed creating pipeline layout")
    };

    let mut compiler = shaderc::Compiler::new().expect("failed creating shader compiler");

    let hashmap_src = shaders::source_with_common(shaders::HASHMAP);
    let localmap_src = shaders::source_with_common(shaders::LOCALMAP);
    let reorder_src = shaders::source_with_common(shaders::REORDER);
    let planes_src = shaders::source_with_common(shaders::PLANES);
    let compact_planes_src = shaders::source_with_common(shaders::COMPACT_PLANES);
    let plane_count_src = shaders::source_with_common(shaders::PLANE_COUNT);
    let portable_sort_src = shaders::source_with_common(shaders::PORTABLE_SORT);
    let rle_src = shaders::source_with_common(shaders::RLE);
    let lru_score_src = shaders::source_with_common(shaders::LRU_SCORE);
    let classify_keys_src = shaders::source_with_common(shaders::CLASSIFY_KEYS);
    let exclusive_scan_src = shaders::source_with_common(shaders::EXCLUSIVE_SCAN);
    let assign_new_targets_src = shaders::source_with_common(shaders::ASSIGN_NEW_TARGETS);
    let finalize_cursors_src = shaders::source_with_common(shaders::FINALIZE_CURSORS);
    let scan_block_sums_src = shaders::source_with_common(shaders::SCAN_BLOCK_SUMS);
    let add_block_prefix_src = shaders::source_with_common(shaders::ADD_BLOCK_PREFIX);
    let downsample_src = shaders::source_with_common(shaders::DOWNSAMPLE);

    let shader_defs = [
        ("hashmap_init", &hashmap_src, vec![("HASHMAP_INIT", None)]),
        (
            "rebuild_hashmap_keyed",
            &hashmap_src,
            vec![("REBUILD_HASHMAP_KEYED", None)],
        ),
        (
            "calculate_keys",
            &localmap_src,
            vec![("CALCULATE_KEYS", None)],
        ),
        ("reorder_points", &reorder_src, vec![]),
        ("build_voxels", &localmap_src, vec![("BUILD_VOXELS", None)]),
        ("plane_krnl", &planes_src, vec![]),
        ("compact_planes", &compact_planes_src, vec![]),
        ("plane_count", &plane_count_src, vec![]),
        ("portable_sort", &portable_sort_src, vec![]),
        ("rle", &rle_src, vec![]),
        ("lru_score", &lru_score_src, vec![]),
        ("classify_keys", &classify_keys_src, vec![]),
        ("exclusive_scan", &exclusive_scan_src, vec![]),
        ("assign_new_targets", &assign_new_targets_src, vec![]),
        ("finalize_cursors", &finalize_cursors_src, vec![]),
        ("scan_block_sums", &scan_block_sums_src, vec![]),
        ("add_block_prefix", &add_block_prefix_src, vec![]),
        (
            "downsample_build_keys",
            &downsample_src,
            vec![("BUILD_KEYS", None)],
        ),
        (
            "downsample_medoids",
            &downsample_src,
            vec![("MEDOIDS", None)],
        ),
        (
            "downsample_timestamp_keys",
            &downsample_src,
            vec![("TIMESTAMP_KEYS", None)],
        ),
    ];

    let mut modules = Vec::with_capacity(shader_defs.len());
    let mut pipelines = Vec::with_capacity(shader_defs.len());
    for (name, source, defines) in shader_defs {
        let spirv = compile_shader(&mut compiler, source, name, &defines);
        let module = create_shader_module(&context.device, &spirv);
        let pipeline = create_compute_pipeline(&context.device, pipeline_layout, module);
        modules.push(module);
        pipelines.push(pipeline);
    }

    Arc::new(VulkanComputePipelines {
        context,
        descriptor_set_layout,
        pipeline_layout,
        hashmap_init: pipelines[0],
        rebuild_hashmap_keyed: pipelines[1],
        calculate_keys: pipelines[2],
        reorder_points: pipelines[3],
        build_voxels: pipelines[4],
        plane_krnl: pipelines[5],
        compact_planes: pipelines[6],
        plane_count: pipelines[7],
        portable_sort: pipelines[8],
        rle: pipelines[9],
        lru_score: pipelines[10],
        classify_keys: pipelines[11],
        exclusive_scan: pipelines[12],
        assign_new_targets: pipelines[13],
        finalize_cursors: pipelines[14],
        scan_block_sums: pipelines[15],
        add_block_prefix: pipelines[16],
        downsample_build_keys: pipelines[17],
        downsample_medoids: pipelines[18],
        downsample_timestamp_keys: pipelines[19],
        shader_modules: modules,
    })
}

#[cfg(not(feature = "vulkan_enable"))]
pub(crate) fn build_pipelines(_context: Arc<VulkanContext>) -> Arc<VulkanComputePipelines> {
    panic!("vulkan pipeline initialization requires feature `vulkan_enable`")
}
