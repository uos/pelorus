use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLLibraryOptimizationLevel,
    MTLMathMode, MTLSize,
};
use std::sync::OnceLock;

use crate::shaders;

/// All compiled compute pipeline states (one per kernel).
pub struct Pipelines {
    pub hashmap_init: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub calculate_keys: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub reorder_points: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub rebuild_hashmap_keyed: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub build_voxels: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub rle: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub assign_indices: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub lru_score: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub plane_krnl: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub radix_zero_hist: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub radix_block_histogram: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub radix_prefix_blocks: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub radix_bucket_offsets: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub radix_scatter_pairs: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub downsample_build_keys: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub downsample_rle: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub downsample_medoids: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub mr_query: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub mr_cache_fits: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub mr_table_init: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub mr_table_build: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for Pipelines {}
unsafe impl Sync for Pipelines {}

impl Pipelines {
    pub fn compile(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let source = shaders::full_source();
        let ns_source = NSString::from_str(&source);

        let opts = MTLCompileOptions::new();
        // Enable optimizations in release builds
        if !cfg!(debug_assertions) {
            opts.setMathMode(MTLMathMode::Safe);
            opts.setOptimizationLevel(MTLLibraryOptimizationLevel::Default);
        } else {
            opts.setOptimizationLevel(MTLLibraryOptimizationLevel::Default);
        }

        let lib = device
            .newLibraryWithSource_options_error(&ns_source, Some(&opts))
            .expect("Metal shader compilation failed");

        let make = |name: &str| -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
            let ns_name = NSString::from_str(name);
            let func = lib
                .newFunctionWithName(&ns_name)
                .unwrap_or_else(|| panic!("get_function({name}) failed"));

            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("pipeline for {name} failed: {e:?}"))
        };

        Self {
            hashmap_init: make("hashmap_init"),
            calculate_keys: make("calculate_keys"),
            reorder_points: make("reorder_points"),
            rebuild_hashmap_keyed: make("rebuild_hashmap_keyed"),
            build_voxels: make("build_voxels"),
            rle: make("rle"),
            assign_indices: make("assign_indices"),
            lru_score: make("lru_score"),
            plane_krnl: make("plane_krnl"),
            radix_zero_hist: make("radix_zero_hist"),
            radix_block_histogram: make("radix_block_histogram"),
            radix_prefix_blocks: make("radix_prefix_blocks"),
            radix_bucket_offsets: make("radix_bucket_offsets"),
            radix_scatter_pairs: make("radix_scatter_pairs"),
            downsample_build_keys: make("downsample_build_keys"),
            downsample_rle: make("downsample_rle"),
            downsample_medoids: make("downsample_medoids"),
            mr_query: make("mr_query"),
            mr_cache_fits: make("mr_cache_fits"),
            mr_table_init: make("mr_table_init"),
            mr_table_build: make("mr_table_build"),
        }
    }
}

static PIPELINES: OnceLock<Pipelines> = OnceLock::new();

pub fn get_pipelines(device: &ProtocolObject<dyn MTLDevice>) -> &'static Pipelines {
    PIPELINES.get_or_init(|| Pipelines::compile(device))
}

/// Preferred threadgroup width for 1-D dispatches.
pub const THREADS_PER_GROUP: u64 = 256;

pub fn grid1d(n: usize) -> MTLSize {
    MTLSize {
        width: n as _,
        height: 1,
        depth: 1,
    }
}

pub fn group1d() -> MTLSize {
    MTLSize {
        width: THREADS_PER_GROUP as _,
        height: 1,
        depth: 1,
    }
}
