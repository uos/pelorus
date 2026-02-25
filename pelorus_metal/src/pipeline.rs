use metal::{ComputePipelineState, Device, MTLSize};
use std::sync::OnceLock;

use crate::shaders;

/// All compiled compute pipeline states (one per kernel).
pub struct Pipelines {
    pub hashmap_init: ComputePipelineState,
    pub rebuild_hashmap_keyed: ComputePipelineState,
    pub build_voxels: ComputePipelineState,
    pub plane_krnl: ComputePipelineState,
}

unsafe impl Send for Pipelines {}
unsafe impl Sync for Pipelines {}

impl Pipelines {
    pub fn compile(device: &Device) -> Self {
        let source = shaders::full_source();
        let opts = metal::CompileOptions::new();
        let lib = device
            .new_library_with_source(&source, &opts)
            .expect("Metal shader compilation failed");

        let make = |name: &str| -> ComputePipelineState {
            let func = lib
                .get_function(name, None)
                .unwrap_or_else(|e| panic!("get_function({name}) failed: {e}"));
            device
                .new_compute_pipeline_state_with_function(&func)
                .unwrap_or_else(|e| panic!("pipeline for {name} failed: {e}"))
        };

        Self {
            hashmap_init: make("hashmap_init"),
            rebuild_hashmap_keyed: make("rebuild_hashmap_keyed"),
            build_voxels: make("build_voxels"),
            plane_krnl: make("plane_krnl"),
        }
    }
}

static PIPELINES: OnceLock<Pipelines> = OnceLock::new();

pub fn get_pipelines(device: &Device) -> &'static Pipelines {
    PIPELINES.get_or_init(|| Pipelines::compile(device))
}

// ── Dispatch helpers ─────────────────────────────────────────────────────────

/// Preferred threadgroup width for 1-D dispatches.
pub const THREADS_PER_GROUP: u64 = 256;

pub fn grid1d(n: usize) -> MTLSize {
    MTLSize { width: n as u64, height: 1, depth: 1 }
}

pub fn group1d() -> MTLSize {
    MTLSize { width: THREADS_PER_GROUP, height: 1, depth: 1 }
}
