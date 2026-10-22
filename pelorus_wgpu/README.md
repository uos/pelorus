# Experimental WebGPU correspondence backend

`pelorus_wgpu` accelerates single- and multiresolution point-to-plane
correspondence search using portable WGSL compute. It is a hybrid backend: deskewing, downsampling,
map updates and the f64 iterated filter still run on the CPU.

The CPU map and deskewed cloud are uploaded once per registered scan and reused
across filter iterations. Each query returns one 16-byte plane per input point,
and CPU compaction preserves point order. Original voxel keys are exported (full
grid coordinates for the pyramid, hashes for single-resolution maps) rather
than reconstructed from distribution means. Buffers grow and are reused.
There is no global device or map. Shader compilation errors, device loss,
readback failures and oversized inputs fall back to the current CPU map.

## Browser

Build with `web/build.sh` and serve `web/` from localhost or HTTPS. The page
uses WebGPU when the browser offers a hardware adapter and the WebAssembly CPU
path otherwise. Both preserve the recording's map settings. The four-level pyramid uses
the finest valid patch, falling back to coarser levels for sparse or degenerate
support. The shader applies the CPU thickness, spread, finite-patch and residual
gates, including full grid-key comparisons at each level. The multiresolution
map amplifies single rounding differences, so the shader reproduces the CPU's
arithmetic: the same evaluation order, no fused multiply-add, correctly rounded
division and square root, and the f64 eigensolver in double-float arithmetic.
Per query, nearly all planes are bit-identical to the CPU. The rest differ by
one ulp in an eigenvector component. This relies on a fused `fma`, which
D3D12 may not provide. Map updates and eviction remain shared CPU code.
The status view reports the actual backend and logs any CPU fallback.
Browser initialization times out after 15 seconds and releases the odometer
borrow before continuing on CPU, including when adapter discovery never resolves.

The SDK feature is `webgpu`. Call `enable_webgpu().await`, then
`next_scan_async().await`. WASM exports are `enableWebGpu()`, `nextScanAsync()`
and `webGpuStatus()`. The synchronous APIs retain their CPU/native behavior.
The SDK supports the multiresolution pyramid and single-resolution Nearby26 maps.
It rejects KD-tree and other single-resolution neighborhoods without changing
the caller's configuration.
The browser worker serializes commands and awaits each GPU readback.

## Verification

```sh
cargo test -p pelorus_wgpu --lib
cargo test -p pelorus_wgpu --lib -- --ignored --nocapture
cargo test -p pelorus_reg --no-default-features --features webgpu --test webgpu -- --ignored
cargo check -p pelorus_web --features webgpu --target wasm32-unknown-unknown
```

The ignored tests require a compute adapter. They check CPU/GPU plane agreement,
pose changes, buffer growth/reuse, empty inputs/maps, rejected points, unsupported
maps, sparse/coarse support, finite patch bounds, degenerate and thick geometry,
map eviction, hash collisions, iterated registration and continuation after
simulated device loss. Software Vulkan is useful for correctness but not a
hardware speed claim.

Open `web/benchmark.html` on the same server to replay a demo on CPU and WebGPU
sequentially with identical map settings and no 3D renderer. Choose
**Multiresolution** (default) or **Single resolution** in the benchmark.
It reports full replay wall time, scan mean/p50/p95, and maximum position
difference at matched timestamps. It refuses a speedup comparison if WebGPU
falls back to CPU. Download and initialization are excluded. Extracting and
transferring display points still contribute to replay wall time. Repeat runs
on the target hardware before drawing conclusions. Position agreement with
CPU is not a ground-truth accuracy evaluation.

For comparisons in the regular demo, use the same browser, GPU and view layout.
Rendering and extracting display points can dominate the displayed frequency.
This prototype still uploads the full map per scan and reads back per iteration.
Moving map maintenance and measurement reductions to the GPU is future work.
