# Metal GPU Timing Implementation Plan

This document outlines how to implement GPU timing measurements for Metal, following the same pattern as CUDA and Vulkan implementations.

## Overview

The goal is to measure GPU execution time for `planes_metal` and `add_to_map` operations using Metal's native timing APIs, integrating with the existing `RuntimeEvaluator`.

## Metal Timing Approaches

Metal provides several ways to measure GPU execution time:

### Option 1: GPUStartTime / GPUEndTime (Recommended)

Metal command buffers have `GPUStartTime` and `GPUEndTime` properties (available on macOS 10.15+):

```swift
let cb = device.commandQueue().commandBuffer()
// ... dispatch compute work ...
cb.commit()
cb.waitUntilCompleted()

let gpuTime = cb.GPUEndTime() - cb.GPUStartTime()
```

**Pros:**
- Native Metal API
- Minimal overhead
- Accurate GPU timing

**Cons:**
- Requires `waitUntilCompleted()` to get timing, or checking status
- Not available on older macOS versions (< 10.15)

### Option 2: Counter Sampling (macOS 10.15+)

Use `MTLCounterSet` for detailed profiling:

```swift
let counterSet = device.counterSets?.first { $0.name == "Timestamp" }
// ... create counter sample buffer ...
```

**Pros:**
- More detailed timing
- Can measure specific pipeline stages

**Cons:**
- More complex to set up
- May have limited availability

### Option 3: CPU Timing with Synchronization (Fallback)

If GPU timing APIs are unavailable:

```swift
let start = CACurrentMediaTime()
let cb = device.commandQueue().commandBuffer()
// ... dispatch compute work ...
cb.commit()
cb.waitUntilCompleted()
let elapsed = CACurrentMediaTime() - start
```

**Pros:**
- Works everywhere

**Cons:**
- Includes CPU overhead
- Less accurate

## Recommended Implementation

Use **Option 1** (GPUStartTime/GPUEndTime) as primary, with **Option 3** as fallback for older macOS.

## Implementation Steps

### 1. Add Timing State to MetalMapState

In `pelorus_metal/src/map.rs`:

```rust
pub struct MetalMapState {
    // ... existing fields ...
    
    // Timing state
    pub planes_command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    pub add_to_map_command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    pub pending_planes_timing: std::sync::atomic::AtomicBool,
    pub pending_add_to_map_timing: std::sync::atomic::AtomicBool,
    pub gpu_timing_available: bool, // Check if GPUStartTime/GPUEndTime available
}
```

### 2. Initialize Timing State

In `init_map_impl`:

```rust
// Check if GPU timing is available (macOS 10.15+)
let gpu_timing_available = {
    // Check if MTLCommandBuffer has GPUStartTime/EndTime methods
    // This can be done via respondsToSelector:
    // [cb respondsToSelector:@selector(GPUStartTime)]
    true // Assume available on modern systems
};

MetalMapState {
    // ... existing fields ...
    planes_command_buffer: None,
    add_to_map_command_buffer: None,
    pending_planes_timing: std::sync::atomic::AtomicBool::new(false),
    pending_add_to_map_timing: std::sync::atomic::AtomicBool::new(false),
    gpu_timing_available,
}
```

### 3. Record Command Buffers in planes_metal_impl

In `pelorus_metal/src/map.rs`, around line 828:

```rust
let cb = stream.queue.commandBuffer().expect("cb");
let enc = cb.computeCommandEncoder().expect("enc");

// ... dispatch plane_krnl ...

enc.endEncoding();
cb.commit();

// Store command buffer for timing
let mut guard = get_state();
let state = guard.as_mut().expect("Map not initialised");
state.planes_command_buffer = Some(cb.clone());
state.pending_planes_timing.store(true, Ordering::Release);
```

### 4. Record Command Buffers in add_to_map_impl

Similar changes in `add_to_map_impl`:

```rust
// After building voxels and hashmap rebuild...
let cb = stream.queue.commandBuffer().expect("cb");
// ... all compute dispatches ...
cb.commit();

let mut guard = get_state();
let state = guard.as_mut().expect("Map not initialised");
state.add_to_map_command_buffer = Some(cb.clone());
state.pending_add_to_map_timing.store(true, Ordering::Release);
```

### 5. Add Retrieval Function

In `pelorus_metal/src/lib.rs`:

```rust
use objc2::runtime::AnyObject;
use objc2_metal::MTLCommandBufferStatus;

/// Try to retrieve GPU timing measurements from previous operations.
/// Returns (planes_ms, add_to_map_ms) if timing results are available.
/// This is non-blocking - returns None if command buffers haven't completed.
pub fn try_get_gpu_timings() -> Option<(f64, f64)> {
    let mut guard = get_state();
    let state = guard.as_mut()?;
    
    let mut planes_ms = None;
    let mut add_to_map_ms = None;

    // Check planes timing
    if state.pending_planes_timing.load(Ordering::Acquire) {
        if let Some(ref cb) = state.planes_command_buffer {
            let status = cb.status();
            if status == MTLCommandBufferStatus::Completed {
                if state.gpu_timing_available {
                    // Use GPU timing (more accurate)
                    let start = cb.GPUStartTime();
                    let end = cb.GPUEndTime();
                    let elapsed = (end - start) * 1000.0; // sec → ms
                    planes_ms = Some(elapsed);
                }
                state.planes_command_buffer = None;
                state.pending_planes_timing.store(false, Ordering::Release);
            }
        }
    }

    // Check add_to_map timing
    if state.pending_add_to_map_timing.load(Ordering::Acquire) {
        if let Some(ref cb) = state.add_to_map_command_buffer {
            let status = cb.status();
            if status == MTLCommandBufferStatus::Completed {
                if state.gpu_timing_available {
                    let start = cb.GPUStartTime();
                    let end = cb.GPUEndTime();
                    let elapsed = (end - start) * 1000.0;
                    add_to_map_ms = Some(elapsed);
                }
                state.add_to_map_command_buffer = None;
                state.pending_add_to_map_timing.store(false, Ordering::Release);
            }
        }
    }

    match (planes_ms, add_to_map_ms) {
        (Some(p), Some(a)) => Some((p, a)),
        (Some(p), None) => Some((p, 0.0)),
        (None, Some(a)) => Some((0.0, a)),
        (None, None) => None,
    }
}
```

### 6. Wire Up in registration.rs

Add the Metal timing retrieval (already done for CUDA/Vulkan, need to add Metal):

```rust
#[cfg(all(not(feature = "cuda"), not(feature = "vulkan"), feature = "metal"))]
if let Some((planes_ms, add_to_map_ms)) = pelorus_metal::try_get_gpu_timings() {
    if planes_ms > 0.0 {
        eval.add_external_measurement("ekf_correct/model/planes/gpu", planes_ms);
    }
    if add_to_map_ms > 0.0 {
        eval.add_external_measurement("map_update/gpu", add_to_map_ms);
    }
}
```

### 7. Add Cleanup

In `cleanup_map_impl`:

```rust
pub fn cleanup_map_impl() {
    let mut guard = get_state();
    if let Some(state) = guard.as_mut() {
        // Clear command buffers (they will be released when dropped)
        state.planes_command_buffer = None;
        state.add_to_map_command_buffer = None;
    }
    *guard = None;
}
```

## Platform Considerations

1. **macOS 10.15+**: Full GPU timing support via `GPUStartTime`/`GPUEndTime`
2. **macOS < 10.15**: Fallback to CPU timing or no timing
3. **iOS**: Similar APIs available (check availability)

## Testing

1. Run with `output.terminal.debug = true`
2. Verify GPU timings appear in `eval.render()` output
3. Compare timing values to CPU-side measurements for sanity check
4. Verify no added latency (operations still async)

## Notes

- Metal command buffers are lightweight - keeping references for timing should not add significant overhead
- The `status` check is non-blocking - returns immediately if not completed
- `GPUStartTime`/`GPUEndTime` are only valid after command buffer completes
- Consider adding a timeout or fallback for very long-running operations

## Related Files

- `pelorus_metal/src/lib.rs` - Public API and timing function
- `pelorus_metal/src/map.rs` - State management and kernel dispatch
- `pelorus_core/src/timing.rs` - RuntimeEvaluator with `add_external_measurement`
- `pelorus_reg/src/registration.rs` - Timing retrieval integration
