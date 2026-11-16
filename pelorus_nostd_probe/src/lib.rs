//! Every entry point of the no_std library behind `extern "C"` functions, for
//! bare-metal targets. `mask nostd-check` links it for the Cortex-M4F and keeps
//! only what the entry points reach. That core has a single-precision FPU, so
//! each `f64` operation left in the library is a call into compiler-builtins.
//!
//! The configuration and every input arrive as opaque bytes, so no branch
//! folds away. On other targets the crate is empty.
#![cfg_attr(target_os = "none", no_std)]

#[cfg(target_os = "none")]
extern crate alloc;

#[cfg(target_os = "none")]
mod probe;
