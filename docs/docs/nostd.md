# Embedded

The Rust library builds without the standard library. It needs an allocator and nothing else, so Pelorus runs on microcontrollers and in kernels as well as on a desktop.

## Adding the crate

The crate is `pelorus_sdk`, imported as `pelorus`. Turn its default features off to drop std:

```sh
cargo add pelorus_sdk --no-default-features
```

The target needs atomic compare-and-swap. Every Cortex-M3 or newer, every RISC-V core with the A extension such as `riscv32imac`, and every 32-bit or 64-bit application processor has it. Cortex-M0 does not. RISC-V cores without the A extension work with a custom target, see [RISC-V without atomics](#risc-v-without-atomics). The binary needs an `#[global_allocator]`.

Pelorus computes in `f64` if the `f32` feature is disabled. The Cortex-M7 has a double-precision FPU, but the target `thumbv7em-none-eabihf` assumes the single-precision FPU of the Cortex-M4F and turns every `f64` operation into a software routine. Name the CPU in `.cargo/config.toml`:

```toml
[target.thumbv7em-none-eabihf]
rustflags = ["-C", "target-cpu=cortex-m7"]
```

Some cores only have a single-precision FPU. We can avoid slow software emulation by turning on the `f32` feature:

```sh
cargo add pelorus_sdk --no-default-features --features f32
```

The library then computes in `f32`. On the paper sequences, the `f32` build is as accurate as the `f64` build.

The odometry does not lose precision however far the robot drives, because Pelorus computes in a local frame that moves along with the robot in map cells. Only the world position needs care. `Scan::position` returns it as an `f32`, and far from the start an `f32` can only express coarse steps, about 1 mm at 10 km. If that matters, use `Scan::local_position` and `Scan::local_points` instead. They stay in the local frame at full precision. `Scan::origin_cells` and `Scan::cell_size` tell where that frame lies in the world, so the world position is the origin cells times the cell size plus the local position. Add them up where `f64` or integers are available, for example on the receiving side.

## Running the odometry

Without std, Pelorus cannot read [Ratslang](https://ratslang.steado.tech/docs/) configuration files. The `Odometer` takes a resolved `LioConfig` instead. `LioConfig::defaults()` is the configuration the embedded `default.rl` resolves to. Change its fields for your sensors before you create the odometer.

Create the odometer once and feed it every message the sensor driver receives:

```rust
use pelorus::{ImuMeasurement, LioConfig, Odometer, Real};

/// One message from the sensor driver.
pub enum Input {
    Imu(ImuMeasurement),
    Points {
        sec: i32,
        nanosec: u32,
        points: Vec<[f32; 5]>,
    },
}

pub fn run(
    mut receive: impl FnMut() -> Input,
    mut publish: impl FnMut((i32, u32), [Real; 3], [Real; 4]),
) -> Result<(), pelorus::Error> {
    let mut config = LioConfig::defaults();
    config.map_resolution = 0.5;
    let mut odometer = Odometer::from_config(config, false)?;

    loop {
        match receive() {
            Input::Imu(imu) => odometer.add_imu(imu),
            Input::Points { sec, nanosec, points } => {
                // Skip a scan while an older one still waits, so the queue cannot grow.
                if odometer.pending_scans() < 2 {
                    odometer.add_points_xyzit(sec, nanosec, &points);
                }
            }
        }
        // An IMU message can complete a waiting scan, so drain after every input.
        while let Some(scan) = odometer.next_scan() {
            publish(scan.timestamp(), scan.position(), scan.orientation());
        }
    }
}
```

The second argument of `from_config` is `false` because `add_points_xyzit` hands over generic points, with `t` as the nanosecond offset to the scan stamp. With `true`, Pelorus reads every cloud in the layout of the configured `lidar.vendor` instead. The default vendor `NoTimestamp` has no time field, so the scans would not be deskewed.

Everything runs on the calling thread:

- `add_imu` and `add_points_xyzit` only queue their input. A cloud is filtered and downsampled as it is added.
- `next_scan` takes the oldest scan whose IMU interval is complete, deskews it and registers it against the map. It returns once that is done.
- A scan is complete once IMU messages past its end have arrived, so it usually comes out a few IMU messages after its cloud went in.
- At the start, clouds that arrive during the IMU initialization are dropped and the first scans only build the map. `next_scan` returns nothing for about the first second.
- While `next_scan` works, the loop receives nothing. Buffer the network packets in the driver, for example in DMA rings, so none are lost during a registration.

## Bridging a configuration to the device

A configuration written in Ratslang can still reach the device. Resolve it on a host with std, where `pelorus::load_config` reads the file, and serialize the `LioConfig` with serde, for example with [postcard](https://docs.rs/postcard):

```rust
// On the host, with the std feature.
let config = pelorus::load_config(Some("pelorus.rl"))?;
std::fs::write("pelorus.bin", postcard::to_allocvec(&config)?)?;
```

```rust
// On the device.
let config: pelorus::LioConfig = postcard::from_bytes(include_bytes!("pelorus.bin"))?;
```

## RISC-V without atomics

Rust's targets for RISC-V cores without the A extension, such as `riscv32imc-unknown-none-elf`, have no atomic compare-and-swap, so the library does not build for them. A custom target solves this. It turns compare-and-swap and the other atomic read-modify-write operations into calls to `__sync` functions, which the firmware provides. On a single core, these functions only have to disable interrupts while they read and write.

The repository contains such a target for a core with the M and F extensions and without A and C, [`riscv32imf-unknown-none-elf.json`](https://codeberg.org/stelzo/pelorus/src/branch/main/pelorus_leszy_qemu/riscv32imf-unknown-none-elf.json). It passes floats in the FPU registers. Change `features` for other cores. A core without an FPU, for example, takes `+m,+c,+forced-atomics` with `llvm-abiname` set to `ilp32`. A custom target builds `core` and `alloc` from source, which needs nightly and `rust-src`. Without a C library, `compiler-builtins-mem` provides `memcpy` and its relatives:

```sh
rustup component add rust-src --toolchain nightly
cargo +nightly build --release -Zbuild-std=core,alloc \
  -Zbuild-std-features=compiler-builtins-mem -Zjson-target-spec \
  --target riscv32imf-unknown-none-elf.json
```

The linker lists the `__sync` functions it misses. With the `critical-section` crate and its implementation for the core, they look like this:

```rust
macro_rules! sync_functions {
    ($t:ty, $cas:ident, $add:ident, $sub:ident, $and:ident, $or:ident) => {
        #[unsafe(no_mangle)]
        extern "C" fn $cas(ptr: *mut $t, expected: $t, new: $t) -> $t {
            critical_section::with(|_| unsafe {
                let old = ptr.read_volatile();
                if old == expected {
                    ptr.write_volatile(new);
                }
                old
            })
        }
        sync_functions!(@op $t, $add, |a: $t, b: $t| a.wrapping_add(b));
        sync_functions!(@op $t, $sub, |a: $t, b: $t| a.wrapping_sub(b));
        sync_functions!(@op $t, $and, |a: $t, b: $t| a & b);
        sync_functions!(@op $t, $or, |a: $t, b: $t| a | b);
    };
    (@op $t:ty, $name:ident, $f:expr) => {
        #[unsafe(no_mangle)]
        extern "C" fn $name(ptr: *mut $t, value: $t) -> $t {
            critical_section::with(|_| unsafe {
                let old = ptr.read_volatile();
                ptr.write_volatile($f(old, value));
                old
            })
        }
    };
}

sync_functions!(u8, __sync_val_compare_and_swap_1, __sync_fetch_and_add_1,
    __sync_fetch_and_sub_1, __sync_fetch_and_and_1, __sync_fetch_and_or_1);
sync_functions!(u32, __sync_val_compare_and_swap_4, __sync_fetch_and_add_4,
    __sync_fetch_and_sub_4, __sync_fetch_and_and_4, __sync_fetch_and_or_4);
```
